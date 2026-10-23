#!/usr/bin/env python3
"""
Data Upload Manager - Upload data folders to remote server via SFTP.

This module provides functionality to monitor a directory for new folders
and upload them to a remote server via SFTP.

Author: Dong Jiulong
"""

import os
import sys
import time
import threading
from typing import Set, Optional, List, Dict, Any
from pathlib import Path

# Ensure project root is on PYTHONPATH when running as a script
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import paramiko
from paramiko import SSHClient, SFTPClient

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from utils.logger_config import logger


def load_config_from_hydra(config_name: str) -> Dict[str, Any]:
    """
    Load configuration from a Hydra config file.

    Args:
        config_name: Name of the config file (e.g., i2rt_data_collect_async)

    Returns:
        Dictionary containing the augmental_data_upload configuration
    """
    import os
    from hydra import initialize, compose
    from omegaconf import OmegaConf

    with initialize(config_path="../configs", version_base="1.3"):
        cfg = compose(
            config_name=config_name,
            overrides=[]
        )

        augmental_cfg = cfg.get("augmental_data_upload", {})

    logger.info(f"Loaded config from: {config_name}")
    return OmegaConf.to_container(augmental_cfg, resolve=True) if augmental_cfg else {}

class DataUploadManager:
    """
    Manages uploading of data folders to a remote server via SFTP.

    This class runs as a background thread that polls a data directory
    at regular intervals and uploads any new folders (not ending with 'temp')
    to a remote SFTP server.
    """

    def __init__(
        self,
        data_dir: str,
        remote_path: str,
        host: str,
        port: int,
        username: str,
        poll_interval: float,
        ssh_key_path: str,
    ) -> None:
        """
        Initialize the Data Upload Manager.

        Args:
            data_dir: Local directory to monitor for new folders
            host: SFTP server hostname
            port: SFTP server port
            username: SFTP username
            remote_path: Remote directory path on SFTP server
            poll_interval: Polling interval in seconds (default: 5.0)
            ssh_key_path: Path to SSH private key file
        """
        self.data_dir = data_dir
        self.host = host
        self.port = port
        self.username = username
        self.remote_path = remote_path
        self.poll_interval = poll_interval
        self.ssh_key_path = os.path.expanduser(ssh_key_path)

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._uploaded_folders: Set[str] = set()

        logger.info(f"DataUploadManager initialized: data_dir={data_dir}, host={host}:{port}")

    def _get_ssh_client(self) -> SSHClient:
        """
        Create and configure an SSH client with key-based authentication.

        Returns:
            Configured SSHClient instance
        """
        ssh = SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(
            hostname=self.host,
            port=self.port,
            username=self.username,
            key_filename=self.ssh_key_path,
            look_for_keys=True,
            allow_agent=True,
            timeout=30  # 30 second timeout
        )
        logger.info(f"[SFTP] Connected to {self.username}@{self.host}:{self.port}")
        return ssh

    def _ensure_remote_dir(self, sftp: SFTPClient, remote_dir: str) -> None:
        """
        Ensure a remote directory exists, creating it if necessary.

        Args:
            sftp: SFTP client instance
            remote_dir: Remote directory path to ensure exists
        """
        try:
            sftp.stat(remote_dir)
        except IOError:
            logger.info(f"Creating remote directory: {remote_dir}")
            sftp.mkdir(remote_dir)

    def _ensure_remote_dirs_recursive(self, sftp: SFTPClient, remote_dir: str) -> None:
        """
        Recursively ensure remote directories exist, creating them if necessary.

        Args:
            sftp: SFTP client instance
            remote_dir: Remote directory path to ensure exists
        """
        parts = Path(remote_dir).parts
        current_path = ""
        for part in parts:
            current_path = os.path.join(current_path, part)
            # Skip creating the base remote path again if it already exists
            if current_path == self.remote_path:
                continue
            self._ensure_remote_dir(sftp, current_path)

    def _upload_completion_marker(self, sftp: SFTPClient, remote_folder: str, payload: str = "1") -> None:
        """
        Upload a completion marker file to the remote folder.

        Args:
            sftp: SFTP client instance
            remote_folder: Path to the remote folder
            payload: Character to write as the completion marker (default: "1")
        """
        complete_file = os.path.join(remote_folder, "complete.txt")
        try:
            with sftp.open(complete_file, 'w') as f:
                f.write(payload)
            logger.info(f"Uploaded completion marker: {complete_file}")
        except Exception as e:
            logger.warning(f"Failed to upload completion marker: {e}")

    def _upload_folder_via_sftp(self, local_folder: str) -> bool:
        """
        Upload a single folder to the remote server via SFTP.

        Args:
            local_folder: Path to the local folder to upload

        Returns:
            True if upload succeeded, False otherwise
        """
        folder_name = os.path.basename(local_folder)
        remote_folder = os.path.join(self.remote_path, folder_name)

        logger.info(f"Starting upload: {local_folder} -> {remote_folder}")

        try:
            logger.info("  [1] Connecting to SSH...")
            ssh = self._get_ssh_client()
            logger.info("  [2] Opening SFTP channel...")
            sftp = ssh.open_sftp()
            logger.info("  [3] Ensuring remote directory exists...")
            # First ensure the base remote path exists (e.g., /home/yuwenye/data_test/)
            self._ensure_remote_dirs_recursive(sftp, self.remote_path)
            # Then create the episode folder (e.g., episode_0001)
            self._ensure_remote_dir(sftp, remote_folder)

            logger.info("  [4] Scanning local files...")
            files_to_upload: List[tuple] = []

            for root, dirs, files in os.walk(local_folder):
                rel_root = os.path.relpath(root, local_folder)
                remote_root = os.path.join(remote_folder, rel_root) if rel_root != '.' else remote_folder

                for f in files:
                    local_file = os.path.join(root, f)
                    remote_file = os.path.join(remote_root, f)
                    files_to_upload.append((local_file, remote_file))

            total_files = len(files_to_upload)
            logger.info(f"  [5] Found {total_files} files to upload")

            uploaded = 0

            if HAS_TQDM:
                pbar = tqdm(total=total_files, desc=f"Uploading {folder_name}", unit="file")
            else:
                pbar = None

            for local_file, remote_file in files_to_upload:
                try:
                    remote_dir = os.path.dirname(remote_file)
                    self._ensure_remote_dirs_recursive(sftp, remote_dir)
                    # print(f"    Uploading: {os.path.basename(local_file)}")
                    sftp.put(local_file, remote_file)
                    uploaded += 1
                    if pbar is not None:
                        pbar.update(1)
                    elif uploaded % 10 == 0 or uploaded == total_files:
                        logger.info(f"  Upload progress: {uploaded}/{total_files} files")
                except Exception as e:
                    logger.warning(f"Failed to upload {local_file}: {e}")

            if pbar is not None:
                pbar.close()
            
            if uploaded == total_files:
                self._upload_completion_marker(sftp, remote_folder)

            sftp.close()
            ssh.close()

            logger.info(f"Upload completed: {folder_name} ({uploaded}/{total_files} files)")
            return uploaded == total_files

        except Exception as e:
            logger.warning(f"Failed to upload folder {local_folder}: {e}")
            logger.exception("Upload folder traceback:")
            return False

    def _get_remote_folders(self) -> Set[str]:
        """
        Get the set of folder names that exist in the remote directory.

        Returns:
            Set of folder names in the remote directory
        """
        remote_folders: Set[str] = set()
        try:
            ssh = self._get_ssh_client()
            sftp = ssh.open_sftp()
            try:
                entries = sftp.listdir(self.remote_path)
                for entry in entries:
                    try:
                        # Check if it's a directory
                        stat_info = sftp.stat(os.path.join(self.remote_path, entry))
                        # In SFTP, check if it's a directory by trying to list it
                        if stat_info.st_mode is not None:
                            # Check if it's a directory (S_IFDIR = 0o40000)
                            if stat_info.st_mode & 0o40000:
                                remote_folders.add(entry)
                    except IOError:
                        continue
            except IOError as e:
                logger.warning(f"Failed to list remote directory: {e}")
            finally:
                sftp.close()
                ssh.close()
        except Exception as e:
            logger.warning(f"Failed to connect to remote: {e}")
        return remote_folders

    def _is_remote_folder_complete(self, folder_name: str) -> bool:
        """
        Check if a remote folder contains the complete.txt marker file.

        Args:
            folder_name: Name of the remote folder to check

        Returns:
            True if complete.txt exists in the remote folder, False otherwise.
            Returns False safely if any connection or permission errors occur.
        """
        complete_file_path = os.path.join(self.remote_path, folder_name, "complete.txt")

        try:
            ssh = self._get_ssh_client()
            try:
                sftp = ssh.open_sftp()
                try:
                    sftp.stat(complete_file_path)
                    logger.debug(f"Found complete.txt in remote folder: {folder_name}")
                    return True
                except IOError:
                    logger.debug(f"No complete.txt found in remote folder: {folder_name}")
                    return False
                finally:
                    try:
                        sftp.close()
                    except Exception:
                        pass
            finally:
                try:
                    ssh.close()
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"Failed to check complete.txt for {folder_name}, assuming not complete: {e}")
            return False

    def _poll_and_upload(self) -> None:
        """
        Poll the data directory for new folders and upload them.
        """
        if not os.path.exists(self.data_dir):
            logger.warning(f"Data directory does not exist: {self.data_dir}")
            return

        # Get list of folders that already exist on the remote server
        remote_folders = self._get_remote_folders()
        logger.debug(f"Remote folders: {remote_folders}")

        try:
            entries = os.listdir(self.data_dir)
        except Exception as e:
            logger.warning(f"Failed to list data directory: {e}")
            return

        for entry in entries:
            full_path = os.path.join(self.data_dir, entry)

            if not os.path.isdir(full_path):
                continue

            # Skip if folder already exists on remote with complete.txt
            if entry in remote_folders and self._is_remote_folder_complete(entry):
                logger.debug(f"Folder already exists on remote with complete.txt, skipping: {entry}")
                continue

            # Skip temp folders
            if entry.endswith('temp'):
                logger.debug(f"Skipping temp folder: {entry}")
                continue

            logger.info(f"New folder detected: {entry}")

            success = self._upload_folder_via_sftp(full_path)

            if success:
                self._uploaded_folders.add(entry)
                logger.info(f"Successfully uploaded: {entry}")
            else:
                logger.warning(f"Failed to upload: {entry}")

    def run(self) -> None:
        """
        Main loop for the upload manager thread.
        """
        logger.info("DataUploadManager thread started")

        while self._running:
            try:
                self._poll_and_upload()
                time.sleep(self.poll_interval)
            except Exception as e:
                logger.warning(f"Error in poll loop: {e}")
                logger.exception("Poll loop traceback:")

        logger.info("DataUploadManager thread stopped")

    def start(self) -> None:
        """
        Start the upload manager thread.
        """
        if self._running:
            logger.warning("DataUploadManager is already running")
            return

        self._running = True
        self._thread = threading.Thread(target=self.run, daemon=True)
        self._thread.start()
        logger.info("DataUploadManager started")

    def stop(self) -> None:
        """
        Stop the upload manager thread.
        """
        if not self._running:
            logger.warning("DataUploadManager is not running")
            return

        logger.info("Stopping DataUploadManager...")
        self._running = False

        if self._thread:
            self._thread.join(timeout=10.0)
            if self._thread.is_alive():
                logger.warning("DataUploadManager thread did not stop gracefully")
            else:
                logger.info("DataUploadManager stopped")

    def get_status(self) -> dict:
        """
        Get the current status of the upload manager.

        Returns:
            Dictionary containing status information
        """
        return {
            'running': self._running,
            'data_dir': self.data_dir,
            'host': self.host,
            'port': self.port,
            'remote_path': self.remote_path,
            'poll_interval': self.poll_interval,
            'uploaded_folders_count': len(self._uploaded_folders),
            'uploaded_folders': list(self._uploaded_folders)
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Data Upload Manager")
    parser.add_argument("--config_name", type=str, default=None, required=True,
                        help="Name of the Hydra config file (e.g., i2rt_data_collect_async)")

    args = parser.parse_args()

    config_dict: Dict[str, Any] = {}

    config_dict = load_config_from_hydra(args.config_name)
    logger.info(f"Loaded configuration from: {args.config_name}")
    logger.info(f"Config: {config_dict}")

    data_dir = config_dict.get("data_dir")
    remote_path = config_dict.get("remote_path")

    assert data_dir is not None, "data_dir is required in config"
    assert remote_path is not None, "remote_path is required in config"

    host = config_dict.get("host")
    port = config_dict.get("port")
    username = config_dict.get("username")
    poll_interval = config_dict.get("poll_interval")
    ssh_key_path = config_dict.get("ssh_key_path")

    logger.info("Data Upload Manager - CLI Mode")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Remote server: {username}@{host}:{port}")
    logger.info(f"Remote path: {remote_path}")
    logger.info(f"Poll interval: {poll_interval}s")

    uploader = DataUploadManager(
        data_dir=data_dir,
        host=host,
        port=port,
        username=username,
        remote_path=remote_path,
        poll_interval=poll_interval,
        ssh_key_path=ssh_key_path
    )

    uploader.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        uploader.stop()
        logger.info("DataUploadManager shut down")
