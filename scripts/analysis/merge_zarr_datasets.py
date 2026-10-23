#!/usr/bin/env python3
"""
Zarr Dataset Merger with Duplicate Detection

This script merges multiple zarr datasets into a single zarr dataset with
comprehensive duplicate detection and metadata management.

Features:
- Merges replay buffer data from multiple zarr files
- Episode naming convention: {id}_{timestamp} for unique identification
- Duplicate detection based on episode names
- Multiple merge strategies: skip, overwrite, keep_both, error
- Automatic timestamp extraction from legacy data
- Preserves all data including images, poses, and gripper data
- Maintains data integrity and compression settings
- Comprehensive logging of merge process with metadata

Usage:
    # Basic merge with duplicate detection
    python merge_zarr_datasets.py input1.zarr.zip input2.zarr.zip output_merged.zarr.zip
    
    # Specify merge strategy for duplicates
    python merge_zarr_datasets.py input1.zarr.zip input2.zarr.zip output.zarr.zip --duplicate-strategy skip
    
    # Merge with custom station ID
    python merge_zarr_datasets.py input1.zarr.zip input2.zarr.zip output.zarr.zip --station-id 1

Author: Jiulong Dong
"""

import argparse
import sys
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
from tqdm import tqdm
from datetime import datetime
import json
import hashlib

# Setup path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import numpy as np
import zarr

from utils.logger_config import logger
from utils.replay_buffer import ReplayBuffer
from utils.imagecodecs_numcodecs import register_codecs, JpegXl

# Register codecs for image compression
register_codecs()


class EpisodeMetadata:
    """Metadata for a single episode."""
    
    def __init__(self, name: str, source_file: str, episode_idx: int, n_steps: int):
        """
        Initialize episode metadata.
        
        Args:
            name: Unique episode name in format {id}_{timestamp}
            source_file: Source zarr file path
            episode_idx: Episode index in source file
            n_steps: Number of steps in this episode
        """
        self.name = name
        self.source_file = source_file
        self.episode_idx = episode_idx
        self.n_steps = n_steps
        self.station_id, self.timestamp = self._parse_name()
    
    def _parse_name(self) -> Tuple[int, int]:
        """Parse episode name to extract station_id and timestamp."""
        try:
            parts = self.name.split('_')
            if len(parts) >= 2:
                station_id = int(parts[0])
                timestamp = int(parts[1])
                return station_id, timestamp
            else:
                logger.warning(f"Invalid episode name format: {self.name}, using defaults")
                return 0, 0
        except Exception as e:
            logger.warning(f"Error parsing episode name {self.name}: {e}, using defaults")
            return 0, 0
    
    def __repr__(self) -> str:
        return f"Episode({self.name}, source={Path(self.source_file).name}, idx={self.episode_idx}, steps={self.n_steps})"


class ZarrMerger:
    """Merge multiple zarr datasets into a single dataset with duplicate detection."""
    
    DUPLICATE_STRATEGIES = ['skip', 'overwrite', 'keep_both', 'error']
    
    def __init__(self, input_zarr_paths: List[Path], output_zarr_path: Path,
                 duplicate_strategy: str = 'skip', default_station_id: int = 0):
        """
        Initialize the zarr merger.
        
        Args:
            input_zarr_paths: List of paths to input zarr.zip files to merge
            output_zarr_path: Path to output merged zarr.zip file
            duplicate_strategy: Strategy for handling duplicates ('skip', 'overwrite', 'keep_both', 'error')
            default_station_id: Default station ID for episodes without metadata (default: 0)
        """
        self.input_zarr_paths = [Path(p) for p in input_zarr_paths]
        self.output_zarr_path = Path(output_zarr_path)
        self.duplicate_strategy = duplicate_strategy
        self.default_station_id = default_station_id
        
        # Validate duplicate strategy
        if self.duplicate_strategy not in self.DUPLICATE_STRATEGIES:
            raise ValueError(f"Invalid duplicate_strategy: {duplicate_strategy}. "
                           f"Must be one of {self.DUPLICATE_STRATEGIES}")
        
        # Validate input files exist
        for path in self.input_zarr_paths:
            if not path.exists():
                raise FileNotFoundError(f"Input zarr file not found: {path}")
        
        # Check output file doesn't already exist
        if self.output_zarr_path.exists():
            logger.warning(f"Output file {self.output_zarr_path} already exists and will be overwritten")
        
        # Statistics tracking
        self.merge_stats = {
            'total_episodes_input': 0,
            'total_episodes_output': 0,
            'duplicates_found': 0,
            'duplicates_skipped': 0,
            'duplicates_overwritten': 0,
            'duplicates_kept_both': 0,
        }
        
        logger.info(f"Initialized ZarrMerger with {len(self.input_zarr_paths)} input files")
        logger.info(f"  Duplicate strategy: {self.duplicate_strategy}")
        logger.info(f"  Default station ID: {self.default_station_id}")
        for i, path in enumerate(self.input_zarr_paths, 1):
            logger.info(f"  Input {i}: {path}")
        logger.info(f"  Output: {self.output_zarr_path}")
    
    def _load_replay_buffer(self, zarr_path: Path) -> ReplayBuffer:
        """
        Load a replay buffer from a zarr file.
        
        Args:
            zarr_path: Path to zarr.zip file
            
        Returns:
            Loaded ReplayBuffer object
        """
        logger.info(f"Loading replay buffer from {zarr_path}")
        
        with zarr.ZipStore(zarr_path, mode='r') as store:
            replay_buffer = ReplayBuffer.copy_from_store(
                src_store=store,
                store=zarr.MemoryStore()
            )
        
        n_episodes = replay_buffer.n_episodes
        n_steps = replay_buffer.n_steps
        
        logger.info(f"  Loaded {n_episodes} episodes with {n_steps} total steps")
        
        # Log data keys
        data_keys = list(replay_buffer.data.keys())
        logger.info(f"  Data keys: {data_keys}")
        
        return replay_buffer
    
    def _get_dataset_info(self, replay_buffer: ReplayBuffer) -> Dict:
        """
        Get information about datasets in the replay buffer.
        
        Args:
            replay_buffer: ReplayBuffer to inspect
            
        Returns:
            Dictionary with dataset information
        """
        info = {
            'n_episodes': replay_buffer.n_episodes,
            'n_steps': replay_buffer.n_steps,
            'datasets': {}
        }
        
        for key in replay_buffer.data.keys():
            if key == 'episode_ends':
                continue
            
            dataset = replay_buffer.data[key]
            info['datasets'][key] = {
                'shape': dataset.shape,
                'dtype': dataset.dtype,
                'compressor': dataset.compressor if hasattr(dataset, 'compressor') else None
            }
        
        return info
    
    def _extract_episode_metadata(self, replay_buffer: ReplayBuffer, source_file: Path) -> List[EpisodeMetadata]:
        """
        Extract or generate episode metadata from replay buffer.
        
        Args:
            replay_buffer: ReplayBuffer to extract metadata from
            source_file: Source zarr file path
            
        Returns:
            List of EpisodeMetadata objects
        """
        logger.info(f"Extracting episode metadata from {source_file.name}")
        
        # Try to load existing metadata from zarr attributes
        episode_names = None
        if hasattr(replay_buffer.root, 'attrs') and 'episode_names' in replay_buffer.root.attrs:
            episode_names = replay_buffer.root.attrs['episode_names']
            logger.info(f"  Found {len(episode_names)} episode names in metadata")
        
        # Generate metadata for each episode
        episodes = []
        for ep_idx in range(replay_buffer.n_episodes):
            # Calculate episode length
            if ep_idx == 0:
                start_idx = 0
            else:
                start_idx = replay_buffer.episode_ends[ep_idx - 1]
            end_idx = replay_buffer.episode_ends[ep_idx]
            n_steps = end_idx - start_idx
            
            # Get or generate episode name
            if episode_names and ep_idx < len(episode_names):
                name = episode_names[ep_idx]
            else:
                # Generate name for legacy data without metadata
                name = self._generate_episode_name_from_data(
                    replay_buffer, ep_idx, start_idx, source_file
                )
            
            episode = EpisodeMetadata(
                name=name,
                source_file=str(source_file),
                episode_idx=ep_idx,
                n_steps=n_steps
            )
            episodes.append(episode)
        
        return episodes
    
    def _generate_episode_name_from_data(self, replay_buffer: ReplayBuffer, 
                                        ep_idx: int, start_idx: int, source_file: Path) -> str:
        """
        Generate episode name for legacy data without metadata.
        Uses first frame left camera timestamp if available, otherwise file hash + index.
        
        Args:
            replay_buffer: ReplayBuffer containing episode data
            ep_idx: Episode index
            start_idx: Start frame index
            source_file: Source file path
            
        Returns:
            Generated episode name in format {station_id}_{timestamp}
        """
        try:
            # Try to extract timestamp from camera data if available
            camera_keys = [k for k in replay_buffer.data.keys() if 'camera' in k.lower() and 'rgb' in k.lower()]
            
            if camera_keys:
                # Use first camera's first frame as reference
                # Since we don't have actual timestamp in data, use a hash of the first frame
                first_camera_key = sorted(camera_keys)[0]  # Get first camera (e.g., camera0_rgb)
                
                # Get first frame data
                first_frame = replay_buffer.data[first_camera_key][start_idx]
                
                # Generate a stable hash from first frame data
                frame_hash = hashlib.md5(first_frame.tobytes()).hexdigest()
                # Use first 12 hex digits as timestamp-like identifier
                pseudo_timestamp = int(frame_hash[:12], 16)
                
                name = f"{self.default_station_id}_{pseudo_timestamp}"
                logger.debug(f"Generated name from camera data: {name}")
            else:
                # Fallback: use file hash + episode index
                file_hash = hashlib.md5(str(source_file).encode()).hexdigest()[:8]
                # Create timestamp-like value from hash
                pseudo_timestamp = int(file_hash, 16) + ep_idx
                name = f"{self.default_station_id}_{pseudo_timestamp}"
                logger.debug(f"Generated name from file hash: {name}")
            
            return name
            
        except Exception as e:
            logger.warning(f"Error generating episode name: {e}, using fallback")
            # Ultimate fallback: use current timestamp + episode index
            timestamp = int(datetime.now().timestamp() * 1000) + ep_idx
            return f"{self.default_station_id}_{timestamp}"
    
    def _detect_duplicates(self, name_to_episodes: Dict[str, List[EpisodeMetadata]]) -> None:
        """
        Detect duplicate episodes based on names and log statistics.
        
        Args:
            name_to_episodes: Dictionary mapping episode names to lists of episodes
        """
        logger.info("Detecting duplicate episodes...")
        
        # Filter to only duplicates
        duplicates = {name: eps for name, eps in name_to_episodes.items() if len(eps) > 1}
        
        if duplicates:
            logger.warning(f"Found {len(duplicates)} duplicate episode names:")
            for name, eps in duplicates.items():
                self.merge_stats['duplicates_found'] += len(eps) - 1  # Count extras
                logger.warning(f"  {name}: {len(eps)} occurrences")
                for ep in eps:
                    logger.warning(f"    - {Path(ep.source_file).name}, episode {ep.episode_idx}")
        else:
            logger.info("  No duplicates found")
    
    def _resolve_duplicates(self, name_to_episodes: Dict[str, List[EpisodeMetadata]]) -> List[EpisodeMetadata]:
        """
        Resolve duplicate episodes based on strategy.
        
        Args:
            name_to_episodes: Dictionary mapping names to episodes
            
        Returns:
            List of episodes to include in merge (after duplicate resolution)
        """
        logger.info(f"Resolving duplicates using strategy: {self.duplicate_strategy}")
        
        resolved_episodes = []
        
        for name, episodes in name_to_episodes.items():
            if len(episodes) == 1:
                # No duplicates, include as-is
                resolved_episodes.append(episodes[0])
            else:
                # Handle duplicates based on strategy
                if self.duplicate_strategy == 'error':
                    raise ValueError(f"Duplicate episode name found: {name}. "
                                   f"Found in files: {[Path(ep.source_file).name for ep in episodes]}")
                
                elif self.duplicate_strategy == 'skip':
                    # Keep first occurrence, skip rest
                    resolved_episodes.append(episodes[0])
                    self.merge_stats['duplicates_skipped'] += len(episodes) - 1
                    logger.info(f"  Skipping {len(episodes) - 1} duplicate(s) of {name}")
                
                elif self.duplicate_strategy == 'overwrite':
                    # Keep last occurrence
                    resolved_episodes.append(episodes[-1])
                    self.merge_stats['duplicates_overwritten'] += len(episodes) - 1
                    logger.info(f"  Overwriting with last occurrence of {name}")
                
                elif self.duplicate_strategy == 'keep_both':
                    # Keep all, add suffix to duplicates
                    for i, ep in enumerate(episodes):
                        if i > 0:
                            # Add suffix to duplicates
                            original_name = ep.name
                            ep.name = f"{original_name}_dup{i}"
                            logger.info(f"  Renamed duplicate: {original_name} -> {ep.name}")
                            self.merge_stats['duplicates_kept_both'] += 1
                        resolved_episodes.append(ep)
        
        # Sort by timestamp for consistent ordering
        # resolved_episodes.sort(key=lambda ep: ep.timestamp)
        
        logger.info(f"Resolved to {len(resolved_episodes)} episodes to merge")
        return resolved_episodes
    
    def merge(self) -> None:
        """Merge all input zarr files into a single output zarr file with duplicate detection."""
        
        logger.info("=" * 80)
        logger.info("STARTING ZARR MERGE PROCESS WITH DUPLICATE DETECTION")
        logger.info("=" * 80)
        
        # Phase 1: Load all input replay buffers and extract metadata
        logger.info("\n[PHASE 1] Loading input datasets and extracting metadata...")
        input_buffers = []
        all_episodes = []
        buffer_to_episodes = {}  # Map buffer index to its episodes
        
        for i, zarr_path in enumerate(self.input_zarr_paths, 1):
            logger.info(f"\n[{i}/{len(self.input_zarr_paths)}] Processing: {zarr_path}")
            buffer = self._load_replay_buffer(zarr_path)
            input_buffers.append(buffer)
            
            # Extract episode metadata
            episodes = self._extract_episode_metadata(buffer, zarr_path)
            buffer_to_episodes[i-1] = episodes
            all_episodes.extend(episodes)
            
            self.merge_stats['total_episodes_input'] += len(episodes)
            
            # Log dataset info for first input
            if i == 1:
                info = self._get_dataset_info(buffer)
                logger.info("\nDataset structure (from first input):")
                for key, dataset_info in info['datasets'].items():
                    logger.info(f"  {key}: shape={dataset_info['shape']}, dtype={dataset_info['dtype']}")
        
        logger.info("\n" + "=" * 80)
        logger.info(f"Total episodes loaded: {len(all_episodes)}")
        logger.info(f"Total input files: {len(self.input_zarr_paths)}")
        logger.info("=" * 80)
        
        # Phase 2: Detect and resolve duplicates
        logger.info("\n[PHASE 2] Detecting and resolving duplicates...")
        name_to_episodes: Dict[str, List[EpisodeMetadata]] = {}
        for episode in all_episodes:
            if episode.name not in name_to_episodes:
                name_to_episodes[episode.name] = []
            name_to_episodes[episode.name].append(episode)
        
        # Detect duplicates (for logging and statistics)
        self._detect_duplicates(name_to_episodes)
        
        # Resolve duplicates based on strategy
        episodes_to_merge = self._resolve_duplicates(name_to_episodes)
        
        logger.info(f"\nEpisodes to merge after duplicate resolution: {len(episodes_to_merge)}")
        self.merge_stats['total_episodes_output'] = len(episodes_to_merge)
        
        # Phase 3: Create output replay buffer and merge episodes
        logger.info("\n[PHASE 3] Creating output replay buffer and merging episodes...")
        output_buffer = ReplayBuffer.create_empty_zarr(storage=zarr.MemoryStore())
        
        # Get all dataset keys (excluding episode_ends which is handled separately)
        first_buffer = input_buffers[0]
        data_keys = [k for k in first_buffer.data.keys() if k != 'episode_ends']
        
        logger.info(f"Data keys to merge: {data_keys}")
        
        # Build a map from (source_file, episode_idx) to ReplayBuffer for quick lookup
        source_to_buffer = {}
        for i, zarr_path in enumerate(self.input_zarr_paths):
            source_to_buffer[str(zarr_path)] = input_buffers[i]
        
        # Merge episodes in the resolved order
        logger.info(f"\nMerging {len(episodes_to_merge)} episodes...")
        output_episode_names = []
        
        for ep_meta in tqdm(episodes_to_merge, desc="Merging episodes"):
            # Get the source buffer
            buffer = source_to_buffer[ep_meta.source_file]
            ep_idx = ep_meta.episode_idx
            
            # Get episode slice
            if ep_idx == 0:
                start_idx = 0
            else:
                start_idx = buffer.episode_ends[ep_idx - 1]
            end_idx = buffer.episode_ends[ep_idx]
            
            # Extract data for this episode
            episode_data = {}
            for key in data_keys:
                episode_data[key] = buffer.data[key][start_idx:end_idx]
            
            # Get compressor from original data
            compressors = {}
            for key in data_keys:
                if hasattr(buffer.data[key], 'compressor'):
                    compressors[key] = buffer.data[key].compressor
            
            # Add episode to output buffer
            output_buffer.add_episode(data=episode_data, compressors=compressors)
            output_episode_names.append(ep_meta.name)
        
        logger.info(f"\nSuccessfully merged {len(episodes_to_merge)} episodes")
        logger.info(f"Output buffer: {output_buffer.n_episodes} episodes, {output_buffer.n_steps} steps")
        
        # Phase 4: Add metadata to output buffer
        logger.info("\n[PHASE 4] Adding metadata to output buffer...")
        self._add_metadata_to_buffer(output_buffer, output_episode_names)
        
        # Phase 5: Verify merged data
        logger.info("\n[PHASE 5] Verifying merged data...")
        self._verify_merge_with_metadata(episodes_to_merge, output_buffer)
        
        # Phase 6: Save output
        logger.info(f"\n[PHASE 6] Saving merged dataset to {self.output_zarr_path}...")
        with zarr.ZipStore(self.output_zarr_path, mode='w') as zip_store:
            output_buffer.save_to_store(store=zip_store)
        
        # Print final summary
        self._print_merge_summary()
        
        # Get file size
        if self.output_zarr_path.exists():
            file_size_mb = self.output_zarr_path.stat().st_size / (1024 * 1024)
            logger.info(f"Output file size: {file_size_mb:.2f} MB")
    
    def _add_metadata_to_buffer(self, replay_buffer: ReplayBuffer, episode_names: List[str]) -> None:
        """
        Add metadata to replay buffer.
        
        Args:
            replay_buffer: ReplayBuffer to add metadata to
            episode_names: List of episode names in order
        """
        metadata = {
            'episode_names': episode_names,
            'creation_time': datetime.now().isoformat(),
            'data_version': '1.0',
            'merge_strategy': self.duplicate_strategy,
            'n_episodes': len(episode_names),
            'n_input_files': len(self.input_zarr_paths)
        }
        
        # Store metadata in zarr attributes
        replay_buffer.root.attrs['episode_names'] = episode_names
        replay_buffer.root.attrs['creation_time'] = metadata['creation_time']
        replay_buffer.root.attrs['data_version'] = metadata['data_version']
        replay_buffer.root.attrs['merge_strategy'] = metadata['merge_strategy']
        replay_buffer.root.attrs['merge_stats'] = json.dumps(self.merge_stats)
        
        logger.info(f"Added metadata for {len(episode_names)} episodes")
        logger.debug(f"Metadata: {metadata}")
    
    def _verify_merge_with_metadata(self, episodes_to_merge: List[EpisodeMetadata], 
                                    output_buffer: ReplayBuffer) -> None:
        """
        Verify that the merge was successful with metadata validation.
        
        Args:
            episodes_to_merge: List of episodes that should be in output
            output_buffer: Merged output replay buffer
        """
        logger.info("Verifying merge with metadata...")
        
        # Check episode count
        expected_episodes = len(episodes_to_merge)
        if output_buffer.n_episodes != expected_episodes:
            logger.error(f"Episode count mismatch! Expected {expected_episodes}, got {output_buffer.n_episodes}")
            raise ValueError("Merge verification failed: episode count mismatch")
        
        # Check step count
        expected_steps = sum(ep.n_steps for ep in episodes_to_merge)
        if output_buffer.n_steps != expected_steps:
            logger.error(f"Step count mismatch! Expected {expected_steps}, got {output_buffer.n_steps}")
            raise ValueError("Merge verification failed: step count mismatch")
        
        # Check metadata exists
        if not hasattr(output_buffer.root, 'attrs') or 'episode_names' not in output_buffer.root.attrs:
            logger.warning("Episode names metadata not found in output buffer")
        else:
            episode_names = output_buffer.root.attrs['episode_names']
            if len(episode_names) != expected_episodes:
                logger.error(f"Episode names count mismatch! Expected {expected_episodes}, got {len(episode_names)}")
                raise ValueError("Merge verification failed: episode names count mismatch")
        
        logger.info("✅ Merge verification passed!")
        logger.info(f"  Episodes: {output_buffer.n_episodes} ✓")
        logger.info(f"  Steps: {output_buffer.n_steps} ✓")
        logger.info(f"  Metadata: present ✓")
    
    def _print_merge_summary(self) -> None:
        """Print comprehensive merge summary."""
        logger.info("\n" + "=" * 80)
        logger.info("MERGE COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"\nOutput file: {self.output_zarr_path}")
        logger.info(f"\nMERGE STATISTICS:")
        logger.info(f"  Input files: {len(self.input_zarr_paths)}")
        logger.info(f"  Total episodes (input): {self.merge_stats['total_episodes_input']}")
        logger.info(f"  Total episodes (output): {self.merge_stats['total_episodes_output']}")
        logger.info(f"  Duplicate strategy: {self.duplicate_strategy}")
        
        if self.merge_stats['duplicates_found'] > 0:
            logger.info(f"\nDUPLICATE HANDLING:")
            logger.info(f"  Duplicates found: {self.merge_stats['duplicates_found']}")
            logger.info(f"  Duplicates skipped: {self.merge_stats['duplicates_skipped']}")
            logger.info(f"  Duplicates overwritten: {self.merge_stats['duplicates_overwritten']}")
            logger.info(f"  Duplicates kept (renamed): {self.merge_stats['duplicates_kept_both']}")
        else:
            logger.info(f"\n  No duplicates found ✓")
        
        logger.info("=" * 80)
    
    def _verify_merge(self, input_buffers: List[ReplayBuffer], output_buffer: ReplayBuffer) -> None:
        """
        Legacy verification method (kept for compatibility).
        
        Args:
            input_buffers: List of input replay buffers
            output_buffer: Merged output replay buffer
        """
        logger.info("\nVerifying merge...")
        
        # Check episode count
        expected_episodes = sum(b.n_episodes for b in input_buffers)
        if output_buffer.n_episodes != expected_episodes:
            logger.error(f"Episode count mismatch! Expected {expected_episodes}, got {output_buffer.n_episodes}")
            raise ValueError("Merge verification failed: episode count mismatch")
        
        # Check step count
        expected_steps = sum(b.n_steps for b in input_buffers)
        if output_buffer.n_steps != expected_steps:
            logger.error(f"Step count mismatch! Expected {expected_steps}, got {output_buffer.n_steps}")
            raise ValueError("Merge verification failed: step count mismatch")
        
        # Check data keys
        first_buffer_keys = set(k for k in input_buffers[0].data.keys())
        output_keys = set(k for k in output_buffer.data.keys())
        
        if first_buffer_keys != output_keys:
            logger.error(f"Data key mismatch!")
            logger.error(f"  Expected: {sorted(first_buffer_keys)}")
            logger.error(f"  Got: {sorted(output_keys)}")
            raise ValueError("Merge verification failed: data key mismatch")
        
        logger.info("✅ Merge verification passed!")
        logger.info(f"  Episodes: {output_buffer.n_episodes} ✓")
        logger.info(f"  Steps: {output_buffer.n_steps} ✓")
        logger.info(f"  Data keys: {len(output_keys)} ✓")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Merge multiple zarr datasets into a single dataset with duplicate detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic merge with default duplicate handling (skip)
  python merge_zarr_datasets.py dataset1.zarr.zip dataset2.zarr.zip merged.zarr.zip
  
  # Merge with error on duplicates
  python merge_zarr_datasets.py data1.zarr.zip data2.zarr.zip output.zarr.zip --duplicate-strategy error
  
  # Merge keeping both duplicates (with renamed suffix)
  python merge_zarr_datasets.py data1.zarr.zip data2.zarr.zip output.zarr.zip --duplicate-strategy keep_both
  
  # Merge with custom station ID
  python merge_zarr_datasets.py data1.zarr.zip data2.zarr.zip output.zarr.zip --station-id 1
  
  # Merge multiple files
  python merge_zarr_datasets.py data1.zarr.zip data2.zarr.zip data3.zarr.zip output.zarr.zip

Episode Naming Convention:
  Episodes are named as: {station_id}_{timestamp}
  - station_id: Collection station ID (default: 0)
  - timestamp: Unique timestamp or hash-based identifier

Duplicate Strategies:
  - skip: Keep first occurrence, skip duplicates (default)
  - overwrite: Keep last occurrence, overwrite duplicates
  - keep_both: Keep all occurrences with renamed suffixes
  - error: Raise error if duplicates are found
        """
    )
    
    parser.add_argument(
        'input_zarr_files',
        type=Path,
        nargs='+',
        help='Input zarr.zip files to merge (at least 2 required)'
    )
    
    parser.add_argument(
        'output_zarr_file',
        type=Path,
        help='Output merged zarr.zip file'
    )
    
    parser.add_argument(
        '--duplicate-strategy',
        type=str,
        choices=['skip', 'overwrite', 'keep_both', 'error'],
        default='skip',
        help='Strategy for handling duplicate episodes (default: skip)'
    )
    
    parser.add_argument(
        '--station-id',
        type=int,
        default=0,
        help='Default station ID for episodes without metadata (default: 0)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if len(args.input_zarr_files) < 1:
        parser.error("At least 1 input zarr file is required")
    
    # The last argument is the output file, all others are inputs
    input_files = args.input_zarr_files
    output_file = args.output_zarr_file
    
    if len(input_files) < 2:
        parser.error("At least 2 input zarr files are required for merging")
    
    try:
        # Create merger and run
        merger = ZarrMerger(
            input_zarr_paths=input_files, 
            output_zarr_path=output_file,
            duplicate_strategy=args.duplicate_strategy,
            default_station_id=args.station_id
        )
        merger.merge()
        
    except Exception as e:
        logger.error(f"Merge failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()