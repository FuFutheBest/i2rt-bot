#!/usr/bin/env python3
"""
Async Episode Visualizer - Visualize episodes saved in async format with per-device timestamps.

- Supports HDF5 files where each device is stored as a group with datasets:
  - data: device-specific samples (e.g., images, arrays)
  - timestamps: float64 seconds aligned to a global episode start
- Aligns different device streams by a common time slider using nearest-sample lookup
- Visualizes:
  - Up to two camera streams as images
  - Vive tracker trajectories (3D + time series)
  - Rotary encoder time series

Author: Assistant, Zheng Wang
"""
import sys
import argparse
import os
import glob
from typing import Dict, Any, Optional, Tuple, List

import base64
import io

import dash
from dash import dcc, html, Input, Output
import dash.dependencies
import plotly.graph_objects as go
import numpy as np
import h5py
from PIL import Image

# Ensure project root is importable when running as a script
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.logger_config import logger


class AsyncEpisode:
    def __init__(self, path: str):
        self.path = path
        self.episode_files: List[str] = []
        self.current_episode_idx: int = 0
        self.devices: Dict[str, Dict[str, Any]] = {}
        self.global_time_range: Tuple[float, float] = (0.0, 0.0)
        self.h5_file: Optional[h5py.File] = None
        self.load_index()
        self.load_episode(0)

    def load_index(self) -> None:
        if os.path.isfile(self.path) and self.path.endswith('.hdf5'):
            self.episode_files = [self.path]
        elif os.path.isdir(self.path):
            self.episode_files = sorted(glob.glob(os.path.join(self.path, 'episode_*.hdf5')))
            if not self.episode_files:
                raise ValueError(f"No HDF5 files found in directory: {self.path}")
        else:
            raise ValueError(f"Path must be a valid HDF5 file or directory: {self.path}")
        logger.info(f"Found {len(self.episode_files)} episode file(s)")

    def close_file(self):
        if self.h5_file:
            self.h5_file.close()
            self.h5_file = None

    def load_episode(self, episode_idx: int) -> None:
        if episode_idx < 0 or episode_idx >= len(self.episode_files):
            logger.warning(f"Episode index {episode_idx} out of range")
            return
        
        self.close_file()
        
        file_path = self.episode_files[episode_idx]
        logger.info(f"Loading async episode {episode_idx}: {file_path}")
        self.current_episode_idx = episode_idx
        self.devices = {}
        
        self.h5_file = h5py.File(file_path, 'r')
        f = self.h5_file

        # root-level: may contain scalar datasets like 'freq' and 'timestamp'
        # devices are groups with 'data' and 'timestamps'
        for key in f.keys():
            item = f[key]
            if isinstance(item, h5py.Group):
                group = item
                device = {'name': key}
                # required
                if 'timestamps' not in group:
                    logger.warning(f"Skip {key}: missing 'timestamps'")
                    continue
                timestamps = group['timestamps'][...].astype(np.float64)
                device['timestamps'] = timestamps

                # data can be images or arrays or strings
                if 'data' in group:
                    device['data'] = group['data']
                    device['data_preview_shape'] = group['data'].shape
                    device['dtype'] = str(group['data'].dtype)
                else:
                    logger.warning(f"{key} has no 'data', timestamps only")
                    device['data'] = None
                    device['data_preview_shape'] = None
                    device['dtype'] = 'none'

                # classify device type heuristically
                dshape = device['data_preview_shape']
                dtype = device['dtype']
                if dshape is not None and len(dshape) >= 3 and dshape[-1] in [1, 3, 4] and 'uint8' in dtype:
                    device['type'] = 'camera'
                    self.devices[key] = device
                elif dshape is not None and len(dshape) == 2 and dshape[1] == 7:
                    device['type'] = 'vive_pose'
                    self.devices[key] = device
                elif dshape is not None and len(dshape) == 3 and dshape[1] == 2 and dshape[2] == 7:
                    # Handle unified vive tracker format [T, 2, 7] by splitting it into two devices
                    logger.info(f"Found unified Vive Tracker format for {key}. Splitting into two virtual devices.")
                    
                    # Left tracker
                    left_device = device.copy()
                    left_device['data'] = device['data'][:, 0, :]
                    left_device['data_preview_shape'] = left_device['data'].shape
                    left_device['type'] = 'vive_pose'
                    self.devices[f"{key}_left"] = left_device

                    # Right tracker
                    right_device = device.copy()
                    right_device['data'] = device['data'][:, 1, :]
                    right_device['data_preview_shape'] = right_device['data'].shape
                    right_device['type'] = 'vive_pose'
                    self.devices[f"{key}_right"] = right_device
                elif dshape is not None and len(dshape) in [1, 2]:
                    device['type'] = 'timeseries'
                    self.devices[key] = device
                else:
                    device['type'] = 'unknown'
                    self.devices[key] = device

            elif isinstance(item, h5py.Dataset):
                # scalar datasets like 'freq', 'timestamp'
                pass

        # Compute global time range
        if self.devices:
            tmins = [np.min(dev['timestamps']) for dev in self.devices.values() if len(dev['timestamps']) > 0]
            tmaxs = [np.max(dev['timestamps']) for dev in self.devices.values() if len(dev['timestamps']) > 0]
            if tmins and tmaxs:
                self.global_time_range = (float(min(tmins)), float(max(tmaxs)))
            else:
                self.global_time_range = (0.0, 0.0)
        else:
            self.global_time_range = (0.0, 0.0)

    def get_time_aligned_indices(self, t: float) -> Dict[str, int]:
        indices: Dict[str, int] = {}
        for key, dev in self.devices.items():
            ts = dev['timestamps']
            if ts is None or len(ts) == 0:
                indices[key] = -1
                continue
            # nearest index
            idx = int(np.clip(np.searchsorted(ts, t, side='left'), 0, len(ts) - 1))
            # choose closer between idx and idx-1
            if idx > 0 and abs(ts[idx] - t) > abs(ts[idx - 1] - t):
                idx = idx - 1
            indices[key] = idx
        return indices

    @staticmethod
    def image_to_base64(img_array: np.ndarray) -> str:
        if img_array.dtype != np.uint8:
            img_array = img_array.astype(np.uint8)
        img = Image.fromarray(img_array)
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        encoded = base64.b64encode(buffer.read()).decode()
        return f"data:image/png;base64,{encoded}"


class AsyncEpisodeVisualizer:
    def __init__(self, path: str):
        self.episode = AsyncEpisode(path)
        self.app = dash.Dash(__name__)
        self.setup_layout()
        self.setup_callbacks()

    def setup_layout(self) -> None:
        t0, t1 = self.episode.global_time_range
        camera_keys = [k for k, d in self.episode.devices.items() if d['type'] == 'camera'][:2]
        vive_keys = [k for k, d in self.episode.devices.items() if d['type'] == 'vive_pose']
        timeseries_keys = [k for k, d in self.episode.devices.items() if d['type'] == 'timeseries']

        self.app.layout = html.Div([
            html.H1("Async Episode Visualizer", style={'textAlign': 'center', 'margin-top': '100px'}),

            # control panel
            html.Div([
                html.Div([
                    html.Label("Episode:", style={'margin-right': '10px', 'color': 'white'}),
                    dcc.Dropdown(
                        id='episode-dropdown',
                        options=[{'label': f"Episode {i}: {os.path.basename(p)}", 'value': i} for i, p in enumerate(self.episode.episode_files)],
                        value=self.episode.current_episode_idx,
                        style={'width': '360px'}
                    )
                ], style={'display': 'inline-block', 'margin-right': '30px'}),
                html.Div([
                    html.Button('Previous', id='prev-button', n_clicks=0, style={'margin-right': '10px', 'padding': '10px 20px'}),
                    html.Button('Next', id='next-button', n_clicks=0, style={'padding': '10px 20px'}),
                ], style={'display': 'inline-block'}),
            ], style={'position': 'fixed','top': '0','left': '0','right': '0','background-color': 'rgba(52,58,64,0.95)','padding': '15px 20px','z-index': '1000','box-shadow': '0 2px 4px rgba(0,0,0,0.1)','backdrop-filter': 'blur(5px)'}),

            # time slider
            html.Div([
                html.Label("Time (s):", style={'color': 'white', 'margin-bottom': '10px', 'display': 'block'}),
                dcc.Slider(id='time-slider', min=t0, max=max(t1, t0 + 1e-3), step=0.001, value=t0,
                           tooltip={"placement": "bottom", "always_visible": True})
            ], style={'margin': '100px 20px 20px 20px'}),

            # images
            html.Div([
                html.H3("Camera Images"),
                html.Div([
                    html.Div([
                        html.H4(camera_keys[i] if i < len(camera_keys) else ''),
                        html.Img(id=f"image-{i}", style={'width': '100%', 'max-width': '640px'})
                    ], style={'display': 'inline-block', 'width': '48%', 'margin': '1%'})
                    for i in range(min(2, len(camera_keys)))
                ])
            ]),

            # vive plots
            html.Div([
                html.H3("Vive Tracker 3D Trajectories"),
                dcc.Graph(id='vive-3d-plot', style={'height': '600px'})
            ]),
            html.Div([
                html.H3("Vive Tracker Position & Rotation"),
                html.Div([
                    html.Div([dcc.Graph(id='vive-pos-plot')], style={'width': '48%', 'display': 'inline-block'}),
                    html.Div([dcc.Graph(id='vive-rot-plot')], style={'width': '48%', 'display': 'inline-block', 'margin-left': '2%'})
                ])
            ]),

            # encoder
            html.Div([
                html.H3("Timeseries (e.g., Encoders)"),
                dcc.Graph(id='timeseries-plot')
            ]),

            html.Div([html.Div(id='data-info')], style={'margin': '20px'})
        ])

        # store keys for callbacks
        self.camera_keys = camera_keys
        self.vive_keys = vive_keys
        self.timeseries_keys = timeseries_keys

    def setup_callbacks(self) -> None:
        # episode navigation updates time range and keys
        @self.app.callback(
            [Output('time-slider', 'min'), Output('time-slider', 'max'), Output('time-slider', 'value'), Output('episode-dropdown', 'value')],
            [Input('prev-button', 'n_clicks'), Input('next-button', 'n_clicks'), Input('episode-dropdown', 'value')],
            [dash.dependencies.State('time-slider', 'value')],
            prevent_initial_call=False
        )
        def update_episode(prev_clicks, next_clicks, episode_value, current_time):
            ctx = dash.callback_context
            new_idx = self.episode.current_episode_idx
            if ctx.triggered:
                bid = ctx.triggered[0]['prop_id'].split('.')[0]
                if bid == 'prev-button' and self.episode.current_episode_idx > 0:
                    new_idx = self.episode.current_episode_idx - 1
                elif bid == 'next-button' and self.episode.current_episode_idx < len(self.episode.episode_files) - 1:
                    new_idx = self.episode.current_episode_idx + 1
                elif bid == 'episode-dropdown' and episode_value is not None:
                    new_idx = int(episode_value)
            if new_idx != self.episode.current_episode_idx:
                self.episode.load_episode(new_idx)
                # refresh device keys
                self.camera_keys = [k for k, d in self.episode.devices.items() if d['type'] == 'camera'][:2]
                self.vive_keys = [k for k, d in self.episode.devices.items() if d['type'] == 'vive_pose']
                self.timeseries_keys = [k for k, d in self.episode.devices.items() if d['type'] == 'timeseries']
            t0, t1 = self.episode.global_time_range
            return t0, max(t1, t0 + 1e-3), t0, self.episode.current_episode_idx

        # camera updates
        for i, cam_key in enumerate(self.camera_keys):
            @self.app.callback(
                Output(f'image-{i}', 'src'),
                [Input('time-slider', 'value'), Input('episode-dropdown', 'value')],
                prevent_initial_call=False
            )
            def update_image(t, episode_idx, cam_idx=i, key=cam_key):
                if key not in self.episode.devices:
                    return ''
                dev = self.episode.devices[key]
                idx_map = self.episode.get_time_aligned_indices(t)
                idx = idx_map.get(key, -1)
                if idx < 0 or dev['data'] is None or idx >= dev['data'].shape[0]:
                    return ''
                img = dev['data'][idx]
                return AsyncEpisode.image_to_base64(img)

        # vive plots
        @self.app.callback(
            [Output('vive-3d-plot', 'figure'), Output('vive-pos-plot', 'figure'), Output('vive-rot-plot', 'figure')],
            [Input('time-slider', 'value'), Input('episode-dropdown', 'value')],
            prevent_initial_call=False
        )
        def update_vive_plots(t, episode_idx):
            colors = ['blue', 'red', 'green', 'orange']
            # 3D
            fig3d = go.Figure()
            pos_fig = go.Figure()
            rot_fig = go.Figure()
            for i, key in enumerate(self.vive_keys):
                if key not in self.episode.devices:
                    continue
                dev = self.episode.devices[key]
                
                # Explicitly read data into a numpy array
                data_h5_obj = dev['data']
                if data_h5_obj is None or len(data_h5_obj.shape) != 2 or data_h5_obj.shape[1] != 7:
                    continue
                
                data = data_h5_obj[...] # Load into memory
                
                ts = dev['timestamps']
                # draw trajectory up to t
                mask = ts <= t
                if not np.any(mask):
                    continue

                pos = data[mask, :3]
                quat = data[mask, 3:]

                color = colors[i % len(colors)]
                # 3D trajectory
                fig3d.add_trace(go.Scatter3d(x=pos[:, 0], y=pos[:, 1], z=pos[:, 2], mode='lines+markers', name=key,
                                             line=dict(color=color, width=4), marker=dict(size=3)))
                # current point
                idx_map = self.episode.get_time_aligned_indices(t)
                idx = idx_map.get(key, -1)
                if 0 <= idx < data.shape[0]:
                    fig3d.add_trace(go.Scatter3d(x=[data[idx, 0]], y=[data[idx, 1]], z=[data[idx, 2]], mode='markers',
                                                 name=f'{key} current', marker=dict(size=8, color=color, symbol='diamond')))
                # position series
                frames = ts[mask]
                pos_fig.add_trace(go.Scatter(x=frames, y=pos[:, 0], name=f'{key} X', line=dict(color=color)))
                pos_fig.add_trace(go.Scatter(x=frames, y=pos[:, 1], name=f'{key} Y', line=dict(color=color, dash='dash')))
                pos_fig.add_trace(go.Scatter(x=frames, y=pos[:, 2], name=f'{key} Z', line=dict(color=color, dash='dot')))
                # quaternion series
                rot_fig.add_trace(go.Scatter(x=frames, y=quat[:, 0], name=f'{key} qx', line=dict(color=color)))
                rot_fig.add_trace(go.Scatter(x=frames, y=quat[:, 1], name=f'{key} qy', line=dict(color=color, dash='dash')))
                rot_fig.add_trace(go.Scatter(x=frames, y=quat[:, 2], name=f'{key} qz', line=dict(color=color, dash='dot')))
                rot_fig.add_trace(go.Scatter(x=frames, y=quat[:, 3], name=f'{key} qw', line=dict(color=color, dash='dashdot')))
            fig3d.update_layout(title=f"Vive 3D Trajectories (t={t:.3f}s)", scene=dict(aspectmode='cube'))
            pos_fig.update_layout(title="Vive Positions", xaxis_title="Time (s)", yaxis_title="Position (m)")
            rot_fig.update_layout(title="Vive Quaternions", xaxis_title="Time (s)", yaxis_title="Quaternion")
            return fig3d, pos_fig, rot_fig

        # timeseries plot
        @self.app.callback(
            Output('timeseries-plot', 'figure'),
            [Input('time-slider', 'value'), Input('episode-dropdown', 'value')],
            prevent_initial_call=False
        )
        def update_timeseries(t, episode_idx):
            fig = go.Figure()
            colors = ['blue', 'red', 'green', 'orange', 'purple']
            for i, key in enumerate(self.timeseries_keys):
                if key not in self.episode.devices:
                    continue
                dev = self.episode.devices[key]
                data = dev['data']
                ts = dev['timestamps']
                if data is None:
                    continue
                # squeeze to 1D if needed
                arr = data[...]
                if arr.ndim == 2 and arr.shape[1] == 1:
                    arr = arr[:, 0]
                elif arr.ndim > 1:
                    continue  # skip unsupported shapes here
                mask = ts <= t
                if np.any(mask):
                    fig.add_trace(go.Scatter(x=ts[mask], y=arr[mask], name=key, line=dict(color=colors[i % len(colors)])))
            fig.update_layout(title=f"Timeseries (t={t:.3f}s)", xaxis_title="Time (s)", yaxis_title="Value")
            return fig

        # info
        @self.app.callback(
            Output('data-info', 'children'),
            [Input('time-slider', 'value'), Input('episode-dropdown', 'value')],
            prevent_initial_call=False
        )
        def update_info(t, episode_idx):
            indices = self.episode.get_time_aligned_indices(t)
            info_lines = [html.P([html.Strong(f"t={t:.3f}s")])]
            for key, idx in indices.items():
                info_lines.append(html.P(f"{key}: idx={idx}"))
            return info_lines

    def run(self, host: str = '127.0.0.1', port: int = 8051, debug: bool = True) -> None:
        logger.info(f"Starting async visualizer at http://{host}:{port}")
        self.app.run(host=host, port=port, debug=debug)


def main():
    parser = argparse.ArgumentParser(description="Async episode visualizer for per-device timestamped HDF5")
    parser.add_argument("path", help="Path to HDF5 episode file or directory of episodes")
    parser.add_argument("--host", default="127.0.0.1", help="Host to run server on")
    parser.add_argument("--port", type=int, default=8051, help="Port to run server on")
    parser.add_argument("--no-debug", action="store_true", help="Disable debug mode")
    args = parser.parse_args()

    try:
        vis = AsyncEpisodeVisualizer(args.path)
        vis.run(host=args.host, port=args.port, debug=not args.no_debug)
    except Exception as e:
        logger.error(f"Failed to start async visualizer: {e}")
        return 1
    return 0


if __name__ == "__main__":
    exit(main())
