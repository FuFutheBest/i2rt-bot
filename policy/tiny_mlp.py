#!/usr/bin/env python3
"""
Tiny MLP policy - simple randomly initialized MLP that maps observation to action.

This is a placeholder policy for wiring the end-to-end loop. It does not aim to
produce meaningful control, only to provide a consistent interface.

Author: Jun Lv
"""

from typing import Any, Dict, List, Optional
import time
import numpy as np

from .base import BasePolicy, ObservationType
from utils.logger_config import logger

def _tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


class TinyMLPPolicy(BasePolicy):
    """
    A very small MLP with random weights.

    Architecture: obs_dim -> h1 -> h2 -> action_dim
    Activation: ReLU on hidden, optional tanh on output
    """

    def __init__(self,
                 hidden_sizes: List[int] = [64, 64],
                 output_tanh: bool = True,
                 seed: Optional[int] = None,
                 **kwargs: Any) -> None:
        super().__init__(hidden_sizes=hidden_sizes, output_tanh=output_tanh, seed=seed, **kwargs)
        self.hidden_sizes = hidden_sizes
        self.output_tanh = output_tanh
        self.seed = seed
        self.params_built = False
        self.step_counter = 0 # Added for predictable action generation
        self.delay_mean = 0.3
        self.delay_std = 0.05

        # Maintain last predicted chunks and their start steps for continuity
        self.last_action_chunks: Dict[str, np.ndarray] = {}  # device_name -> chunk array
        self.last_chunk_start_step: Dict[str, int] = {}  # device_name -> step_counter when chunk started

        if self.seed is not None:
            np.random.seed(self.seed)

    def _build_params(self, obs_dim: int, action_dim: int) -> None:
        layer_dims = [obs_dim] + self.hidden_sizes + [action_dim]
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []

        for in_dim, out_dim in zip(layer_dims[:-1], layer_dims[1:]):
            # He initialization for hidden layers; smaller scale for output
            scale = np.sqrt(2.0 / in_dim) if out_dim in self.hidden_sizes else np.sqrt(1.0 / in_dim)
            W = scale * np.random.randn(in_dim, out_dim).astype(np.float64)
            b = np.zeros((out_dim,), dtype=np.float64)
            self.weights.append(W)
            self.biases.append(b)

        self.params_built = True

    def _forward(self, x: np.ndarray) -> np.ndarray:
        h = x
        for i in range(len(self.hidden_sizes)):
            h = h @ self.weights[i] + self.biases[i]
            h = _relu(h)
        # output layer
        y = h @ self.weights[-1] + self.biases[-1]
        if self.output_tanh:
            y = _tanh(y)
        return y

    def _extract_current_action_from_observation(
            self,
            device_name: str,
            action_dim: int,
            observation: ObservationType
    ) -> Optional[np.ndarray]:
        """
        Extract current executed action from observation.

        Args:
            device_name: Name of the device
            action_dim: Expected action dimension
            observation: Current observation (dict or array) containing current action values

        Returns:
            Current action array of shape (action_dim,), or None if cannot extract
        """
        try:
            if isinstance(observation, dict):
                # Try to get device-specific observation
                if device_name in observation:
                    current_action = np.asarray(observation[device_name]).reshape(-1)
                    if len(current_action) >= action_dim:
                        return current_action[:action_dim]

                # Try to find action-related keys (e.g., device_name + "_action", "action", etc.)
                for key in [f"{device_name}_action", "action", "current_action"]:
                    if key in observation:
                        current_action = np.asarray(observation[key]).reshape(-1)
                        if len(current_action) >= action_dim:
                            return current_action[:action_dim]

                # If device name not found, try to extract from concatenated observation
                # Assume observation contains action values at the beginning
                obs_vec = self._to_vector(observation)
                if len(obs_vec) >= action_dim:
                    return obs_vec[:action_dim]
            else:
                # If observation is array, use first action_dim values
                obs_array = np.asarray(observation).reshape(-1)
                if len(obs_array) >= action_dim:
                    return obs_array[:action_dim]

            return None
        except Exception as e:
            logger.warning(f"Failed to extract current action from observation for {device_name}: {e}")
            return None

    def _find_matching_position_in_last_chunk(
            self,
            device_name: str,
            current_action: np.ndarray,
            tolerance: float = 0.1
    ) -> Optional[int]:
        """
        Find the position in last chunk that best matches current action.

        Args:
            device_name: Name of the device
            current_action: Current executed action array
            tolerance: Maximum allowed difference for matching

        Returns:
            Index in last chunk (0 to chunk_length-1), or None if no match found
        """
        if device_name not in self.last_action_chunks:
            return None

        last_chunk = self.last_action_chunks[device_name]
        if last_chunk.shape[1] != len(current_action):
            return None

        # Find the position with minimum total difference
        best_match_idx = None
        min_diff = float('inf')

        for idx in range(len(last_chunk)):
            chunk_action = last_chunk[idx]
            # Calculate L2 distance
            diff = np.linalg.norm(chunk_action - current_action)
            if diff < min_diff:
                min_diff = diff
                best_match_idx = idx

        # Check if match is within tolerance
        if min_diff <= tolerance * len(current_action):
            return best_match_idx

        # If no good match, return the best match anyway (might be due to noise)
        return best_match_idx

    def _compute_step_counter_for_device(
            self,
            device_name: str,
            action_dim: int,
            observation: ObservationType,
            chunk_length: int
    ) -> int:
        """
        Compute the appropriate step_counter for generating continuous action chunk.

        Args:
            device_name: Name of the device
            action_dim: Action dimension
            observation: Current observation
            chunk_length: Length of chunk to generate

        Returns:
            step_counter value to use for this device
        """
        # Extract current action from observation
        current_action = self._extract_current_action_from_observation(
            device_name, action_dim, observation
        )

        if current_action is None:
            # Cannot extract current action, use global step_counter
            return self.step_counter

        # Try to find matching position in last chunk
        match_idx = self._find_matching_position_in_last_chunk(device_name, current_action)

        if match_idx is not None and device_name in self.last_chunk_start_step:
            # Found match in last chunk, compute new step_counter
            last_start_step = self.last_chunk_start_step[device_name]
            # Current action corresponds to position match_idx in last chunk
            # So we should start new chunk from (last_start_step + match_idx + 1)
            new_step_counter = last_start_step + match_idx + 1
            logger.debug(
                f"Device {device_name}: Found match at index {match_idx} in last chunk. "
                f"Last start step: {last_start_step}, New step_counter: {new_step_counter}"
            )
            return new_step_counter
        else:
            # No match found or no last chunk, infer step_counter from current action
            # Try to infer phase from current action value
            device_offset = hash(device_name) % 10 * 0.3
            freq = 0.015

            # For each dimension, try to infer the phase
            inferred_steps = []
            for j in range(action_dim):
                phase_offset = j * 0.5 + device_offset
                current_val = current_action[j]

                # Clamp to valid range for arcsin
                current_val = np.clip(current_val, -1.0, 1.0)

                # Infer phase: sin(phase + offset) = current_val
                # phase + offset = arcsin(current_val) + 2*pi*k
                principal_phase = np.arcsin(current_val)
                inferred_phase = principal_phase - phase_offset

                # Convert phase to step: time_step = step * freq, phase = time_step + offset
                # So: step = (phase - offset) / freq
                # But we have: phase + offset = arcsin(current_val), so phase = arcsin(current_val) - offset
                # Actually: sin(step * freq + offset) = current_val
                # So: step * freq + offset = arcsin(current_val) + 2*pi*k
                # step = (arcsin(current_val) - offset + 2*pi*k) / freq

                # Use principal value (k=0) as estimate
                inferred_step = (principal_phase - phase_offset) / freq
                inferred_steps.append(inferred_step)

            # Use median of inferred steps to be robust to outliers
            if inferred_steps:
                inferred_step_counter = int(np.median(inferred_steps))
                logger.debug(
                    f"Device {device_name}: Inferred step_counter from current action: {inferred_step_counter}"
                )
                return inferred_step_counter

        # Fallback: use global step_counter
        return self.step_counter

    def predict(self, observation: ObservationType, action_configs: List[Dict], chunk_length: int = 1) -> Dict[str, np.ndarray]:
        if observation is None:
            # Return zero actions for all devices
            result = {}
            for config in action_configs:
                device_name = config['device_name']
                action_dim = config['action_dim'] 
                result[device_name] = np.zeros((chunk_length, action_dim), dtype=np.float64)
            return result
        
        obs_vec = self._to_vector(observation)
        
        # Build params if needed (use total action dim for network output)
        total_action_dim = sum(config['action_dim'] for config in action_configs)
        if not self.params_built:
            self._build_params(obs_dim=obs_vec.shape[0], action_dim=total_action_dim)
        
        # Generate predictable action patterns for debugging
        action_chunks = {}
        freq = 0.015  # Frequency for sine wave

        
        for config in action_configs:
            device_name = config['device_name']
            action_dim = config['action_dim']

            # Compute step_counter for this device based on current observation
            device_step_counter = self._compute_step_counter_for_device(
                device_name, action_dim, observation, chunk_length
            )

            # Store the start step for this chunk
            self.last_chunk_start_step[device_name] = device_step_counter

            # Generate action chunk starting from device_step_counter
            actions = []
            for i in range(chunk_length):
                # Create a sine wave pattern that cycles from -1 to 1
                # Each device and dimension has a different pattern
                action = np.zeros(action_dim, dtype=np.float64)
                for j in range(action_dim):
                    # Use time-based sine wave with different frequencies
                    time_step = (i + device_step_counter) * freq
                    # Different phase for each device and dimension
                    device_offset = hash(device_name) % 10 * 0.3  # Device-specific offset
                    phase_offset = j * 0.5 + device_offset
                    action[j] = np.sin(time_step + phase_offset)
                
                actions.append(action)
            
            action_chunks[device_name] = np.array(actions, dtype=np.float64)

            # Store this chunk for next prediction
            self.last_action_chunks[device_name] = action_chunks[device_name].copy()

        logger.info(f"action_chunks: {action_chunks}")

        # Simulate prediction delay (random between 200-400ms to mimic real inference time)
        prediction_delay_sec = np.random.uniform(0.2, 0.4)
        logger.debug(f"Policy prediction delay: {prediction_delay_sec*1000:.2f} ms")
        time.sleep(prediction_delay_sec)

        # Update global step_counter to the maximum device_step_counter + chunk_length
        # This ensures we don't go backwards
        max_device_step = max(
            self.last_chunk_start_step.values() if self.last_chunk_start_step else [self.step_counter]
        )
        self.step_counter = max(self.step_counter, max_device_step + chunk_length)
        logger.info(f"Policy prediction step: {self.step_counter}, device steps: {self.last_chunk_start_step}")
        
        return action_chunks 