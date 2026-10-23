    
from . import BaseChunkManager
import time
import numpy as np
from utils.logger_config import logger
from typing import Dict, List, Any, Optional, Tuple
from utils.logger_config import logger

class NoDelayChunkManager(BaseChunkManager):
    """
    This manager eliminates the delaying actions from each new coming action chunk.

    Args:
        dt (float): the duration between a pair of action steps 
    """
    def __init__(self, dt: float = 0.05):
        super().__init__()
        self.dt = int(dt * 1e9) # s -> ns
    
    def put(self, action_chunk_dict: Dict[str, np.ndarray], observation_timestamp: Optional[int]=None) -> None:
        """
        Clip the action chunk to drop the undesired pose caused by inference latency
        """
        if self.current_action_chunk_dict is None:
            super().put(action_chunk_dict, observation_timestamp)
            return
        delay = time.time_ns() - observation_timestamp
        delayed_start_idx = delay//self.dt

        # Get the chunk length from any device (they should all have the same chunk length)
        device_names = list(action_chunk_dict.keys())
        first_chunk = action_chunk_dict[device_names[0]]
        chunk_length = len(first_chunk)

        if delayed_start_idx < chunk_length:
            # print("Delay Start Idx:", delayed_start_idx)
            for device_name, chunk in action_chunk_dict.items():
                action_chunk_dict[device_name] = action_chunk_dict[device_name][delayed_start_idx:]
            super().put(action_chunk_dict, observation_timestamp)
        else:
            logger.warning(f"Action chunk was directly dropped due to large delay {delay/1e9}s")        
            

