#!/usr/bin/env python3
"""
Latency calculation utilities for SuperInference.

This module provides functions for calculating latency between device pairs
using cross-correlation analysis.

Author: SuperInference Team
"""

import numpy as np
from scipy.interpolate import interp1d
import scipy.signal as ss
from typing import Tuple, Dict, Any, Optional


def regular_sample(x: np.ndarray, t: np.ndarray, t_samples: np.ndarray) -> np.ndarray:
    """
    Resample data to regular time intervals using spline interpolation.
    
    Args:
        x: Input data array
        t: Input time array
        t_samples: Target time samples
        
    Returns:
        np.ndarray: Resampled data
    """
    spline = interp1d(x=t, y=x, bounds_error=False, fill_value=(x[0], x[-1]))
    result = spline(t_samples)
    return result


def get_latency(
    x_target: np.ndarray, 
    t_target: np.ndarray, 
    x_actual: np.ndarray, 
    t_actual: np.ndarray, 
    t_start: Optional[float] = None, 
    t_end: Optional[float] = None,
    resample_dt: float = 1/1000,
    force_positive: bool = False
) -> Tuple[float, Dict[str, Any]]:
    """
    Calculate latency between target and actual signals using cross-correlation.
    
    Args:
        x_target: Target signal values
        t_target: Target signal timestamps
        x_actual: Actual signal values
        t_actual: Actual signal timestamps
        t_start: Start time for analysis (default: max of both start times)
        t_end: End time for analysis (default: min of both end times)
        resample_dt: Time step for resampling (default: 1ms)
        force_positive: Force positive latency only (default: False)
        
    Returns:
        Tuple[float, Dict]: (latency, analysis_info)
    """
    assert len(x_target) == len(t_target), "Target data and time arrays must have same length"
    assert len(x_actual) == len(t_actual), "Actual data and time arrays must have same length"
    
    if t_start is None:
        t_start = max(t_target[0], t_actual[0])
    if t_end is None:
        t_end = min(t_target[-1], t_actual[-1])
    
    n_samples = int((t_end - t_start) / resample_dt)
    t_samples = np.arange(n_samples) * resample_dt + t_start
    
    target_samples = regular_sample(x_target, t_target, t_samples)
    actual_samples = regular_sample(x_actual, t_actual, t_samples)

    # Normalize samples to zero mean unit std
    mean = np.mean(np.concatenate([target_samples, actual_samples]))
    std = np.std(np.concatenate([target_samples, actual_samples]))
    target_samples = (target_samples - mean) / std
    actual_samples = (actual_samples - mean) / std

    # Cross correlation
    correlation = ss.correlate(actual_samples, target_samples)
    lags = ss.correlation_lags(len(actual_samples), len(target_samples))
    t_lags = lags * resample_dt

    latency = None
    if force_positive:
        latency = t_lags[np.argmax(correlation[t_lags >= 0])]
    else:
        latency = t_lags[np.argmax(correlation)]
    
    info = {
        't_samples': t_samples,
        'x_target': target_samples,
        'x_actual': actual_samples,
        'correlation': correlation,
        'lags': t_lags
    }

    return latency, info 