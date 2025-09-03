"""Custom metrics for IPL Analysis."""

import numpy as np
from typing import Union
import pandas as pd


def mape(y_actual: Union[np.ndarray, pd.Series], 
         y_predicted: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculate Mean Absolute Percentage Error.
    
    Args:
        y_actual: Actual values
        y_predicted: Predicted values
        
    Returns:
        MAPE value as percentage
    """
    return np.mean(np.abs((y_actual - y_predicted) / y_actual)) * 100


def rmspe(y_true: Union[np.ndarray, pd.Series], 
          y_pred: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculate Root Mean Square Percentage Error.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        RMSPE value
    """
    return np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))


def calculate_strike_rate(runs: int, balls: int) -> float:
    """
    Calculate batting strike rate.
    
    Args:
        runs: Total runs scored
        balls: Total balls faced
        
    Returns:
        Strike rate (runs per 100 balls)
    """
    if balls == 0:
        return 0.0
    return (runs / balls) * 100


def calculate_economy_rate(runs: int, overs: float) -> float:
    """
    Calculate bowling economy rate.
    
    Args:
        runs: Runs conceded
        overs: Overs bowled
        
    Returns:
        Economy rate (runs per over)
    """
    if overs == 0:
        return 0.0
    return runs / overs


def calculate_bowling_average(runs: int, wickets: int) -> float:
    """
    Calculate bowling average.
    
    Args:
        runs: Runs conceded
        wickets: Wickets taken
        
    Returns:
        Bowling average
    """
    if wickets == 0:
        return float('inf')
    return runs / wickets
