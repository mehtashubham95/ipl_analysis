"""IPL Analysis Package."""

__version__ = "1.5.01
__author__ = "mehtaShubham95"

# Import main components for easy access
from .data.loader import IPLDataLoader
from .data.preprocessor import PlayerStatsProcessor
from .models.player_evaluation import PlayerEvaluationModel
from .utils.metrics import mape, rmspe
from .utils.logger import setup_logger

__all__ = [
    "IPLDataLoader",
    "PlayerStatsProcessor",
    "PlayerEvaluationModel",
    "mape",
    "rmspe",
    "setup_logger"
]
