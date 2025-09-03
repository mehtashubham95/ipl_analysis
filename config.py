"""Configuration settings for IPL Analysis project."""

import os
from pathlib import Path

class Config:
    """Configuration class for IPL Analysis."""
    
    # Paths
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    MODELS_DIR = BASE_DIR / "models" / "saved"
    OUTPUTS_DIR = BASE_DIR / "outputs"
    
    # Data files
    MAIN_DATA_FILE = "ipl_08_22_bob_data.csv"
    PLAYER_CHAR_FILE = "IPL_Player_Char.csv"
    SALARY_DATA_FILE = "ipl_player_salary.csv"
    PLAYER_LOOKUP_FILE = "sal_player_lu.csv"
    SEMIFINAL_FLAG_FILE = "semi_final_flag.csv"
    
    # Model parameters
    BATTING_ALLROUNDER_WEIGHTS = {"batting": 0.6, "bowling": 0.4}
    BOWLING_ALLROUNDER_WEIGHTS = {"batting": 0.4, "bowling": 0.6}
    BOWL_RAA_MULTIPLIER = 1.5
    
    # Analysis parameters
    MIN_SEASON = 2010
    CURRENT_SEASON = 2023
    
    # Model hyperparameters
    XGB_PARAMS = {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'random_state': 42
    }
    
    LOGISTIC_PARAMS = {
        'random_state': 42,
        'max_iter': 1000
    }
    
    # Cross-validation
    CV_FOLDS = 5
    TEST_SIZE = 0.2
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist."""
        for directory in [cls.DATA_DIR, cls.MODELS_DIR, cls.OUTPUTS_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
