"""Data loading utilities for IPL Analysis."""

import pandas as pd
from pathlib import Path
from typing import Dict, Optional

from config import Config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class IPLDataLoader:
    """Class to handle loading of IPL datasets."""
    
    def __init__(self, data_path: Optional[Path] = None):
        """
        Initialize data loader.
        
        Args:
            data_path: Path to data directory
        """
        self.data_path = data_path or Config.DATA_DIR
        
    def load_main_data(self) -> pd.DataFrame:
        """Load main ball-by-ball data."""
        filepath = self.data_path / Config.MAIN_DATA_FILE
        return self._load_csv_safely(filepath, "main data")
    
    def load_player_characteristics(self) -> pd.DataFrame:
        """Load player characteristics data."""
        filepath = self.data_path / Config.PLAYER_CHAR_FILE
        return self._load_csv_safely(filepath, "player characteristics")
    
    def load_salary_data(self) -> pd.DataFrame:
        """Load player salary data."""
        filepath = self.data_path / Config.SALARY_DATA_FILE
        return self._load_csv_safely(filepath, "salary data")
    
    def load_player_lookup(self) -> pd.DataFrame:
        """Load player lookup data."""
        filepath = self.data_path / Config.PLAYER_LOOKUP_FILE
        return self._load_csv_safely(filepath, "player lookup")
    
    def load_semifinal_flags(self) -> pd.DataFrame:
        """Load semifinal flags data."""
        filepath = self.data_path / Config.SEMIFINAL_FLAG_FILE
        return self._load_csv_safely(filepath, "semifinal flags")
    
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load all required datasets.
        
        Returns:
            Dictionary containing all datasets
        """
        logger.info("Loading all IPL datasets...")
        
        datasets = {
            'main_data': self.load_main_data(),
            'player_char': self.load_player_characteristics(),
            'salary_data': self.load_salary_data(),
            'player_lookup': self.load_player_lookup(),
            'semifinal_flags': self.load_semifinal_flags()
        }
        
        logger.info("All datasets loaded successfully")
        return datasets
    
    def _load_csv_safely(self, filepath: Path, description: str) -> pd.DataFrame:
        """
        Safely load CSV file with error handling.
        
        Args:
            filepath: Path to CSV file
            description: Description for logging
            
        Returns:
            Loaded DataFrame
        """
        try:
            logger.info(f"Loading {description} from {filepath}")
            df = pd.read_csv(filepath)
            logger.info(f"Loaded {description}: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
            
        except FileNotFoundError:
            logger.error(f"File not found: {filepath}")
            raise
            
        except pd.errors.EmptyDataError:
            logger.error(f"Empty file: {filepath}")
            raise
            
        except Exception as e:
            logger.error(f"Error loading {description}: {str(e)}")
            raise
