"""Player evaluation models for IPL Analysis."""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib

from config import Config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class PlayerEvaluationModel:
    """Model for evaluating player performance and potential."""
    
    def __init__(self):
        self.config = Config()
        self.scaler = StandardScaler()
        self.models = {}
        
    def prepare_features(self, data: pd.DataFrame):
        """
        Prepare features and target for training.
        """
        logger.info("Preparing features...")
        
        feature_columns = [
            'cum_runs','cum_balls','cum_innings_bat','cum_bat_dots',
            'cum_runs_given','cum_balls_bowled','cum_wickets',
            'cum_innings_bowl','cum_dot_balls'
        ]
        
        available_features = [col for col in feature_columns if col in data.columns]
        X = data[available_features].fillna(0)
        
        # Simplified target: use Runs Above Average (`raa`) if exists
        if 'raa' in data.columns:
            y = (data['raa'] > data['raa'].median()).astype(int)
        else:
            y = pd.Series([0]*len(data))
        
        return X, y
    
    def train_logistic_model(self, X: pd.DataFrame, y: pd.Series):
        """
        Train logistic regression model.
        """
        logger.info("Training Logistic Regression...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.TEST_SIZE, random_state=42, stratify=y
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        score = model.score(X_test_scaled, y_test)
        logger.info(f"Logistic Regression Accuracy: {score:.3f}")
        
        self.models['logistic'] = model
        return model
    
    def train_xgboost_model(self, X: pd.DataFrame, y: pd.Series):
        """
        Train XGBoost model.
        """
        logger.info("Training XGBoost...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.TEST_SIZE, random_state=42, stratify=y
        )
        
        model = xgb.XGBClassifier(**self.config.XGB_PARAMS)
        model.fit(X_train, y_train)
        
        score = model.score(X_test, y_test)
        logger.info(f"XGBoost Accuracy: {score:.3f}")
        
        self.models['xgboost'] = model
        return model
    
    def predict(self, X: pd.DataFrame, model_type: str = "xgboost") -> np.ndarray:
        """
        Predict using trained model.
        """
        if model_type not in self.models:
            raise ValueError(f"Model {model_type} not trained. Train it first.")
        
        model = self.models[model_type]
        
        if model_type == "logistic":
            X_scaled = self.scaler.transform(X)
            return model.predict_proba(X_scaled)[:, 1]
        return model.predict_proba(X)[:, 1]
    
    def save_models(self, filepath: str):
        """Save trained models to a file."""
        logger.info(f"Saving models to {filepath}")
        joblib.dump({"models": self.models, "scaler": self.scaler}, filepath)
    
    def load_models(self, filepath: str):
        """Load saved models from a file."""
        logger.info(f"Loading models from {filepath}")
        saved = joblib.load(filepath)
        self.models = saved["models"]
        self.scaler = saved["scaler"]
