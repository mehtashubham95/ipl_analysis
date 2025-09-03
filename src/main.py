"""Main entry point for IPL Analysis."""

import argparse
import sys
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from config import Config
from src.data.loader import IPLDataLoader
from src.data.preprocessor import PlayerStatsProcessor
from src.models.player_evaluation import PlayerEvaluationModel
from src.utils.logger import setup_logger

# Setup logging
logger = setup_logger(__name__, Config.BASE_DIR / "logs" / "ipl_analysis.log")


def main():
    """Main function to run IPL analysis."""
    parser = argparse.ArgumentParser(description="IPL Player Analysis and Team Optimization")
    parser.add_argument("--stage", choices=["data_processing", "model_training", "prediction", "all"],
                        default="all", help="Stage of analysis to run")
    parser.add_argument("--data-path", type=str, help="Path to data directory")
    parser.add_argument("--output-path", type=str, help="Path to output directory")
    
    args = parser.parse_args()
    
    try:
        # Create directories
        Config.create_directories()
        
        logger.info("Starting IPL Analysis...")
        logger.info(f"Stage: {args.stage}")
        
        data_loader = IPLDataLoader(Path(args.data_path) if args.data_path else None)
        processor = PlayerStatsProcessor()
        model = PlayerEvaluationModel()
        
        # --- Data Processing ---
        if args.stage in ["data_processing", "all"]:
            datasets = data_loader.load_all_data()
            main_data = datasets['main_data']
            player_stats = processor.process_all_stats(main_data)
            
            output_path = Path(args.output_path) if args.output_path else Config.OUTPUTS_DIR
            output_path.mkdir(parents=True, exist_ok=True)
            
            processed_file = output_path / "processed_player_stats.csv"
            player_stats.to_csv(processed_file, index=False)
            logger.info(f"Processed data saved at {processed_file}")
        
        # --- Model Training ---
        if args.stage in ["model_training", "all"]:
            processed_file = Config.OUTPUTS_DIR / "processed_player_stats.csv"
            if not processed_file.exists():
                logger.error("Processed data not found. Run --stage data_processing first.")
                return
            
            player_stats = pd.read_csv(processed_file)
            X, y = model.prepare_features(player_stats)
            
            model.train_logistic_model(X, y)
            model.train_xgboost_model(X, y)
            
            model_file = Config.MODELS_DIR / "player_evaluation_models.pkl"
            model.save_models(str(model_file))
            logger.info("Models trained and saved successfully.")
        
        # --- Prediction ---
        if args.stage in ["prediction", "all"]:
            model_file = Config.MODELS_DIR / "player_evaluation_models.pkl"
            if not model_file.exists():
                logger.error("Models not found. Run --stage model_training first.")
                return
            
            model.load_models(str(model_file))
            
            processed_file = Config.OUTPUTS_DIR / "processed_player_stats.csv"
            if not processed_file.exists():
                logger.error("Processed data not found. Run --stage data_processing first.")
                return
            
            player_stats = pd.read_csv(processed_file)
            X, _ = model.prepare_features(player_stats)
            
            predictions = model.predict(X, "xgboost")
            player_stats["performance_score"] = predictions
            
            output_path = Path(args.output_path) if args.output_path else Config.OUTPUTS_DIR
            results_file = output_path / "player_predictions.csv"
            player_stats.to_csv(results_file, index=False)
            logger.info(f"Predictions saved to {results_file}")
            
            # Display top 5 predicted performers
            top_players = player_stats.nlargest(5, "performance_score")[["Player", "Season", "performance_score"]]
            logger.info("Top predicted performers:")
            logger.info(f"\n{top_players}")
        
        logger.info("IPL Analysis pipeline completed successfully.")
    
    except Exception as e:
        logger.error(f"Error in IPL Analysis pipeline: {str(e)}")
        raise


if __name__ == "__main__":
    main()
