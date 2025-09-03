"""Data preprocessing utilities for IPL Analysis."""

import pandas as pd
import numpy as np

from config import Config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class PlayerStatsProcessor:
    """Class to process and calculate player statistics."""
    
    def __init__(self):
        """Initialize the processor."""
        self.config = Config()
    
    def calculate_batting_stats(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate batting statistics for players.
        """
        logger.info("Calculating batting statistics...")
        
        batting_stats = data[data["wide_runs"] == 0].groupby([
            "Strike Batsman Lookup Name", "Season", "Batsman Type", 
            "Batsman Role", "Batting team"
        ]).agg({
            'batsman_runs': 'sum',
            'batsman': 'count',
            'match_id': 'nunique'
        }).reset_index()
        
        batting_stats.columns = [
            "batsman", "Season", "Batsman Type", 
            "Batsman Role", "team", "runs", "balls", "innings"
        ]
        
        # Dismissals
        dismissals = data[data["player_dismissed"].notnull()].groupby([
            "Strike Batsman Lookup Name", "Season", "Batsman Type", 
            "Batsman Role", "Batting team"
        ])['player_dismissed'].count().reset_index(name="out")
        dismissals = dismissals.rename(columns={"Strike Batsman Lookup Name": "batsman",
                                                "Batting team": "team"})

        # Dot balls
        dot_balls = data[data["batsman_runs"] == 0].groupby([
            "Strike Batsman Lookup Name", "Season", "Batsman Type", 
            "Batsman Role", "Batting team"
        ])['batsman'].count().reset_index(name="bat_dots")
        dot_balls = dot_balls.rename(columns={"Strike Batsman Lookup Name": "batsman",
                                              "Batting team": "team"})
        
        # Merge
        batting_stats = batting_stats.merge(dismissals, 
                                            on=["batsman", "Season", "Batsman Type", "Batsman Role", "team"], 
                                            how="left")
        batting_stats = batting_stats.merge(dot_balls, 
                                            on=["batsman", "Season", "Batsman Type", "Batsman Role", "team"], 
                                            how="left")
        
        logger.info(f"Calculated batting stats for {len(batting_stats)} records")
        return batting_stats
    
    def calculate_bowling_stats(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate bowling statistics for players.
        """
        logger.info("Calculating bowling statistics...")
        
        bowling_stats = data[data["wide_runs"] == 0].groupby([
            "Bowler Lookup Name", "Season", "Bowler Type", 
            "Bowler Role", "Bowling Team"
        ]).agg({
            'total_runs': 'sum',
            'batsman': 'count',
            'match_id': 'nunique'
        }).reset_index()
        
        bowling_stats = bowling_stats.rename(columns={
            "Bowler Lookup Name": "bowler",
            "Bowling Team": "team",
            "total_runs": "runs",
            "batsman": "balls",
            "match_id": "innings"
        })
        
        # Wickets
        wickets = data[data["player_dismissed"].notnull()].groupby([
            "Bowler Lookup Name", "Season", "Bowler Type", 
            "Bowler Role", "Bowling Team"
        ])['player_dismissed'].count().reset_index(name="wickets")
        wickets = wickets.rename(columns={"Bowler Lookup Name": "bowler",
                                          "Bowling Team": "team"})
        
        # Dot balls
        dot_balls = data[data["total_runs"] == 0].groupby([
            "Bowler Lookup Name", "Season", "Bowler Type", 
            "Bowler Role", "Bowling Team"
        ])['batsman'].count().reset_index(name="dot_balls")
        dot_balls = dot_balls.rename(columns={"Bowler Lookup Name": "bowler",
                                              "Bowling Team": "team"})
        
        # Merge
        bowling_stats = bowling_stats.merge(wickets, 
                                            on=["bowler","Season","Bowler Type","Bowler Role","team"],
                                            how="left")
        bowling_stats = bowling_stats.merge(dot_balls, 
                                            on=["bowler","Season","Bowler Type","Bowler Role","team"],
                                            how="left")
        
        logger.info(f"Calculated bowling stats for {len(bowling_stats)} records")
        return bowling_stats
    
    def process_all_stats(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Process all player statistics.
        """
        logger.info("Starting complete player statistics processing...")
        
        batting_stats = self.calculate_batting_stats(data)
        bowling_stats = self.calculate_bowling_stats(data)
        
        # Merge batting and bowling stats
        player_stats = batting_stats.merge(
            bowling_stats, 
            how="outer", 
            left_on=["batsman","Season"], 
            right_on=["bowler","Season"]
        ).fillna(0)
        
        player_stats = player_stats.rename(columns={"batsman": "Player"})
        
        logger.info("Player statistics processing completed")
        return player_stats
