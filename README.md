# IPL Player Analysis and Team Optimization Framework

## Overview

This project provides an end-to-end framework for Indian Premier League (IPL) franchises to build optimal squads that maximize their chances of qualifying for playoffs. The framework combines descriptive, predictive, and prescriptive analytics to identify key player attributes, predict player salaries, and create optimized team compositions within IPL constraints.

Based on research presented at the MIT Sports Conference, this framework addresses the complex challenge of player drafting in the world's second most valued sporting league, serving an estimated 2.5 billion cricket fans worldwide.

## Key Objectives

1. **Identify key player attributes** associated with team playoff qualification
2. **Formulate player performance scores** using identified attributes
3. **Predict player auction prices** using historical data and machine learning
4. **Build optimal squads** that maximize playoff qualification probability within IPL constraints


## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd ipl_analysis
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up data:**
   - Place all input CSV files in the `data/` directory
   - Ensure the following files are present:
     - `ipl_08_22_bob_data.csv`
     - `IPL_Player_Char.csv`
     - `ipl_player_salary.csv`
     - `sal_player_lu.csv`
     - `semi_final_flag.csv`

## Usage

### Running the Full Pipeline

```bash
python -m src.main --stage all
```

### Running Individual Stages

1. **Data Processing Only:**
   ```bash
   python -m src.main --stage data_processing
   ```

2. **Model Training Only:**
   ```bash
   python -m src.main --stage model_training
   ```

3. **Prediction Only:**
   ```bash
   python -m src.main --stage prediction
   ```

### Command Line Arguments

- `--stage`: Specify which stage to run (`data_processing`, `model_training`, `prediction`, `all`)
- `--data-path`: Custom path to data directory
- `--output-path`: Custom path to output directory

## Methodology

### Stage 1: Descriptive Analysis
- Comprehensive analysis of historical IPL data (2008-2022)
- Identification of key performance metrics and attributes
- Calculation of player scores using logistic regression
- Key metrics include:
  - Batting statistics (runs, strike rate, dot balls)
  - Bowling statistics (wickets, economy rate, dot balls)
  - Player roles and types
  - Team performance indicators

### Stage 2: Predictive Analysis
- Machine learning models for player salary prediction
- XGBoost and Multiple Linear Regression implementations
- Historical auction data analysis
- Feature engineering for price prediction

### Stage 3: Prescriptive Analysis
- Optimization techniques for squad building
- Constraint satisfaction (team composition rules, salary caps)
- Maximization of playoff qualification probability
- Generation of optimal squad recommendations in Excel format

## Key Features

### Data Loading & Preprocessing
- **IPLDataLoader**: Handles loading of all required datasets with error handling
- **PlayerStatsProcessor**: Calculates comprehensive batting and bowling statistics
- Automated data validation and cleaning

### Machine Learning Models
- **PlayerEvaluationModel**: Evaluates player performance and potential
- Multiple model types (Logistic Regression, XGBoost)
- Cross-validation and performance metrics
- Model serialization and loading

### Custom Metrics
- **MAPE** (Mean Absolute Percentage Error)
- **RMSPE** (Root Mean Square Percentage Error)
- Cricket-specific metrics (strike rate, economy rate, bowling average)

### Logging & Configuration
- Comprehensive logging system
- Centralized configuration management
- Directory structure management

## Output Files

### Intermediate Files (CSV)
- `processed_player_stats.csv`: Cleaned and processed player statistics
- `player_predictions.csv`: Machine learning predictions with performance scores

### Final Deliverables (Excel)
- `Optimal_Squad_<TEAM>_<YEAR>_Stage3.xlsx`: Final optimized squad recommendations
  - Player names and roles
  - Performance statistics
  - Predicted salaries
  - Nationality information
  - Retention status

## Example Output

The framework produces optimized squads for the teams

## Dependencies

- pandas >= 1.5.0
- numpy >= 1.21.0
- scikit-learn >= 1.1.0
- xgboost >= 1.6.0
- seaborn >= 0.11.0
- matplotlib >= 3.5.0
- scipy >= 1.9.0
- openpyxl >= 3.0.0
- python-dotenv >= 0.19.0

## Configuration

Edit `config.py` to customize:
- Data file paths and names
- Model hyperparameters
- Analysis parameters (seasons, weights)
- Directory structure

## Contributing

1. Follow the modular structure for new features
2. Add appropriate logging and error handling
3. Update requirements.txt for new dependencies
4. Maintain CSV input/Excel output convention
5. Update documentation accordingly

## Applications

This framework can be:
- Used by IPL franchises for player drafting
- Adapted for other cricket leagues worldwide
- Modified for international tournament team selection
- Extended to other team sports with similar constraints

## Support

For questions about the framework structure or implementation, refer to the code documentation and migration guide.
