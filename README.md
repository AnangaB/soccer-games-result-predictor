# Soccer Match Outcome Prediction

This repository contains code for predicting soccer match outcomes using machine learning models. It predicts two key metrics for each match: the winning team and the "win-by" margin. 
The project leverages `XGBoost` and data preprocessing through custom feature functions.

## Table of Contents
- [Project Overview](#project-overview)
- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Files and Functions](#files-and-functions)
- [Contributing](#contributing)

## Project Overview

This is a group project for **SFU STAT 440: Learning from Big Data**. The project builds two models:
1. **Win-by Model** - Predicts the "win-by" margin, or the difference in goals between the winning and losing team.
2. **Winner Model** - Predicts the winning team based on custom team and player features. 1 represents home team and 0 represents away team.

The models use the following data files:
- `data/games.csv`: Contains game data, including goals scored by home and away teams.
- `data/players.csv`: Contains player statistics.

Feature engineering is handled through custom functions for player ratings, work rate, and crossing scores.

## Directory Structure

```
Project/
├── data/                     # Contains input CSV data files (games.csv, players.csv, test.csv)
├── Features/                 # Directory with feature engineering functions
├── Output/                   # Output directory for predictions
├── main.py                   # Main script for training and prediction
└── README.md                 # Project documentation
```

## Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd Project
   ```

2. Install the required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To run the prediction script:
```bash
python main.py
```

The script reads data from `data/` directory, processes it, and outputs a CSV file `pred_for_test.csv` in the `Output/` directory containing predictions for the `winner` and `winby` columns.

## Files and Functions

### `main.py`
- **Imports** feature engineering functions from the `Features/` directory.
- **Defines models** for predicting `winby` (win margin) and `winner`.
- **Preprocessing pipeline** includes custom feature extraction, data imputation, and `XGBoost` regressor.
- Outputs predictions to `Output/pred_for_test.csv`.

### Feature Functions (in `Features/`)
- **condense_columns**: Removes unnecessary columns in the preprocessing pipeline.
- **add_player_work_rate_columns**: Adds teams avg attack work rate, avg defense work rate and max attack work rate in each team.
- **add_team_avg_rating**: Adds team average player ratings.
- **get_team_avg_crossings**: Adds average crossing value for each team.
