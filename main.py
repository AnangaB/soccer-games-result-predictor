import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error, roc_auc_score, root_mean_squared_error
from sklearn.preprocessing import FunctionTransformer

import numpy as np
from sklearn.svm import SVC, SVR
class InfrequentDistributionImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.value_frequencies_ = {}
        for column in X.columns:
            # Count occurrences of each value
            value_counts = X[column].value_counts(normalize=True)  # Get normalized frequencies
            # Sort values by frequency (increasing order) and invert for sampling
            self.value_frequencies_[column] = value_counts.sort_values().index.tolist()
        return self

    def transform(self, X):
        X_imputed = X.copy()
        for column in X_imputed.columns:
            # Impute missing values based on the inverted frequency distribution
            missing_mask = X_imputed[column].isna()
            if missing_mask.any():  # Only process if there are missing values
                    # Get counts for the current column and compute the least frequent values
                value_counts = X_imputed[column].value_counts()
                values = value_counts.sort_values().index.tolist()

                # Calculate weights based on the frequency counts
                weights = [1 - (count / value_counts.sum()) for count in value_counts.values]

                # Ensure that weights are normalized
                if(sum(weights) == 0):
                    print(value_counts)
                weights = np.nan_to_num(np.array(weights), nan=0)

                weights = np.array(weights) / sum(weights)
                # Sample from the least frequent values based on calculated weights
                sampled_values = np.random.choice(
                values, 
                size=missing_mask.sum(), 
                p=weights
                )

                X_imputed.loc[missing_mask, column] = sampled_values
        return X_imputed

def calculate_score(true_winner, predicted_winner, true_winby, predicted_winby):
    # Calculate AUC for the winner predictions
    auc = roc_auc_score(true_winner, predicted_winner)
    
    # RMSE for the predicted winby
    rmse_pred = np.sqrt(root_mean_squared_error (true_winby, predicted_winby))
    
    # RMSE for the baseline (all ones) winby
    ones_vector = np.ones_like(true_winby)
    rmse_baseline = np.sqrt(root_mean_squared_error (ones_vector, true_winby))
    
    # Score calculation
    score = .50 * 2 * (auc - 0.5) + .50 * (1 - rmse_pred / rmse_baseline)
    
    return score

# Function to add players' average rating to the data
def add_players_avg_rating(X, player_avg_overall_rating_dict):
    X = X.copy()
    home_players_cols = ["home_player_" + str(i) for i in range(1, 12)]
    away_players_cols = ["away_player_" + str(i) for i in range(1, 12)]
    player_cols = home_players_cols + away_players_cols
    for col in player_cols:
            avg_rating_col = col + "_avg_rating"
            X[avg_rating_col] = X[col].map(player_avg_overall_rating_dict)

            # Impute missing values wth 30
            X[avg_rating_col].fillna(30, inplace=True)
    return X

#
def remove_player_ids(df):
    # List of home and away player columns
    home_players_cols = ["home_player_" + str(i) for i in range(1, 12)]
    away_players_cols = ["away_player_" + str(i) for i in range(1, 12)]
    home_playersX_cols = ["home_player_X" + str(i) for i in range(1, 12)]
    away_playersX_cols = ["away_player_X" + str(i) for i in range(1, 12)]
    home_playersY_cols = ["home_player_Y" + str(i) for i in range(1, 12)]
    away_playersY_cols = ["away_player_Y" + str(i) for i in range(1, 12)]


    # Combine both lists
    all_player_cols = home_players_cols + away_players_cols
    xy_cols = home_playersX_cols + away_playersX_cols  + home_playersY_cols + away_playersY_cols
    df[all_player_cols] = df[all_player_cols].fillna(0)
    return df

    # Drop the player ID columns

    #return df.drop(columns=all_player_cols)
# Function to add past win rates for home and away teams
def add_past_win_rate(df, win_rate_map):
    df = df.copy()  # To avoid modifying the original DataFrame
    # Map past win rates for home teams
    df['home_team_past_win_rate'] = df.apply(
        lambda row: win_rate_map.get((row["home_team_id"], row["away_team_id"]), 0), axis=1)*100

    df.to_csv("training_data.csv")
    return df

#calculate team_wins
def get_home_team_win_rate(games):
    # keep track of how many wins each team has agaisnt the other
   # Create a dictionary to count how many wins the home_team has against the away_team in the past
    team_wins = {}
    team_matches = {}  # To count total games between home and away teams

    for _, row in games.iterrows():
        home_team = row['home_team_id']
        away_team = row['away_team_id']
        winby = row['winby']
        if home_team and away_team:
            # Count the total games played between the home and away teams
            if (home_team, away_team) in team_matches:
                team_matches[(home_team, away_team)] += 1
            else:
                team_matches[(home_team, away_team)] = 1
            
            # Count wins for home team against away team
            if winby > 0:  # Home team won
                if (home_team, away_team) in team_wins:
                    team_wins[(home_team, away_team)] += 1
                else:
                    team_wins[(home_team, away_team)] = 1

    # Create a dictionary to store the win rate
    team_win_rate = {}

    for t in team_wins:
        # Calculate win rate as number of wins / total matches played
        team_win_rate[t] = team_wins[t] / team_matches[t]
    return team_win_rate

#add other player info like attacking_work_rate,defensive_work_rate
def add_player_work_rate_columns(df,avg_work_rates_df):
    home_players_cols = ["home_player_" + str(i) for i in range(1, 12)]
    away_players_cols = ["away_player_" + str(i) for i in range(1, 12)]
    player_cols = home_players_cols + away_players_cols

    ids_in_avg_work_rates_df = avg_work_rates_df.index.unique()
    for col in player_cols:
        # Extract player_id from the column name (e.g., "home_player_1" to get player_id)
        player_id_col = df[col]
        
        def get_work_rate(id, is_attack):
            if id in ids_in_avg_work_rates_df:
                if is_attack:
                    return avg_work_rates_df.loc[id,"attacking_work_rate"]
                else:
                    return avg_work_rates_df.loc[id,"defensive_work_rate"]
            return 0
        # Map work rates based on player IDs for both attacking and defensive work rates
        df[col + "_attack_work_rate"] = player_id_col.map(lambda id: get_work_rate(id,True))
        df[col + "_defense_work_rate"] = player_id_col.map(lambda id: get_work_rate(id,False))

    return df

# adds teams total crossigns
def add_player_crossing_row(df, player_crossings_map):
    home_players_cols = ["home_player_" + str(i) for i in range(1, 12)]
    away_players_cols = ["away_player_" + str(i) for i in range(1, 12)]
    
    # Initialize total_crossings to zero
    df["total_crossings_home"] = 0
    df["total_crossings_away"] = 0

    for col in home_players_cols:
        # Extract player_id from the column name (e.g., "home_player_1" to get player_id)
        player_id_col = pd.to_numeric(df[col], errors='coerce')
        df["total_crossings_home"] += player_id_col.map(lambda b: player_crossings_map.get(b,0))
    for col in away_players_cols:
        # Extract player_id from the column name (e.g., "home_player_1" to get player_id)
        player_id_col = pd.to_numeric(df[col], errors='coerce')
        df["total_crossings_away"] += player_id_col.map(lambda b: player_crossings_map.get(b,0))
        
    return df
#subset only a few rows:
def condense_columns(df):
    # Player averages and work rates for home and away teams
    num_players = 11  # Total number of players

    # Generate home player average ratings
    home_avg_ratings = [f'home_player_{i+1}_avg_rating' for i in range(num_players)]
    # Generate away player average ratings
    away_avg_ratings = [f'away_player_{i+1}_avg_rating' for i in range(num_players)]

    # Generate work rate columns for home and away players
    home_attack_work_rates = [f'home_player_{i+1}_attack_work_rate' for i in range(num_players)]
    home_defense_work_rates = [f'home_player_{i+1}_defense_work_rate' for i in range(num_players)]
    away_attack_work_rates = [f'away_player_{i+1}_attack_work_rate' for i in range(num_players)]
    away_defense_work_rates = [f'away_player_{i+1}_defense_work_rate' for i in range(num_players)]

    # Combine all columns into a single list
    cols = (
        home_avg_ratings +
        away_avg_ratings +
        home_attack_work_rates +
        home_defense_work_rates +
        away_attack_work_rates +
        away_defense_work_rates
    )

    # Step 2: Calculate the means
    mean_home_avg_rating = df[home_avg_ratings].mean(axis=1)
    mean_away_avg_rating = df[away_avg_ratings].mean(axis=1)
    mean_home_attack_work_rate = df[home_attack_work_rates].mean(axis=1)
    mean_home_defense_work_rate = df[home_defense_work_rates].mean(axis=1)
    mean_away_attack_work_rate = df[away_attack_work_rates].mean(axis=1)
    mean_away_defense_work_rate = df[away_defense_work_rates].mean(axis=1)

    # Step 3: Drop the original columns
    df.drop(columns=home_avg_ratings + away_avg_ratings +
            home_attack_work_rates + home_defense_work_rates +
            away_attack_work_rates + away_defense_work_rates, inplace=True)

    # Step 4: Add the mean columns
    df['mean_home_avg_rating'] = mean_home_avg_rating
    df['mean_away_avg_rating'] = mean_away_avg_rating
    df['mean_home_attack_work_rate'] = mean_home_attack_work_rate
    df['mean_home_defense_work_rate'] = mean_home_defense_work_rate
    df['mean_away_attack_work_rate'] = mean_away_attack_work_rate
    df['mean_away_defense_work_rate'] = mean_away_defense_work_rate
    df.to_csv("training_data.csv")

    return df

def get_player_potential_rows(df, player_id_to_potential_map):
    home_players_cols = ["home_player_" + str(i) for i in range(1, 12)]
    away_players_cols = ["away_player_" + str(i) for i in range(1, 12)]
    player_cols = home_players_cols + away_players_cols
    for col in player_cols:
        # Extract player_id from the column name (e.g., "home_player_1" to get player_id)
        player_id_col = df[col]
        
        # Map work rates based on player IDs for both attacking and defensive work rates
        df[col + "_potential"] = player_id_col.map(lambda id: player_id_to_potential_map.get(id,50))

    return df
    
def __main__():
    test = pd.read_csv("project01/test.csv")
    games = pd.read_csv("project01/games.csv")
    players = pd.read_csv("project01/players.csv")

    games["winby"] = games["home_team_goal"] - games["away_team_goal"]


    games = games[(games["winby"] < 9)]

    #add data for players into training set
    player_avg_overall_rating = players[["player_id", "overall_rating"]].groupby("player_id").agg("mean")
    player_avg_overall_rating["overall_rating"] = player_avg_overall_rating["overall_rating"]
    player_avg_overall_rating_dict = player_avg_overall_rating["overall_rating"].to_dict()

    
    # Create a FunctionTransformer that wraps the add_players_avg_rating function
    add_avg_player_rating = FunctionTransformer(add_players_avg_rating, kw_args={'player_avg_overall_rating_dict': player_avg_overall_rating_dict})


    # Work rate mapping
    attack_work_rate_map = {"medium": 50, "low": 0, "high": 100}
    defense_work_rate_map = {"medium": 50, "low": 0, "high": 100}

    # Map work rates for both attacking and defensive work rates
    players["attacking_work_rate"] = players["attacking_work_rate"].map(lambda r: attack_work_rate_map.get(r,0))
    players["defensive_work_rate"] = players["defensive_work_rate"].map(lambda r: defense_work_rate_map.get(r,0)) 

    # Group by player_id and calculate the mean for work rates
    avg_work_rates_df = players[["player_id", "attacking_work_rate", "defensive_work_rate"]].groupby("player_id").agg({"attacking_work_rate": "mean", "defensive_work_rate": "mean"})

    add_player_work_rates = FunctionTransformer(add_player_work_rate_columns, kw_args={'avg_work_rates_df': avg_work_rates_df})

    #remove player ids
    remove_player_id_cols = FunctionTransformer(remove_player_ids)

    
    #get winrate of hometeams
    team_win_rate = get_home_team_win_rate(games)
    add_home_team_past_wins = FunctionTransformer(add_past_win_rate, kw_args={'win_rate_map': team_win_rate})

    #add player crossings
    players["crossing"].fillna(0,inplace = True)
    player_crossings_map = players.groupby("player_id")["crossing"].max().to_dict()
    add_player_crossings_columns = FunctionTransformer(add_player_crossing_row, kw_args={'player_crossings_map': player_crossings_map})


    #combine some of the columns by replacing with avg
    combine_columns = FunctionTransformer(condense_columns)

    #add player potential
    players["potential"].fillna(players["potential"].mean(), inplace=True)
    player_id_to_potential_map = players[["player_id","potential"]].groupby("player_id")["potential"].max().to_dict()
    add_player_potential_cols = FunctionTransformer(get_player_potential_rows, kw_args={'player_id_to_potential_map': player_id_to_potential_map})

    # Define features and target
    X = games.drop(columns=["winby", "home_team_goal", "away_team_goal", "year"])  # Drop the target variable
    y = games["winby"]



    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


    if y_test.isna().any():
        print("There are NaN values in the series.")
    else:
        print("No NaN values in the series.")
    # Initialize the pipeline
    """
    model_winby = make_pipeline(
        add_avg_player_rating,  # Add player avg rating
        add_player_work_rates,
        add_player_crossings_columns,                
        add_player_potential_cols,
        remove_player_id_cols,
        add_home_team_past_wins,
        SimpleImputer(strategy="median"),
        HistGradientBoostingClassifier()
        #RandomForestClassifier(n_estimators=300)
    )"""
    model_winby = make_pipeline(
        add_avg_player_rating,  # Add player avg rating               
        add_player_potential_cols,
        SimpleImputer(strategy="median"),
        #LinearRegression()
        HistGradientBoostingClassifier()
    )

    # Fit the model with one-hot encoded targets
    model_winby.fit(X_train, y_train)

    # Make predictions
    y_test_predict_winby = model_winby.predict(X_test)

    # Flatten y_test and y_test_predict_winby to ensure both are 1D
    y_test_predict_winby = pd.Series(y_test_predict_winby.flatten())
    

    # Create DataFrame with predicted and actual values
    y_test_df = pd.DataFrame({
        "predicted": y_test_predict_winby,
    })   
    y_test_df["actual"] =  y_test.values
    y_test_df.to_csv("ytestdf.csv")

    # Calculate model score
    winby_score = model_winby.score(X_test, y_test)
    print("winby_model_score: ", winby_score)
   
    # Map predicted class indices back to original classes
    win_by_prediction = model_winby.predict(test)

    # Convert to DataFrame for final predictions
    prediction_for_test = pd.DataFrame({'winby': win_by_prediction})


    ######
     # Define features and target
    X = games.drop(columns=["winby","home_team_goal","away_team_goal","year"])  # Drop the target variable
    y = games["winby"].map(lambda w: 0 if w > 0 else 1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Initialize the pipeline
    """model_winner = make_pipeline(
        add_avg_player_rating,  # Add player avg rating
        add_player_work_rates,
        add_player_crossings_columns,                
        add_player_potential_cols,
        remove_player_id_cols,   
        add_home_team_past_wins,
        SimpleImputer(strategy="median"),
        #LinearRegression()
        RandomForestRegressor(n_estimators=300)
    )"""
    model_winner = make_pipeline(
        add_avg_player_rating,  # Add player avg rating               
        add_player_potential_cols,
        SimpleImputer(strategy="median"),
        #LinearRegression()
        RandomForestRegressor(n_estimators=300)
    )
    
    model_winner.fit(X_train, y_train)
    # Make predictions on the test set
    winner_score = model_winner.score(X_test, y_test)
    print("winer_model_score: ", winner_score)

    winner_predictions = model_winby.predict(X_test)

    # Calculate RMSE for win-by predictions
    rmse = root_mean_squared_error(y_test, winner_predictions)
    print("RMSE for winner predictions: ", rmse)
    
    
    
    prediction_for_test["winner"] =  model_winner.predict(test)

    # Save the DataFrame to CSV
    prediction_for_test[["winner","winby"]].to_csv("pred_for_test.csv", index=None)
    



if __name__ == "__main__":
    __main__()