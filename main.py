import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier,XGBRegressor

from sklearn.preprocessing import FunctionTransformer

import numpy as np

from get_features.drop_some_columns import condense_columns
from get_features.get_player_avg_potential import get_player_potential_rows
from get_features.get_player_avg_rating import add_players_avg_rating
from get_features.get_player_work_rate import add_player_work_rate_columns
from get_features.get_team_past_win_rate import add_past_win_rate
from get_features.get_player_avg_crossing import get_team_total_crossings

"""
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
"""

def __main__():
    #get required data
    test = pd.read_csv("project01/test.csv")
    games = pd.read_csv("project01/games.csv")
    players = pd.read_csv("project01/players.csv")
    
    #make winner and winby column
    games["winner"] = np.where((games["home_team_goal"] - games["away_team_goal"]) > 0, 0, np.where((games["home_team_goal"] - games["away_team_goal"]) < 0, 1,0.5) )
    games["winby"] = abs(games["home_team_goal"] - games["away_team_goal"])
    games = games[(games["winby"] < 9)]

    """    
    #add data for players into training set
    player_avg_overall_rating = players[["player_id", "overall_rating"]].groupby("player_id").agg("mean")
    player_avg_overall_rating["overall_rating"] = player_avg_overall_rating["overall_rating"]
    player_avg_overall_rating_dict = player_avg_overall_rating["overall_rating"].to_dict()

    
    # Create a FunctionTransformer that wraps the add_players_avg_rating function
    add_avg_player_rating = FunctionTransformer(add_players_avg_rating, kw_args={'player_avg_overall_rating_dict': player_avg_overall_rating_dict})


    # Work rate mapping
    attack_work_rate_map = {"medium": 50, "low": 0, "high": 120}
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
    """


    # Define features and target for calculating winby
    X = games.drop(columns=["winby", "home_team_goal", "away_team_goal", "year","winner"])  # Drop the target variable
    y = games["winby"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,stratify=y)

    model_winby = make_pipeline(
        FunctionTransformer(add_player_work_rate_columns,kw_args={"players_df":players}),
        FunctionTransformer(add_players_avg_rating,kw_args={"players_df":players}),
        FunctionTransformer(add_past_win_rate,kw_args={"games_df":games}),
        FunctionTransformer(get_player_potential_rows,kw_args={"players_df":players}),
        FunctionTransformer(get_team_total_crossings,kw_args={"players_df":players}),
        FunctionTransformer(condense_columns),
        SimpleImputer(strategy="mean"),
        XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss" 
        )    
    )

    model_winby.fit(X_train, y_train)

    #test model_winby using X_test data
    print("Model_winby test data score: ",model_winby.score(X_test,y_test))
    
    #predict winby for test dataset
    win_by_test_prediction = model_winby.predict(test)
    # Convert to DataFrame for final predictions
    prediction_for_test = pd.DataFrame({'winby': win_by_test_prediction})


    #now predicting winner
    y = games["winner"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # model to predict winner
    model_winner = make_pipeline(
        FunctionTransformer(add_player_work_rate_columns,kw_args={"players_df":players}),
        FunctionTransformer(add_players_avg_rating,kw_args={"players_df":players}),
        FunctionTransformer(add_past_win_rate,kw_args={"games_df":games}),
        FunctionTransformer(get_player_potential_rows,kw_args={"players_df":players}),
        FunctionTransformer(get_team_total_crossings,kw_args={"players_df":players}),
        FunctionTransformer(condense_columns),
        SimpleImputer(strategy="mean"),
        XGBRegressor()
    )
    model_winner.fit(X_train, y_train)
    print("Model_winner score:",model_winner.score(X_test,y_test))

    winner_predictions = model_winby.predict(X_test)

    # Calculate RMSE for win-by predictions
    rmse = root_mean_squared_error(y_test, winner_predictions)
    print("RMSE for Model_winner predictions: ", rmse)

    #predict winner values on test dataset
    prediction_for_test["winner"] =  model_winner.predict(test)

    # Save the DataFrame to CSV
    prediction_for_test[["winner","winby"]].to_csv("pred_for_test.csv", index=None)



if __name__ == "__main__":
    __main__()




"""
    model_winby = make_pipeline(
        add_avg_player_rating,  # Add player avg rating
        add_player_work_rates,
        add_player_crossings_columns,                
        add_player_potential_cols,
        #remove_player_id_cols,
        add_home_team_past_wins,
        SimpleImputer(strategy="mean"),
        #HistGradientBoostingClassifier()
        RandomForestClassifier()
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
    X = games.drop(columns=["winby","home_team_goal","away_team_goal","year","winner"]) 
    y = games["winner"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Initialize the pipeline
    model_winner = make_pipeline(
        add_avg_player_rating,  # Add player avg rating
        add_player_work_rates,
        add_player_crossings_columns,                
        add_player_potential_cols,
        #remove_player_id_cols,   
        add_home_team_past_wins,
        SimpleImputer(strategy="median"),
        #LinearRegression()
        RandomForestRegressor()
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

    prediction_for_test[]
    # Save the DataFrame to CSV
    prediction_for_test[["winner","winby"]].to_csv("pred_for_test.csv", index=None)
"""