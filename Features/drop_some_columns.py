
from sklearn.pipeline import FunctionTransformer
import pandas as pd

"""
    Drops columns in the input df, to only keep relevant features, and returns it
"""
def condense_columns(df):
    home_players_cols = ["home_player_" + str(i) for i in range(1, 12)]
    away_players_cols = ["away_player_" + str(i) for i in range(1, 12)]
    home_players_X_cols = ["home_player_X" + str(i) for i in range(1, 12)]
    away_players_X_cols = ["away_player_X" + str(i) for i in range(1, 12)]
    home_players_Y_cols = ["home_player_Y" + str(i) for i in range(1, 12)]
    away_players_Y_cols = ["away_player_Y" + str(i) for i in range(1, 12)]
    player_location_cols =home_players_cols + away_players_cols +  home_players_X_cols + away_players_X_cols +  home_players_Y_cols + away_players_Y_cols 
    df = df.drop(columns=player_location_cols +["home_team_id","away_team_id"] )
    #df.to_csv("training_data.csv")
    return df