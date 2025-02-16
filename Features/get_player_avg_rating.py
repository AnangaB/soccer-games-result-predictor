import pandas as pd


""" Function to add the average number of overall_rating stats each team has. 
The overall_rating per player is calculated by taking an average of their overall_rating values, in all their playing years.

df is the input dataset, where features will be added to
players_df contains data on players and their overall_rating, for different years they have played
"""
def add_team_avg_rating(df, players_df):
    
    # Create a dictionary of player IDs mapped to their average ratings
    player_avg_overall_rating_dict = get_player_avg_rating(players_df)


    # Define player columns for home and away teams
    home_players_cols = ["home_player_" + str(i) for i in range(1, 12)]
    away_players_cols = ["away_player_" + str(i) for i in range(1, 12)]
    player_cols = home_players_cols + away_players_cols

    # Initialize a DataFrame to hold average ratings
    player_avg_ratings = pd.DataFrame(index=df.index)  # Ensure it matches df's index
    
    for col in player_cols:
        player_id_col = df[col]

        # Ensure player IDs are of the same type as the keys in the dictionary
        player_avg_ratings[col + "_avg_rating"] = player_id_col.map(
            lambda v: player_avg_overall_rating_dict.get(int(v), pd.NA) if pd.notna(v) and v is not None else pd.NA
        )  

    # Concatenate the player average ratings to the main DataFrame
    df = pd.concat([df, player_avg_ratings], axis=1)

    # Calculate the average ratings for home and away players
    home_rating_cols = [col + "_avg_rating" for col in home_players_cols]
    away_rating_cols = [col + "_avg_rating" for col in away_players_cols]

    #df["avg_team_rating_difference"] =  df[home_rating_cols].mean(axis=1) - df[away_rating_cols].mean(axis=1)
    df["avg_home_players_rating"] = df[home_rating_cols].mean(axis=1)
    df["avg_away_players_rating"] = df[away_rating_cols].mean(axis=1)

    # Drop the extra columns from player_avg_ratings
    df.drop(columns=player_avg_ratings.columns, inplace=True)

    return df

""" Returns a dict mapping player id to their average of number of overall_rating values, in all their playing years.
"""

def get_player_avg_rating(players_df):
    players_copy = players_df.copy()

    # Group by player_id and calculate the mean for overall ratings
    avg_player_rating_df = players_copy[["player_id", "overall_rating"]].groupby("player_id").agg({
        "overall_rating": "mean",
    })

    # Convert the DataFrame to a dictionary mapping player_id to average rating
    player_avg_overall_rating_dict = avg_player_rating_df["overall_rating"].to_dict()
    return player_avg_overall_rating_dict