import pandas as pd


""" Function to add the average proportion of left footed players each team has. 

df is the input dataset, where features will be added to
players_df contains data on players and their left_foot data, for different years they have played
"""
def get_team_left_foot_rate_rows(df, players_df):
    home_players_cols = ["home_player_" + str(i) for i in range(1, 12)]
    away_players_cols = ["away_player_" + str(i) for i in range(1, 12)]
    player_cols = home_players_cols + away_players_cols

    player_id_to_prefered_foot_map = get_player_prefered_foot_map(players_df)

    player_prefered_foot = pd.DataFrame(index=df.index)

    for col in player_cols:
        player_id_col = df[col]
        
        # Map work rates based on player IDs for both attacking and defensive work rates
        player_prefered_foot[col + "_foot"] = player_id_col.map(lambda id: player_id_to_prefered_foot_map.get(id,0))
    
    
    # Concatenate the player average ratings to the main DataFrame
    df = pd.concat([df, player_prefered_foot], axis=1)

    # Calculate the average ratings for home and away players
    home_preferred_foot_cols = [col + "_foot" for col in home_players_cols]
    away_preferred_foot_cols = [col + "_foot" for col in away_players_cols]

    df["home_team_left_foot_rate"] = df[home_preferred_foot_cols].sum(axis=1) / 11
    df["away_team_left_foot_rate"] = df[away_preferred_foot_cols].sum(axis=1) / 11

    # Drop the extra columns from player_avg_ratings
    df.drop(columns=player_prefered_foot.columns, inplace=True)


    return df
    

""" Return dict mappings player id to 1 if that player is left footed, or 0 otherwise
"""
def get_player_prefered_foot_map(players_df):
    players_copy = players_df.copy()

    prefered_foot_values_map = {"left": 1, "right": 0}

    players_copy["preferred_foot"] =  players_copy["preferred_foot"].map(lambda f: prefered_foot_values_map.get(f,0))
    player_prefered_foot_df = players_copy[["player_id", "preferred_foot"]].groupby("player_id").agg({
        "preferred_foot": "median",
    })

    player_id_to_prefered_foot_map = player_prefered_foot_df["preferred_foot"].to_dict()
    return player_id_to_prefered_foot_map
