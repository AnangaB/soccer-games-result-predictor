import pandas as pd

""" Function to add the average number of crossings each team has. 
The number of crossings per player is calculated by taking an average of their number of crossings, in all their playing years.

df is the input dataset, where features will be added to
players_df contains data on players and their crossings, for different years they have played
"""
def get_team_avg_crossings(df, players_df):
    home_players_cols = ["home_player_" + str(i) for i in range(1, 12)]
    away_players_cols = ["away_player_" + str(i) for i in range(1, 12)]
    player_cols = home_players_cols + away_players_cols

    player_id_to_crossings_map = get_player_avg_crossing_map(players_df)
    player_total_crossings = pd.DataFrame(index=df.index)

    for col in player_cols:
        player_id_col = df[col]
        
        player_total_crossings[col + "_crossing"] = player_id_col.map(lambda id: player_id_to_crossings_map.get(id,0))
    
    
    df = pd.concat([df, player_total_crossings], axis=1)

    home_crossing_cols = [col + "_crossing" for col in home_players_cols]
    away_crossing_cols = [col + "_crossing" for col in away_players_cols]

    df["avg_home_crossing"] = df[home_crossing_cols].mean(axis=1)
    df["avg_away_crossing"] = df[away_crossing_cols].mean(axis=1)

    # Drop the extra columns from player_avg_ratings
    df.drop(columns=player_total_crossings.columns, inplace=True)


    return df
    
""" Returns a dict mapping player id to their average of number of crossings, in all their playing years.
"""
def get_player_avg_crossing_map(players_df):
    players_copy = players_df.copy()

    avg_player_crossing_df = players_copy[["player_id", "crossing"]].groupby("player_id").agg({
        "crossing": "mean",
    })

    avg_player_crossing_map = avg_player_crossing_df["crossing"].to_dict()
    return avg_player_crossing_map