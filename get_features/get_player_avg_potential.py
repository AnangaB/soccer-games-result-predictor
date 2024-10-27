import pandas as pd


def get_player_avg_potential_map(players_df):
    players_copy = players_df.copy()

    # Group by player_id and calculate the mean for overall ratings
    avg_player_rating_df = players_copy[["player_id", "potential"]].groupby("player_id").agg({
        "potential": "mean",
    })

    # Convert the DataFrame to a dictionary mapping player_id to average rating
    player_avg_potential_map = avg_player_rating_df["potential"].to_dict()
    return player_avg_potential_map



def get_player_potential_rows(df, players_df):
    home_players_cols = ["home_player_" + str(i) for i in range(1, 12)]
    away_players_cols = ["away_player_" + str(i) for i in range(1, 12)]
    player_cols = home_players_cols + away_players_cols

    player_id_to_potential_map = get_player_avg_potential_map(players_df)
    player_avg_potentials = pd.DataFrame(index=df.index)

    for col in player_cols:
        player_id_col = df[col]
        
        # Map work rates based on player IDs for both attacking and defensive work rates
        player_avg_potentials[col + "_potential"] = player_id_col.map(lambda id: player_id_to_potential_map.get(id,50))
    
    
    # Concatenate the player average ratings to the main DataFrame
    df = pd.concat([df, player_avg_potentials], axis=1)

    # Calculate the average ratings for home and away players
    home_potential_cols = [col + "_potential" for col in home_players_cols]
    away_potential_cols = [col + "_potential" for col in away_players_cols]

    df["avg_team_potential_difference"] = df[home_potential_cols].mean(axis=1) -  df[away_potential_cols].mean(axis=1)

    # Drop the extra columns from player_avg_ratings
    df.drop(columns=player_avg_potentials.columns, inplace=True)


    return df
    
