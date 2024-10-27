
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
    