import pandas as pd

def get_avg_player_work_rates(players_df):
    
    players_copy = players_df.copy()
    attack_work_rate_map = {"medium": 75, "low": 0, "high": 200}
    defense_work_rate_map = {"medium": 75, "low": 0, "high": 200}

    # Map work rates for both attacking and defensive work rates
    players_copy["attacking_work_rate"] = players_copy["attacking_work_rate"].map(lambda r: attack_work_rate_map.get(r, 50))
    players_copy["defensive_work_rate"] = players_copy["defensive_work_rate"].map(lambda r: defense_work_rate_map.get(r, 50)) 

    # Group by player_id and calculate the mean for work rates
    avg_work_rates_df = players_copy[["player_id", "attacking_work_rate", "defensive_work_rate"]].groupby("player_id").agg({
        "attacking_work_rate": "mean", 
        "defensive_work_rate": "mean"
    })

    return avg_work_rates_df

def add_player_work_rate_columns(df, players_df):
    home_players_cols = ["home_player_" + str(i) for i in range(1, 12)]
    away_players_cols = ["away_player_" + str(i) for i in range(1, 12)]
    player_cols = home_players_cols + away_players_cols

    avg_work_rates_df = get_avg_player_work_rates(players_df)

    player_ids = avg_work_rates_df.index.unique()

    player_work_rates = pd.DataFrame()

    for col in player_cols:
        player_id_col = df[col]
        
        def get_work_rate(id, is_attack):
            if id in player_ids:
                return avg_work_rates_df.loc[id, "attacking_work_rate"] if is_attack else avg_work_rates_df.loc[id, "defensive_work_rate"]
            return 50  # Impute missing values with 50
        
        player_work_rates[col + "_attack_work_rate"] = player_id_col.map(lambda id: get_work_rate(id, True))
        player_work_rates[col + "_defense_work_rate"] = player_id_col.map(lambda id: get_work_rate(id, False))

    # Concatenate the player work rates to the main DataFrame
    df = pd.concat([df, player_work_rates], axis=1)

    # Calculate the average work rate differences for attack and defense
    home_attack_work_rate_cols = [col + "_attack_work_rate" for col in home_players_cols]
    home_defense_work_rate_cols = [col + "_defense_work_rate" for col in home_players_cols]
    away_attack_work_rate_cols = [col + "_attack_work_rate" for col in away_players_cols]
    away_defense_work_rate_cols = [col + "_defense_work_rate" for col in away_players_cols]

    df["avg_attack_work_rate_difference"] = df[home_attack_work_rate_cols].mean(axis=1) - df[away_attack_work_rate_cols].mean(axis=1)
    df["avg_defense_work_rate_difference"] = df[home_defense_work_rate_cols].mean(axis=1) - df[away_defense_work_rate_cols].mean(axis=1)

    df["max_attack_work_rate_difference"] = df[home_attack_work_rate_cols].max(axis=1) - df[away_attack_work_rate_cols].max(axis=1)
    df["max_defense_work_rate_difference"] = df[home_defense_work_rate_cols].max(axis=1) - df[away_defense_work_rate_cols].max(axis=1)

    # Drop the extra columns from player_work_rates
    df.drop(columns=player_work_rates.columns, inplace=True)

    return df

