import pandas as pd

""" Add the proportion of times each of the team has won in all of their past matches
    
    - df is the input dataset, where features will be added to
    - games_df contains data on past matches
"""
def add_team_total_wins(df,games_df):
    
    total_team_wins_count_map = get_team_win_rate(games_df)
    # Map past win rates for home teams
    df['home_team_total_wins'] = df["home_team_id"].map(total_team_wins_count_map)
    df['away_team_total_wins'] = df["away_team_id"].map(total_team_wins_count_map)

    return df

""" Return a dict mapping team ids to their win rate in all the past games they have played in
"""
def get_team_win_rate(games):
    # Dictionaries to track total wins and total games for each team
    team_wins = {}
    team_games = {}

    for _, row in games.iterrows():
        home_team = row['home_team_id']
        away_team = row['away_team_id']
        winby = row['winby']
        winner = row["winner"]

        # Count each game played for both teams
        team_games[home_team] = team_games.get(home_team, 0) + 1
        team_games[away_team] = team_games.get(away_team, 0) + 1

        # Increment win count based on winby
        if winby > 0 and winner == 1:  # Home team won
            team_wins[home_team] = team_wins.get(home_team, 0) + 1
        elif winby < 0 and winner == 0:  # Away team won
            team_wins[away_team] = team_wins.get(away_team, 0) + 1

    # Calculate win rate for each team
    team_win_rate = {}
    for team, wins in team_wins.items():
        total_games = team_games.get(team, 0)
        if total_games > 0:
            team_win_rate[team] = wins / total_games
        else:
            team_win_rate[team] = 0  # Avoid division by zero

    return team_win_rate
