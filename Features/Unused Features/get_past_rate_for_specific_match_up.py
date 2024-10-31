import pandas as pd


""" Function to add the win rate based on all the times the home team has played against that away team (fixing their postions at home and away respectively)

df is the input dataset, where features will be added to
games_df contains data on past matches
"""
def add_past_win_rate(df,games_df):
    
    win_rate_map = get_home_team_win_rate(games_df)
    # Map past win rates for home teams
    df['home_team_past_win_rate'] = df.apply(
        lambda row: win_rate_map.get((row["home_team_id"], row["away_team_id"]), 0), axis=1)*100
    return df


#returns a dictionary that gives for each pair of home and away teams, say (Team H , Team B), the win rate in the past against each other (when fixing Team H at home and Team B at away)
def get_home_team_win_rate(games):
   
    team_wins = {} # Create a dictionary to count how many wins the home_team has against the away_team in the past
    team_matches = {}  # To count total games between home and away teams

    for _, row in games.iterrows():
        home_team = row['home_team_id']
        away_team = row['away_team_id']
        winby = row['winby']
        winner = row['winner']
        if home_team and away_team:
            # Count the total games played between the home and away teams
            if (home_team, away_team) in team_matches:
                team_matches[(home_team, away_team)] += 1
            else:
                team_matches[(home_team, away_team)] = 1
            
            # Count wins for home team against away team
            if winby > 0 and winner == 1:  # Home team won
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
