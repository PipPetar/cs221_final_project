import pandas as pd
import numpy as np
from nba_api.stats.endpoints import leaguegamelog
from sklearn.metrics import accuracy_score, mean_squared_error, log_loss, classification_report, confusion_matrix

SEASONS = ["2022-23"]
SEASON_TYPE = "Regular Season"


def fetch_nba_data(seasons, season_type="Regular Season"):
    frames = []
    for season in seasons:
        print(f"Fetching data for season {season}...")
        logs = leaguegamelog.LeagueGameLog(
            season=season,
            season_type_all_star=season_type,
            player_or_team_abbreviation='T'
        )
        df = logs.get_data_frames()[0]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


df = fetch_nba_data(SEASONS, SEASON_TYPE)

df = df[['GAME_ID', 'GAME_DATE', 'TEAM_ABBREVIATION', 'MATCHUP', 'WL', 'PTS']]
df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])


# Parse matchup to find home and away teams
def parse_matchup(row):
    teams = row['MATCHUP'].split(' ')
    if 'vs.' in teams:
        home_team = row['TEAM_ABBREVIATION']
        away_team = teams[-1]
    else:  # '@' indicates away team
        home_team = teams[-1]
        away_team = row['TEAM_ABBREVIATION']
    return pd.Series([home_team, away_team])


df[['HOME_TEAM', 'AWAY_TEAM']] = df.apply(parse_matchup, axis=1)

# Pivot table to get home and away points
game_df = df.pivot_table(index=['GAME_ID', 'GAME_DATE', 'HOME_TEAM', 'AWAY_TEAM'],
                         columns='TEAM_ABBREVIATION', values='PTS', aggfunc='first').reset_index()

# Corrected replacement of deprecated 'lookup' method
game_df['HOME_PTS'] = game_df.apply(lambda row: row[row['HOME_TEAM']], axis=1)
game_df['AWAY_PTS'] = game_df.apply(lambda row: row[row['AWAY_TEAM']], axis=1)

# Calculate game outcomes
game_df['HOME_WIN'] = (game_df['HOME_PTS'] > game_df['AWAY_PTS']).astype(int)

# Baseline 1: Always predict home win
game_df['baseline_home'] = 1

# Baseline 2: 5-game momentum rule
team_histories = {}
momentum_preds = []

# Ensure data is sorted chronologically
game_df.sort_values(by='GAME_DATE', inplace=True)

for _, row in game_df.iterrows():
    home, away = row['HOME_TEAM'], row['AWAY_TEAM']

    for team in [home, away]:
        if team not in team_histories:
            team_histories[team] = []

    home_recent_wins = sum(team_histories[home][-5:])
    away_recent_wins = sum(team_histories[away][-5:])

    # Momentum prediction logic
    if home_recent_wins > away_recent_wins:
        pred = 1
    elif home_recent_wins < away_recent_wins:
        pred = 0
    else:
        pred = 1  # default to home if tied

    momentum_preds.append(pred)

    # Update team win-loss records
    home_win = row['HOME_WIN']
    team_histories[home].append(home_win)
    team_histories[away].append(1 - home_win)

game_df['baseline_momentum'] = momentum_preds


def evaluate_baseline(name, actual, predictions):
    accuracy = accuracy_score(actual, predictions)
    mse = mean_squared_error(actual, predictions)
    logloss = log_loss(actual, predictions)
    cm = confusion_matrix(actual, predictions)
    report = classification_report(actual, predictions, target_names=['Away Win (0)', 'Home Win (1)'])

    print(f"\n--- Baseline: {name} ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"Log-loss: {logloss:.4f}")
    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:\n", report)


# Evaluate baselines
evaluate_baseline("Home-team always wins", game_df['HOME_WIN'], game_df['baseline_home'])
evaluate_baseline("5-game momentum", game_df['HOME_WIN'], game_df['baseline_momentum'])
