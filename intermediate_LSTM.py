import os
import time
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, confusion_matrix, classification_report
from torch.optim.lr_scheduler import ReduceLROnPlateau

from nba_api.stats.endpoints import leaguegamelog

"""
##############################################################################
CONFIGURATION / HYPERPARAMETERS
This block sets up configuration paths, data splits, seeds, etc.
We further split into LSTM hyperparameters and MLP hyperparameters.
We also include details about caching/pickling to save/reload intermediate data.
##############################################################################
"""

DATA_CSV_PATH = "nba_15seasons_data.csv"

# >>>> ADJUST WINDOW_SIZE HERE <<<<
WINDOW_SIZE = 8
NUM_LAGS = WINDOW_SIZE - 1

AGG_DATA_PICKLE_TEMPLATE = "aggregated_dataset_w{}.pkl"
AGG_DATA_PICKLE = AGG_DATA_PICKLE_TEMPLATE.format(WINDOW_SIZE)

SEASONS_LIST = [
    "2008-09", "2009-10", "2010-11", "2011-12", "2012-13",
    "2013-14", "2014-15", "2015-16", "2016-17", "2017-18",
    "2018-19", "2019-20", "2020-21", "2021-22", "2022-23"
]

BASE_BOX_COLS = [
    "MIN", "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT",
    "FTM", "FTA", "FT_PCT", "OREB", "DREB", "REB",
    "AST", "STL", "BLK", "TOV", "PF", "PTS"
]

AGG_METHODS = ["sum","mean","min","max","median","std"]

TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1

BATCH_SIZE = 64
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 8e-2
EPOCHS = 100   # Increased to 200

"""
LSTM Hyperparameters
"""
LSTM_LAYERS = 3
LSTM_HIDDEN_DIM = 130
LSTM_DROPOUT = 0.50

"""
MLP Hyperparameters
"""
FFN_DROPOUT = 0.70
MLP_HIDDEN_UNITS = [130, 64, 32, 64, 16]
print(MLP_HIDDEN_UNITS)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

"""
##############################################################################
1) Load / Fetch
Handles loading data from CSV if exists, otherwise fetches from NBA Stats API.
Caching/pickling is used to avoid repeated API calls.
##############################################################################
"""
def load_or_fetch_nba_data(csv_path=DATA_CSV_PATH, seasons_list=SEASONS_LIST, season_type="Regular Season"):
    if os.path.exists(csv_path):
        print(f"Loading data from {csv_path} ...")
        df = pd.read_csv(csv_path)
        if 'GAME_DATE' in df.columns:
            df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
        df.sort_values(by=['PLAYER_ID','GAME_DATE'], inplace=True, ignore_index=True)
        return df

    print("CSV not found. Fetching data from NBA Stats API ...")
    all_dfs = []
    for season in seasons_list:
        print(f"Fetching data for season: {season} ...")
        logs = leaguegamelog.LeagueGameLog(
            season=season,
            season_type_all_star=season_type,
            player_or_team_abbreviation='P'
        )
        df_season = logs.get_data_frames()[0]
        all_dfs.append(df_season)

    df_all = pd.concat(all_dfs, ignore_index=True)
    print(f"Total records pulled: {len(df_all)}")

    base_cols = ["GAME_ID","PLAYER_ID","TEAM_ABBREVIATION","MATCHUP","WL","GAME_DATE"] + BASE_BOX_COLS
    actual_cols = [c for c in base_cols if c in df_all.columns]
    df_all = df_all[actual_cols].copy()

    if 'GAME_DATE' in df_all.columns:
        df_all['GAME_DATE'] = pd.to_datetime(df_all['GAME_DATE'])

    df_all.sort_values(by=['PLAYER_ID','GAME_DATE'], inplace=True, ignore_index=True)
    df_all.to_csv(csv_path, index=False)
    print(f"Data saved to {csv_path}")
    return df_all

"""
##############################################################################
Determine game-level scoreboard (TEAM_PTS for each row)
We figure out home team vs away team points, but we want each row to know
how many points that row's team scored. Used later for ratio features.
##############################################################################
"""
def attach_team_points(df):
    """
    Return a copy of df with a new column 'TEAM_PTS' that shows how many points
    that row's team scored in that same game. This will let us create ratio features
    at the player level (e.g., player's PTS / TEAM_PTS).
    """
    tmp = df.copy()
    team_pts_df = tmp.groupby(['GAME_ID','TEAM_ABBREVIATION'])['PTS'].sum().reset_index()
    team_pts_df.rename(columns={'PTS':'TEAM_PTS'}, inplace=True)

    merged = tmp.merge(team_pts_df, on=['GAME_ID','TEAM_ABBREVIATION'], how='left')
    return merged

"""
##############################################################################
2) Sliding Window
Creates lag-based features for each player's recent games, including deltas/ratios.
##############################################################################
"""
def create_player_windows(df, window_size=8):
    print(f"Creating sliding windows (window_size={window_size})...")
    df = df.copy()
    df['WL'] = df['WL'].fillna('L')
    df['WL_NUM'] = df['WL'].map(lambda x: 1 if x == 'W' else 0)

    # Convert MIN if needed
    if 'MIN' in df.columns and df['MIN'].dtype == object:
        def convert_min_to_float(m):
            try:
                mm, ss = m.split(':')
                return float(mm) + float(ss)/60.0
            except:
                return 0.0
        df['MIN'] = df['MIN'].apply(convert_min_to_float).astype(float)

    # We'll attach a new column 'TEAM_PTS' so we can do ratio features
    df = attach_team_points(df)  # now df has col 'TEAM_PTS'

    numeric_cols = []
    for c in BASE_BOX_COLS + ["WL_NUM","TEAM_PTS"]:
        if c in df.columns:
            numeric_cols.append(c)

    all_rows = []
    grouped = df.groupby('PLAYER_ID', group_keys=False)
    total_players = grouped.ngroups
    player_count = 0

    for pid, group in grouped:
        player_count += 1
        if player_count % 100 == 0:
            print(f"  Processed {player_count} / {total_players} players ...")

        group = group.sort_values('GAME_DATE', ascending=True)
        n_games = len(group)
        if n_games < window_size:
            continue

        for start_idx in range(0, n_games - window_size + 1):
            window_df = group.iloc[start_idx : start_idx + window_size]
            current_game = window_df.iloc[-1]
            current_game_id = current_game['GAME_ID']

            row_dict = {
                "GAME_ID": current_game_id,
                "PLAYER_ID": pid
            }

            lag_df = window_df.iloc[0 : window_size-1]
            for col in numeric_cols:
                # store each lag
                for i in range(window_size-1):
                    val = lag_df.iloc[i][col]
                    row_dict[f"{col}_lag{i+1}"] = val

                # We'll do 'delta' & 'ratio' expansions
                for i in range(1, window_size-1):
                    prev_val = lag_df.iloc[i-1][col]
                    curr_val = lag_df.iloc[i][col]
                    row_dict[f"{col}_lag{i+1}_delta"] = curr_val - prev_val

                    denom = prev_val + 1.0
                    ratio_val = curr_val / denom
                    ratio_val = np.clip(ratio_val, -10.0, 10.0)
                    row_dict[f"{col}_lag{i+1}_ratio"] = ratio_val

            all_rows.append(row_dict)

    feat_df = pd.DataFrame(all_rows)
    feat_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    feat_df.dropna(inplace=True)
    feat_df.reset_index(drop=True, inplace=True)
    print(f"  Finished creating sliding windows: feat_df.shape={feat_df.shape}")
    return feat_df

"""
##############################################################################
Determine final scoreboard => point differential
We define a label for regression (POINT_DIFF).
##############################################################################
"""
def determine_scoreboard(df):
    tmp = df.copy()

    def home_team_abbr(matchup):
        if 'vs.' in matchup:
            return matchup.split(' vs. ')[0]
        elif '@' in matchup:
            return matchup.split(' @ ')[1]
        return None

    tmp['HOME_TEAM_ABBR'] = tmp['MATCHUP'].apply(home_team_abbr)
    tmp['IS_HOME_PLAYER'] = (tmp['TEAM_ABBREVIATION'] == tmp['HOME_TEAM_ABBR']).astype(int)

    scoreboard = tmp.groupby(['GAME_ID','IS_HOME_PLAYER'])['PTS'].sum().unstack(fill_value=0).reset_index()
    scoreboard.rename(columns={0:'AWAY_PTS',1:'HOME_PTS'}, inplace=True)

    scoreboard['POINT_DIFF'] = scoreboard['HOME_PTS'] - scoreboard['AWAY_PTS']
    return scoreboard[['GAME_ID','POINT_DIFF']]

"""
##############################################################################
Aggregate player features to game-level
We group by GAME_ID, apply aggregators (sum, mean, etc.), and attach POINT_DIFF.
##############################################################################
"""
def aggregate_player_features_to_game(feat_df, original_df):
    print("Aggregating per-game features ...")
    exclude_cols = ['GAME_ID','PLAYER_ID']
    feature_cols = [c for c in feat_df.columns if c not in exclude_cols]

    agg_dict = {}
    for c in feature_cols:
        agg_dict[c] = AGG_METHODS

    grouped = feat_df.groupby('GAME_ID').agg(agg_dict)
    grouped.columns = [f"{col}_{agg}" for col,agg in grouped.columns]
    grouped.reset_index(drop=False, inplace=True)

    scoreboard_df = determine_scoreboard(original_df)
    merged = grouped.merge(scoreboard_df, on='GAME_ID', how='inner')
    print(f"  Aggregation done. merged.shape={merged.shape}")
    return merged

"""
##############################################################################
Torch Dataset for Regression
Packs our features (X) and labels (y) into a PyTorch-friendly structure.
##############################################################################
"""
class GameDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y  # continuous => point diff

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.float32)
        )

"""
##############################################################################
Model => Output 1 dimension
We define an LSTM + MLP architecture for point-differential regression.
##############################################################################
"""
class BigLSTMModel(nn.Module):
    def __init__(
        self,
        input_dim,
        lstm_hidden_dim=256,
        lstm_layers=2,
        mlp_hidden_units=[256, 128, 64, 32, 16, 8],
        lstm_dropout=0.3,
        ffn_dropout=0.3
    ):
        super().__init__()
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_layers = lstm_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=lstm_dropout
        )

        layers = []
        prev_dim = lstm_hidden_dim
        for h in mlp_hidden_units:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.Mish())
            layers.append(nn.Dropout(ffn_dropout))
            prev_dim = h

        layers.append(nn.Linear(prev_dim, 1))  # final => 1 dimension
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        # x => (batch, seq=1, input_dim)
        batch_size = x.size(0)
        device = x.device
        h0 = torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden_dim, device=device)
        c0 = torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden_dim, device=device)

        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]  # shape => (batch, hidden_dim)
        out = self.mlp(out)  # => (batch,1)
        return out

"""
##############################################################################
TRAIN with MSE, sign-based accuracy
Custom training loop and evaluation with an additional sign-based metric.
##############################################################################
"""
def compute_sign_accuracy(preds, trues):
    pred_sign = (preds > 0).astype(int)
    true_sign = (trues > 0).astype(int)
    return (pred_sign == true_sign).mean()

def evaluate_regression(model, loader, criterion, device='cpu'):
    model.eval()
    running_loss = 0.0
    preds_all = []
    trues_all = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.unsqueeze(1).to(device)
            y_batch = y_batch.to(device).unsqueeze(-1)
            out = model(X_batch)
            loss = criterion(out, y_batch)
            running_loss += loss.item() * X_batch.size(0)
            preds_all.extend(out.squeeze(-1).cpu().numpy())
            trues_all.extend(y_batch.squeeze(-1).cpu().numpy())

    mse = running_loss / len(loader.dataset)
    sign_acc = compute_sign_accuracy(np.array(preds_all), np.array(trues_all))
    return mse, sign_acc

def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler=None,
    epochs=10,
    device='cpu'
):
    model.to(device)
    train_losses = []
    val_losses = []
    train_sign_accs = []
    val_sign_accs = []

    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.unsqueeze(1).to(device)
            y_batch = y_batch.to(device).unsqueeze(-1)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # measure train & val MSE and sign-acc
        train_mse, train_sacc = evaluate_regression(model, train_loader, criterion, device=device)
        val_mse, val_sacc = evaluate_regression(model, val_loader, criterion, device=device)

        train_losses.append(train_mse)
        val_losses.append(val_mse)
        train_sign_accs.append(train_sacc)
        val_sign_accs.append(val_sacc)

        if scheduler is not None:
            scheduler.step(val_mse)

        print(f"Epoch [{epoch+1}/{epochs}] "
              f"- Train MSE: {train_mse:.4f}, Val MSE: {val_mse:.4f}, "
              f"Train SignAcc: {train_sacc:.4f}, Val SignAcc: {val_sacc:.4f}")

    return train_losses, val_losses, train_sign_accs, val_sign_accs

"""
##############################################################################
MAIN
Coordinates the entire process: data loading, feature creation, training/validation,
and final evaluation with metrics + plots.
##############################################################################
"""
def main():
    df_pickle = AGG_DATA_PICKLE
    if os.path.exists(df_pickle):
        print(f"Loading aggregated data from {df_pickle}...")
        with open(df_pickle,'rb') as f:
            data_dict = pickle.load(f)
        game_df = data_dict['game_df']
        y_all = data_dict['labels']  # => point diff
    else:
        df_raw = load_or_fetch_nba_data()
        # sliding window with ratio expansions
        feat_df = create_player_windows(df_raw, WINDOW_SIZE)
        # aggregator
        print("Aggregating to game-level with point diff ...")
        merged = aggregate_player_features_to_game(feat_df, df_raw)
        y_all = merged['POINT_DIFF'].values.astype(float)
        merged.drop(columns=['GAME_ID','POINT_DIFF'], inplace=True)

        print("Pickling final aggregated data ...")
        with open(df_pickle,'wb') as f:
            pickle.dump({"game_df": merged, "labels": y_all}, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved aggregated data => {df_pickle}")
        game_df = merged

    # Clean up
    game_df.replace([np.inf,-np.inf], np.nan, inplace=True)
    game_df.fillna(0, inplace=True)
    X_all = game_df.values.astype(np.float32)

    # shuffle
    df_shuf = pd.DataFrame(X_all)
    df_shuf['diff'] = y_all
    df_shuf = df_shuf.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    y_all = df_shuf['diff'].values
    X_all = df_shuf.drop(columns=['diff']).values

    N = len(X_all)
    train_end = int(TRAIN_SPLIT*N)
    val_end = int((TRAIN_SPLIT+VAL_SPLIT)*N)

    X_train, y_train = X_all[:train_end], y_all[:train_end]
    X_val, y_val = X_all[train_end:val_end], y_all[train_end:val_end]
    X_test, y_test = X_all[val_end:], y_all[val_end:]

    print(f"Data => Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    train_ds = GameDataset(X_train_s, y_train)
    val_ds = GameDataset(X_val_s, y_val)
    test_ds = GameDataset(X_test_s, y_test)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    input_dim = X_train_s.shape[1]
    print(f"Input dimension => {input_dim}")

    model = BigLSTMModel(
        input_dim=input_dim,
        lstm_hidden_dim=LSTM_HIDDEN_DIM,
        lstm_layers=LSTM_LAYERS,
        mlp_hidden_units=MLP_HIDDEN_UNITS,
        lstm_dropout=LSTM_DROPOUT,
        ffn_dropout=FFN_DROPOUT
    )

    criterion = nn.MSELoss()  # regression

    # AdamW optimizer with decoupled weight decay
    decay_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad
           and not n.endswith(".bias")
           and "bn" not in n.lower()
           and "ln" not in n.lower()
    ]
    no_decay_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad
           and (n.endswith(".bias") or "bn" in n.lower() or "ln" in n.lower())
    ]

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params,   "weight_decay": WEIGHT_DECAY},
            {"params": no_decay_params,"weight_decay": 0.0}
        ],
        lr=LEARNING_RATE
    )
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on => {device}")

    # Train
    train_mses, val_mses, train_sign_accs, val_sign_accs = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler=scheduler,
        epochs=EPOCHS,
        device=device
    )

    # Evaluate
    model.eval()
    from math import sqrt
    test_preds = []
    test_trues = []
    with torch.no_grad():
        for Xb, Yb in test_loader:
            Xb = Xb.unsqueeze(1).to(device)
            Yb = Yb.to(device)
            out = model(Xb).squeeze(-1)
            test_preds.extend(out.cpu().numpy())
            test_trues.extend(Yb.cpu().numpy())

    test_preds = np.array(test_preds)
    test_trues = np.array(test_trues)
    test_mse = mean_squared_error(test_trues, test_preds)
    test_rmse = sqrt(test_mse)
    print(f"\nTest RMSE => {test_rmse:.4f}")

    # sign-based classification
    test_pred_sign = (test_preds > 0).astype(int)
    test_true_sign = (test_trues > 0).astype(int)
    test_sign_acc = (test_pred_sign == test_true_sign).mean()
    print(f"Sign-based Test Accuracy => {test_sign_acc:.4f}")

    # confusion matrix & classification report
    cm = confusion_matrix(test_true_sign, test_pred_sign)
    clr = classification_report(test_true_sign, test_pred_sign, digits=4)
    print("Confusion Matrix (0=loss,1=win):")
    print(cm)
    print("Classification Report:")
    print(clr)

    # PLOTS
    # 1) MSE
    plt.figure(figsize=(8,6))
    plt.plot(range(1,EPOCHS+1), train_mses, label='Train MSE')
    plt.plot(range(1,EPOCHS+1), val_mses, label='Val MSE')
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title(f"Train/Val MSE Over Epochs (Point Diff Regression) - {EPOCHS} epochs")
    plt.legend()
    plt.savefig("training_mse_plot.png")
    print("Saved training MSE plot to training_mse_plot.png")

    # 2) Sign Accuracy
    plt.figure(figsize=(8,6))
    plt.plot(range(1,EPOCHS+1), np.array(train_sign_accs)*100, label='Train SignAcc (%)')
    plt.plot(range(1,EPOCHS+1), np.array(val_sign_accs)*100, label='Val SignAcc (%)')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(f"Train/Val Sign Accuracy Over Epochs - {EPOCHS} epochs")
    plt.legend()
    plt.savefig("training_signacc_plot.png")
    print("Saved training sign accuracy plot to training_signacc_plot.png")

    print("\nAll done!")

if __name__ == "__main__":
    main()
