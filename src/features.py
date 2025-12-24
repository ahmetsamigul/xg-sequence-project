# src/features.py
import numpy as np
import pandas as pd

PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0
GOAL_X = 120.0
GOAL_Y = 40.0
GOAL_HALF_WIDTH = 3.66  # 7.32m / 2 (StatsBomb ölçeği metrik değil ama oran doğru)

def compute_distance_angle(df: pd.DataFrame) -> pd.DataFrame:
    """
    Neden?
    - xG'nin en güçlü 2 açıklayıcısı genellikle mesafe + açıdır.
    - Logistic regression baseline için ideal başlangıç.
    """
    dfx = df.copy()

    # distance to goal center
    dx = (GOAL_X - dfx["x"].astype(float))
    dy = (GOAL_Y - dfx["y"].astype(float))
    dfx["distance"] = np.sqrt(dx**2 + dy**2)

    # shot angle: goal posts to shooter angle
    # Sol ve sağ direk noktaları:
    left_post = np.stack([np.full(len(dfx), GOAL_X), np.full(len(dfx), GOAL_Y - GOAL_HALF_WIDTH)], axis=1)
    right_post = np.stack([np.full(len(dfx), GOAL_X), np.full(len(dfx), GOAL_Y + GOAL_HALF_WIDTH)], axis=1)
    shooter = np.stack([dfx["x"].astype(float).to_numpy(), dfx["y"].astype(float).to_numpy()], axis=1)

    a = left_post - shooter
    b = right_post - shooter

    # açı = arccos( (a·b) / (|a||b|) )
    dot = (a * b).sum(axis=1)
    norm = (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9)
    cosang = np.clip(dot / norm, -1.0, 1.0)
    dfx["angle"] = np.arccos(cosang)  # radyan

    return dfx

def add_simple_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Neden?
    - Sadece distance/angle değil, bazı durum bayrakları da gol olasılığını etkiler.
    - under_pressure, one_on_one, first_time gibi.
    """
    dfx = df.copy()
    for col in ["under_pressure", "shot_first_time", "shot_one_on_one", "shot_aerial_won"]:
        if col in dfx.columns:
            dfx[col] = dfx[col].fillna(False).astype(int)
        else:
            dfx[col] = 0
    return dfx

def one_hot_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Neden?
    - shot_body_part / shot_type gibi kategorikler model için sayısala çevrilmeli.
    - Baseline için one-hot yeterli.
    """
    dfx = df.copy()
    cat_cols = []
    for c in ["shot_body_part", "shot_type"]:
        if c in dfx.columns:
            cat_cols.append(c)

    if cat_cols:
        dfx = pd.get_dummies(dfx, columns=cat_cols, dummy_na=True)

    return dfx

def build_tabular_features(csv_path="data/shots.csv") -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Temizlik: x,y yoksa at
    df = df.dropna(subset=["x", "y", "is_goal"]).copy()

    df = compute_distance_angle(df)
    df = add_simple_flags(df)
    df = one_hot_categoricals(df)

    return df
