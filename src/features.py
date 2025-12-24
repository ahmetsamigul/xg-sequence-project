# src/features.py
import numpy as np
import pandas as pd

PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0
GOAL_X = 120.0
GOAL_Y = 40.0
GOAL_HALF_WIDTH = 3.66  # 7.32 / 2  (ölçek metrik değil ama oran doğru)

def maybe_standardize_coords(df: pd.DataFrame, enable: bool = False) -> pd.DataFrame:
    """
    Bazı datasetlerde şut koordinatları iki yöne dağılmış olabilir.
    Eğer çoğunluk x<60 ise, hedef kaleyi x=120 olacak şekilde flip eder:
      x' = 120 - x
      y' =  80 - y

    enable=False iken dokunmaz (varsayılan güvenli).
    """
    if not enable:
        return df

    dfx = df.copy()
    x = dfx["x"].astype(float)
    frac_left = (x < 60).mean()

    if frac_left > 0.55:
        dfx["x"] = 120.0 - dfx["x"].astype(float)
        dfx["y"] = 80.0 - dfx["y"].astype(float)
        print(f"[coords] flipped (frac_x<60={frac_left:.2f})")
    else:
        print(f"[coords] no flip (frac_x<60={frac_left:.2f})")

    return dfx

def compute_distance_angle(df: pd.DataFrame) -> pd.DataFrame:
    """
    xG'nin en güçlü açıklayıcıları: mesafe + açı.
    """
    dfx = df.copy()

    dx = (GOAL_X - dfx["x"].astype(float))
    dy = (GOAL_Y - dfx["y"].astype(float))
    dfx["distance"] = np.sqrt(dx**2 + dy**2)

    # Direk noktaları
    left_post = np.stack(
        [np.full(len(dfx), GOAL_X), np.full(len(dfx), GOAL_Y - GOAL_HALF_WIDTH)],
        axis=1
    )
    right_post = np.stack(
        [np.full(len(dfx), GOAL_X), np.full(len(dfx), GOAL_Y + GOAL_HALF_WIDTH)],
        axis=1
    )
    shooter = np.stack(
        [dfx["x"].astype(float).to_numpy(), dfx["y"].astype(float).to_numpy()],
        axis=1
    )

    a = left_post - shooter
    b = right_post - shooter

    dot = (a * b).sum(axis=1)
    norm = (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9)
    cosang = np.clip(dot / norm, -1.0, 1.0)
    dfx["angle"] = np.arccos(cosang)  # radyan

    return dfx

def add_simple_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basit durum bayrakları.
    """
    dfx = df.copy()
    for col in ["under_pressure", "shot_first_time", "shot_one_on_one", "shot_aerial_won"]:
        if col in dfx.columns:
            dfx[col] = dfx[col].fillna(False).map(bool).astype(int)
        else:
            dfx[col] = 0
    return dfx

def one_hot_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tabular baseline için kategorikleri one-hot yap.
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
    df = df.dropna(subset=["x", "y", "is_goal"]).copy()

    # Koordinat standardizasyonu gerekiyorsa enable=True yap
    df = maybe_standardize_coords(df, enable=False)

    df = compute_distance_angle(df)
    df = add_simple_flags(df)
    df = one_hot_categoricals(df)
    return df
