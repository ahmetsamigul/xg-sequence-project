# src/train_tabular.py
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

from features import build_tabular_features

def brier_score(y_true, y_prob):
    return np.mean((y_prob - y_true) ** 2)

def main():
    df = build_tabular_features("data/shots.csv")

    # Target
    y = df["is_goal"].astype(int).to_numpy()

    # Feature kolonlarını seç (is_goal, id gibi alanları çıkar)
    drop_cols = {"is_goal", "id", "player", "team", "minute", "second"}
    X = df[[c for c in df.columns if c not in drop_cols]].copy()

    # match bazlı split (neden: leakage engeli)
    match_ids = df["match_id"].astype(int).to_numpy()
    unique_matches = np.unique(match_ids)

    train_m, test_m = train_test_split(unique_matches, test_size=0.2, random_state=42)
    train_mask = np.isin(match_ids, train_m)
    test_mask = np.isin(match_ids, test_m)

    X_train, y_train = X.loc[train_mask], y[train_mask]
    X_test, y_test = X.loc[test_mask], y[test_mask]

    # Logistic Regression baseline
    model = LogisticRegression(max_iter=2000, n_jobs=-1)
    model.fit(X_train, y_train)

    p_train = model.predict_proba(X_train)[:, 1]
    p_test = model.predict_proba(X_test)[:, 1]

    # Metrics
    metrics = {
        "train_logloss": log_loss(y_train, p_train),
        "test_logloss": log_loss(y_test, p_test),
        "train_auc": roc_auc_score(y_train, p_train),
        "test_auc": roc_auc_score(y_test, p_test),
        "train_brier": brier_score(y_train, p_train),
        "test_brier": brier_score(y_test, p_test),
    }
    print("=== BASELINE RESULTS (LogReg) ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # Calibration curve (sunumda çok iyi durur)
    prob_true, prob_pred = calibration_curve(y_test, p_test, n_bins=10, strategy="uniform")

    plt.figure()
    plt.plot(prob_pred, prob_true, marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("Predicted probability")
    plt.ylabel("True fraction of positives")
    plt.title("Calibration Curve (Test)")
    plt.tight_layout()
    plt.savefig("results/calibration_curve.png", dpi=200)
    print("Saved: results/calibration_curve.png")

if __name__ == "__main__":
    main()
