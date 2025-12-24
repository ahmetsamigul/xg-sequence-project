# src/train_tabular.py
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

from features import build_tabular_features

def brier_score(y_true, y_prob):
    y_true = np.asarray(y_true, dtype=np.float32)
    y_prob = np.asarray(y_prob, dtype=np.float32)
    return float(np.mean((y_prob - y_true) ** 2))

def ece_score(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true, dtype=np.float32)
    y_prob = np.asarray(y_prob, dtype=np.float32)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i < n_bins - 1:
            m = (y_prob >= lo) & (y_prob < hi)
        else:
            m = (y_prob >= lo) & (y_prob <= hi)
        if m.sum() == 0:
            continue
        acc = float(y_true[m].mean())
        conf = float(y_prob[m].mean())
        ece += (m.sum() / len(y_true)) * abs(acc - conf)
    return float(ece)

def main():
    os.makedirs("results", exist_ok=True)

    df = build_tabular_features("data/shots.csv")

    # Target
    y = df["is_goal"].astype(int).to_numpy()

    # Feature kolonlarını seç
    # match_id'yi mutlaka çıkar (kimlik bilgisi)
    drop_cols = {"is_goal", "id", "player", "team", "minute", "second", "match_id"}
    X = df[[c for c in df.columns if c not in drop_cols]].copy()

    # match bazlı split
    match_ids = df["match_id"].astype(int).to_numpy()
    unique_matches = np.unique(match_ids)

    train_m, test_m = train_test_split(unique_matches, test_size=0.2, random_state=42)
    train_mask = np.isin(match_ids, train_m)
    test_mask = np.isin(match_ids, test_m)

    X_train, y_train = X.loc[train_mask], y[train_mask]
    X_test, y_test = X.loc[test_mask], y[test_mask]

    # Logistic Regression baseline (gol az olduğu için balanced faydalı olabilir)
    base_lr = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        solver="lbfgs",
    )

    model = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("cal", CalibratedClassifierCV(base_lr, method="isotonic", cv=3)),
    ])
    model.fit(X_train, y_train)

    p_train = model.predict_proba(X_train)[:, 1]
    p_test = model.predict_proba(X_test)[:, 1]

    # logloss stabilitesi
    eps = 1e-6
    p_train = np.clip(p_train, eps, 1 - eps)
    p_test = np.clip(p_test, eps, 1 - eps)

    metrics = {
        "train_logloss": log_loss(y_train, p_train),
        "test_logloss": log_loss(y_test, p_test),
        "train_auc": roc_auc_score(y_train, p_train),
        "test_auc": roc_auc_score(y_test, p_test),
        "train_brier": brier_score(y_train, p_train),
        "test_brier": brier_score(y_test, p_test),
        "test_ece": ece_score(y_test, p_test, n_bins=10),
    }

    print("=== BASELINE RESULTS (LogReg) ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # Calibration curve + bin count
    n_bins = 10
    prob_true, prob_pred = calibration_curve(y_test, p_test, n_bins=n_bins, strategy="uniform")
    bins = np.linspace(0, 1, n_bins + 1)
    counts, _ = np.histogram(p_test, bins=bins)
    print("Bin counts (uniform, 10):", counts.tolist())

    plt.figure()
    plt.plot(prob_pred, prob_true, marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("Predicted probability")
    plt.ylabel("True fraction of positives")
    plt.title(f"Calibration (Test) - LogReg | ECE={metrics['test_ece']:.3f}")
    plt.tight_layout()
    plt.savefig("results/calibration_curve.png", dpi=200)
    print("Saved: results/calibration_curve.png")

if __name__ == "__main__":
    main()
