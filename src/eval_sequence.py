# src/eval_sequence.py
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

from train_sequence import SeqConfig, GRULSTMxG, ShotSeqDataset, build_type_vocab_from_events, DEVICE, EVENT_DIR
from features import compute_distance_angle, add_simple_flags

def brier_score(y_true, y_prob):
    y_true = np.asarray(y_true, dtype=np.float32)
    y_prob = np.asarray(y_prob, dtype=np.float32)
    return np.mean((y_prob - y_true) ** 2)

@torch.no_grad()
def predict_probs(model, loader):
    model.eval()
    ys, ps = [], []
    for type_ids, xyt, mask, static, y in loader:
        type_ids = type_ids.to(DEVICE)
        xyt = xyt.to(DEVICE)
        mask = mask.to(DEVICE)
        static = static.to(DEVICE)

        logits = model(type_ids, xyt, mask, static)
        prob = torch.sigmoid(logits).cpu().numpy().reshape(-1)

        ys.append(y.numpy().reshape(-1))
        ps.append(prob)

    return np.concatenate(ys), np.concatenate(ps)

def main(rnn_type="lstm"):
    cfg = SeqConfig(seq_len=10, epochs=1, rnn_type=rnn_type)  # epochs burada önemli değil, sadece config
    shots = pd.read_csv("data/shots.csv").dropna(subset=["x", "y", "is_goal"]).copy()
    shots = compute_distance_angle(shots)
    shots = add_simple_flags(shots)

    match_ids = shots["match_id"].astype(int).unique()

    # aynı split'i yakalamak için seed sabit
    from sklearn.model_selection import train_test_split
    train_m, test_m = train_test_split(match_ids, test_size=0.2, random_state=42)

    type_to_id = build_type_vocab_from_events(train_m, max_matches=120)

    test_ds = ShotSeqDataset(shots, test_m, type_to_id, cfg)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=256, shuffle=False)

    model = GRULSTMxG(n_types=len(type_to_id), cfg=cfg).to(DEVICE)

    ckpt = f"checkpoints/seq_{rnn_type}.pt"
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))

    y_true, y_prob = predict_probs(model, test_loader)

    ll = log_loss(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    brier = brier_score(y_true, y_prob)

    print(f"{rnn_type.upper()} | logloss={ll:.4f} | auc={auc:.4f} | brier={brier:.4f}")

    # Calibration curve
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")
    os.makedirs("results", exist_ok=True)

    plt.figure()
    plt.plot(prob_pred, prob_true, marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("Predicted probability")
    plt.ylabel("True fraction of positives")
    plt.title(f"Calibration Curve (Test) - {rnn_type.upper()}")
    plt.tight_layout()
    out = f"results/calibration_curve_{rnn_type}.png"
    plt.savefig(out, dpi=200)
    print("Saved:", out)

if __name__ == "__main__":
    main("gru")
    main("lstm")
