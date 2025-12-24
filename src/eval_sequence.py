# src/eval_sequence.py
import os
import json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

from train_sequence import SeqConfig, GRULSTMxG, ShotSeqDataset, DEVICE
from features import compute_distance_angle, add_simple_flags, maybe_standardize_coords


def brier_score(y_true, y_prob):
    y_true = np.asarray(y_true, dtype=np.float32)
    y_prob = np.asarray(y_prob, dtype=np.float32)
    return float(np.mean((y_prob - y_true) ** 2))


def clip_probs(p, eps=1e-6):
    return np.clip(np.asarray(p, dtype=np.float32), eps, 1 - eps)


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
        prob = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)

        ys.append(y.numpy().reshape(-1))
        ps.append(prob)

    y_true = np.concatenate(ys)
    y_prob = clip_probs(np.concatenate(ps))
    return y_true, y_prob


def load_split(path="checkpoints/split.json"):
    with open(path, "r") as f:
        sp = json.load(f)
    train_m = np.array(sp["train_m"], dtype=int)
    val_m = np.array(sp.get("val_m", []), dtype=int)
    test_m = np.array(sp["test_m"], dtype=int)
    return train_m, val_m, test_m


def load_vocab(path="checkpoints/type_vocab.json"):
    with open(path, "r") as f:
        type_to_id = json.load(f)
    # json'da key'ler string, value'lar int olmalı; garantiye alalım:
    type_to_id = {str(k): int(v) for k, v in type_to_id.items()}
    return type_to_id


def eval_one(rnn_type="gru", ckpt_kind="best"):
    # cfg: train ile uyumlu olmalı (seq_len, hidden_size, type_emb_dim vs.)
    # Eğer train'de farklı kullandıysan burada da aynı yap.
    cfg = SeqConfig(seq_len=10, epochs=1, rnn_type=rnn_type)

    shots = pd.read_csv("data/shots.csv").dropna(subset=["x", "y", "is_goal"]).copy()

    # Koordinat standardizasyonu train ile aynı olmalı:
    shots = maybe_standardize_coords(shots, enable=False)

    shots = compute_distance_angle(shots)
    shots = add_simple_flags(shots)

    train_m, val_m, test_m = load_split("checkpoints/split.json")
    type_to_id = load_vocab("checkpoints/type_vocab.json")

    test_ds = ShotSeqDataset(shots, test_m, type_to_id, cfg)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=256, shuffle=False)

    model = GRULSTMxG(n_types=len(type_to_id), cfg=cfg).to(DEVICE)

    ckpt = f"checkpoints/seq_{rnn_type}_{ckpt_kind}.pt"
    if not os.path.exists(ckpt):
        # fallback: last
        ckpt = f"checkpoints/seq_{rnn_type}_last.pt"

    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))

    y_true, y_prob = predict_probs(model, test_loader)

    ll = float(log_loss(y_true, y_prob))
    brier = brier_score(y_true, y_prob)
    ece = ece_score(y_true, y_prob, n_bins=10)

    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc = float("nan")

    print(f"{rnn_type.upper()} | ckpt={os.path.basename(ckpt)} | logloss={ll:.4f} | auc={auc:.4f} | brier={brier:.4f} | ece={ece:.4f}")

    # Calibration curve + bin counts
    os.makedirs("results", exist_ok=True)

    n_bins = 10
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    bins = np.linspace(0, 1, n_bins + 1)
    counts, _ = np.histogram(y_prob, bins=bins)
    print("Bin counts (uniform, 10):", counts.tolist())

    plt.figure()
    plt.plot(prob_pred, prob_true, marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("Predicted probability")
    plt.ylabel("True fraction of positives")
    plt.title(f"Calibration (Test) - {rnn_type.upper()} | ECE={ece:.3f}")
    plt.tight_layout()

    out = f"results/calibration_curve_{rnn_type}.png"
    plt.savefig(out, dpi=200)
    print("Saved:", out)


if __name__ == "__main__":
    eval_one("gru", ckpt_kind="best")
    eval_one("lstm", ckpt_kind="best")
