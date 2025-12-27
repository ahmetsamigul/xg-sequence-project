# src/eval_sequence.py
import os
import json
import math
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, log_loss

from features import compute_distance_angle, add_simple_flags, maybe_standardize_coords

EVENT_DIR = "data/events"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------
# Small helpers
# -----------------------------
def clip_probs(p, eps=1e-6):
    p = np.asarray(p, dtype=np.float32)
    return np.clip(p, eps, 1.0 - eps)

def brier_score(y_true, y_prob):
    y_true = np.asarray(y_true, dtype=np.float32)
    y_prob = np.asarray(y_prob, dtype=np.float32)
    return float(np.mean((y_prob - y_true) ** 2))

def ece_from_bins(y_true, y_prob, bin_edges):
    """
    Generic ECE for given bin edges (len = K+1).
    """
    y_true = np.asarray(y_true, dtype=np.float32)
    y_prob = np.asarray(y_prob, dtype=np.float32)
    N = len(y_true)

    ece = 0.0
    counts = []

    for i in range(len(bin_edges) - 1):
        lo, hi = float(bin_edges[i]), float(bin_edges[i + 1])
        if i < len(bin_edges) - 2:
            m = (y_prob >= lo) & (y_prob < hi)
        else:
            m = (y_prob >= lo) & (y_prob <= hi)

        c = int(m.sum())
        if c == 0:
            continue

        acc = float(y_true[m].mean())
        conf = float(y_prob[m].mean())
        ece += (c / N) * abs(acc - conf)
        counts.append(c)

    return float(ece), counts

def uniform_edges(n_bins=10):
    return np.linspace(0.0, 1.0, n_bins + 1)

def quantile_edges(y_prob, n_bins=10):
    """
    Quantile bin edges -> (almost) equal sample per bin.
    Duplicate edges can happen if probabilities are repeated a lot.
    We also force [0,1] endpoints for nicer plotting.
    """
    y_prob = np.asarray(y_prob, dtype=np.float32)
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(y_prob, qs).astype(np.float32)

    # Force endpoints for nicer axes
    edges[0] = 0.0
    edges[-1] = 1.0

    # Remove duplicates (can happen with many identical probabilities)
    uniq = np.unique(edges)
    if len(uniq) < 3:
        # fallback to uniform if quantiles collapse
        return uniform_edges(n_bins)
    return uniq

def calibration_points(y_true, y_prob, bin_edges):
    """
    For calibration curve: returns (conf_list, acc_list, counts_list)
    per non-empty bin.
    """
    y_true = np.asarray(y_true, dtype=np.float32)
    y_prob = np.asarray(y_prob, dtype=np.float32)

    confs, accs, counts = [], [], []

    for i in range(len(bin_edges) - 1):
        lo, hi = float(bin_edges[i]), float(bin_edges[i + 1])
        if i < len(bin_edges) - 2:
            m = (y_prob >= lo) & (y_prob < hi)
        else:
            m = (y_prob >= lo) & (y_prob <= hi)

        c = int(m.sum())
        if c == 0:
            continue

        confs.append(float(y_prob[m].mean()))
        accs.append(float(y_true[m].mean()))
        counts.append(c)

    return confs, accs, counts


# -----------------------------
# Minimal SeqConfig (match train_sequence defaults)
# -----------------------------
class SeqConfig:
    def __init__(self, seq_len=10, batch_size=256, hidden_size=64, rnn_type="gru", type_emb_dim=16):
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        self.type_emb_dim = type_emb_dim


# -----------------------------
# Dataset (same logic as train_sequence)
# -----------------------------
class ShotSeqDataset(Dataset):
    def __init__(self, shots_df, match_ids, type_to_id, cfg: SeqConfig):
        self.cfg = cfg
        self.type_to_id = type_to_id

        df = shots_df.copy()
        df = df[df["match_id"].astype(int).isin(match_ids)].reset_index(drop=True)
        self.df = df

        self._event_cache = {}  # match_id -> events df

    def __len__(self):
        return len(self.df)

    def _load_events(self, match_id: int) -> pd.DataFrame:
        if match_id in self._event_cache:
            return self._event_cache[match_id]

        path = os.path.join(EVENT_DIR, f"events_{match_id}.csv")
        ev = pd.read_csv(path)

        for col in ["minute", "second"]:
            if col not in ev.columns:
                ev[col] = 0
        for col in ["period", "index"]:
            if col in ev.columns:
                ev[col] = ev[col].fillna(0).astype(int)
            else:
                ev[col] = 0

        ev["t_sec"] = ev["minute"].fillna(0).astype(int) * 60 + ev["second"].fillna(0).astype(int)

        if "x" not in ev.columns:
            ev["x"] = 0.0
        if "y" not in ev.columns:
            ev["y"] = 0.0
        ev["x_norm"] = ev["x"].fillna(0).astype(float) / 120.0
        ev["y_norm"] = ev["y"].fillna(0).astype(float) / 80.0

        unk_id = self.type_to_id.get("<UNK>", 1)
        if "type" not in ev.columns:
            ev["type"] = "<UNK>"
        ev["type_id"] = ev["type"].astype(str).map(self.type_to_id).fillna(unk_id).astype(int)

        if "team" not in ev.columns:
            ev["team"] = ""

        self._event_cache[match_id] = ev
        return ev

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        match_id = int(row["match_id"])
        shot_team = str(row["team"])
        shot_t = int(row["minute"]) * 60 + int(row["second"])

        ev = self._load_events(match_id)

        sub = ev[ev["t_sec"] < shot_t].copy()
        sub["same_team"] = (sub["team"].astype(str) == shot_team).astype(np.float32)
        sub = sub.sort_values(["period", "t_sec", "index"]).tail(self.cfg.seq_len)

        T = self.cfg.seq_len
        type_ids = np.zeros((T,), dtype=np.int64)
        xyt = np.zeros((T, 4), dtype=np.float32)   # x,y,t,same_team
        mask = np.zeros((T,), dtype=np.float32)

        if len(sub) > 0:
            sub = sub.reset_index(drop=True)
            start = T - len(sub)

            type_ids[start:] = sub["type_id"].to_numpy()

            dt = (shot_t - sub["t_sec"].to_numpy()).astype(np.float32)
            t_norm = np.clip(dt / 60.0, 0.0, 200.0) / 200.0

            xyt[start:, 0] = sub["x_norm"].to_numpy(dtype=np.float32)
            xyt[start:, 1] = sub["y_norm"].to_numpy(dtype=np.float32)
            xyt[start:, 2] = t_norm
            xyt[start:, 3] = sub["same_team"].to_numpy(dtype=np.float32)

            mask[start:] = 1.0

        static = np.array([
            float(row["distance"]),
            float(row["angle"]),
            int(row.get("under_pressure", 0)),
            int(row.get("shot_one_on_one", 0)),
            int(row.get("shot_first_time", 0)),
            int(row.get("shot_aerial_won", 0)),
        ], dtype=np.float32)

        static[0] = static[0] / 120.0
        static[1] = static[1] / math.pi

        y = float(row["is_goal"])

        return (
            torch.tensor(type_ids, dtype=torch.long),
            torch.tensor(xyt, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.float32),
            torch.tensor(static, dtype=torch.float32),
            torch.tensor([y], dtype=torch.float32),
        )


# -----------------------------
# Model (same as train_sequence)
# -----------------------------
class GRULSTMxG(nn.Module):
    def __init__(self, n_types: int, cfg: SeqConfig):
        super().__init__()
        self.cfg = cfg

        self.type_emb = nn.Embedding(n_types, cfg.type_emb_dim, padding_idx=0)
        step_in = cfg.type_emb_dim + 4  # emb + x,y,t,same_team

        rnn_cls = nn.GRU if cfg.rnn_type.lower() == "gru" else nn.LSTM
        self.rnn = rnn_cls(
            input_size=step_in,
            hidden_size=cfg.hidden_size,
            batch_first=True
        )

        self.head = nn.Sequential(
            nn.Linear(cfg.hidden_size + 6, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # logits
        )

    def forward(self, type_ids, xyt, mask, static):
        emb = self.type_emb(type_ids)            # (B,T,E)
        seq = torch.cat([emb, xyt], dim=-1)      # (B,T,E+4)
        out, _ = self.rnn(seq)                   # (B,T,H)

        lengths = mask.sum(dim=1).long()
        idx = (lengths - 1).clamp(min=0)
        b = torch.arange(out.size(0), device=out.device)
        h_last = out[b, idx, :]

        z = torch.cat([h_last, static], dim=-1)
        return self.head(z)


# -----------------------------
# Evaluation
# -----------------------------
@torch.no_grad()
def get_probs(model, loader):
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

def metrics(y_true, y_prob):
    auc = None
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float("nan")

    return {
        "logloss": float(log_loss(y_true, y_prob)),
        "auc": float(auc),
        "brier": brier_score(y_true, y_prob),
    }

def plot_calibration(y_true, y_prob, model_name, out_path, n_bins=10):
    # Uniform ECE
    u_edges = uniform_edges(n_bins)
    ece_u, u_counts = ece_from_bins(y_true, y_prob, u_edges)

    # Quantile bins for plot (+ a quantile-based ECE just for the plot)
    q_edges = quantile_edges(y_prob, n_bins)
    ece_q, q_counts = ece_from_bins(y_true, y_prob, q_edges)

    confs, accs, _ = calibration_points(y_true, y_prob, q_edges)

    # ---- Zoom to model's prediction range (makes plot readable) ----
    pmax = float(np.max(y_prob))
    x_max = min(1.0, pmax + 0.02)   # small margin
    x_max = max(x_max, 0.15)        # avoid too tiny axis if all probs are very small

    plt.figure()
    plt.plot(confs, accs, marker="o")
    plt.plot([0, x_max], [0, x_max], linestyle="--")  # diagonal matched to zoom
    plt.xlabel("Predicted probability")
    plt.ylabel("True fraction of positives")
    plt.title(f"Calibration (Test) - {model_name} | quantile bins | ECE={ece_q:.3f}")

    plt.xlim(0.0, x_max)

    # y-axis zoom: based on observed accuracies in bins
    ymax = float(np.max(accs)) if len(accs) else 1.0
    plt.ylim(0.0, min(1.0, max(0.25, ymax + 0.05)))

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    return ece_u, ece_q, u_counts, q_counts



def eval_one(rnn_type: str, ckpt_kind: str = "best", n_bins: int = 10):
    # Load split & vocab
    with open("checkpoints/split.json", "r") as f:
        split = json.load(f)
    with open("checkpoints/type_vocab.json", "r") as f:
        type_to_id = json.load(f)

    test_m = split["test_m"]

    # Data
    shots = pd.read_csv("data/shots.csv").dropna(subset=["x", "y", "is_goal"]).copy()
    shots = maybe_standardize_coords(shots, enable=False)
    shots = compute_distance_angle(shots)
    shots = add_simple_flags(shots)

    cfg = SeqConfig(seq_len=10, batch_size=256, hidden_size=64, rnn_type=rnn_type, type_emb_dim=16)

    test_ds = ShotSeqDataset(shots, test_m, type_to_id, cfg)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    model = GRULSTMxG(n_types=len(type_to_id), cfg=cfg).to(DEVICE)

    ckpt = f"checkpoints/seq_{rnn_type.lower()}_{ckpt_kind}.pt"
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))

    y_true, y_prob = get_probs(model, test_loader)
    m = metrics(y_true, y_prob)

    os.makedirs("results", exist_ok=True)
    out_path = f"results/calibration_curve_{rnn_type.lower()}.png"

    ece_u, ece_q, u_counts, q_counts = plot_calibration(
        y_true, y_prob,
        model_name=rnn_type.upper(),
        out_path=out_path,
        n_bins=n_bins
    )

    # Print like your log
    print(
        f"{rnn_type.upper()} | ckpt=seq_{rnn_type.lower()}_{ckpt_kind}.pt | "
        f"logloss={m['logloss']:.4f} | auc={m['auc']:.4f} | brier={m['brier']:.4f} | "
        f"ece_uniform={ece_u:.4f} | ece_quantile_plot={ece_q:.4f}"
    )
    print(f"Bin counts (uniform, {n_bins}): {u_counts}")
    print(f"Bin counts (quantile, {len(q_counts)} non-empty): {q_counts}")
    print("Saved:", out_path)


def main():
    eval_one("gru", ckpt_kind="best", n_bins=10)
    eval_one("lstm", ckpt_kind="best", n_bins=10)


if __name__ == "__main__":
    main()
