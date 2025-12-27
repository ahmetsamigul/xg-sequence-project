# src/train_sequence.py
import os
import math
import json
import random
import numpy as np
import pandas as pd
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss

from features import compute_distance_angle, add_simple_flags, maybe_standardize_coords

EVENT_DIR = "data/events"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------
# Metrics helpers
# -----------------------------
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


# -----------------------------
# Config
# -----------------------------
@dataclass
class SeqConfig:
    seq_len: int = 10
    batch_size: int = 64
    epochs: int = 8
    lr: float = 1e-3
    weight_decay: float = 0.0
    hidden_size: int = 64
    rnn_type: str = "gru"      # "gru" veya "lstm"
    type_emb_dim: int = 16
    grad_clip: float = 1.0


# -----------------------------
# Vocab
# -----------------------------
def build_type_vocab_from_events(match_ids, max_matches=None):
    """
    Train maçlarından event type vocab çıkarır.
    PAD ve UNK ayrı:
      <PAD>: 0
      <UNK>: 1
    """
    type_to_id = {"<PAD>": 0, "<UNK>": 1}
    seen = set()

    use_ids = list(match_ids) if max_matches is None else list(match_ids)[:max_matches]
    for mid in use_ids:
        path = os.path.join(EVENT_DIR, f"events_{int(mid)}.csv")
        if not os.path.exists(path):
            continue

        ev = pd.read_csv(path, usecols=["type"])
        for t in ev["type"].astype(str).unique():
            if t and t != "nan" and t not in seen:
                seen.add(t)
                type_to_id[t] = len(type_to_id)

    return type_to_id


# -----------------------------
# Dataset
# -----------------------------
class ShotSeqDataset(Dataset):
    """
    Her örnek:
      - type_ids: (T,)
      - xyt: (T,4)  -> x_norm, y_norm, t_norm, same_team_flag
      - mask: (T,)  -> 1 valid, 0 pad
      - static: (6,) -> distance, angle, flags...
      - y: (1,) -> goal label
    """
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

        # güvenli kolonlar
        for col in ["minute", "second"]:
            if col not in ev.columns:
                ev[col] = 0
        for col in ["period", "index"]:
            if col in ev.columns:
                ev[col] = ev[col].fillna(0).astype(int)
            else:
                ev[col] = 0

        # zaman saniye
        ev["t_sec"] = ev["minute"].fillna(0).astype(int) * 60 + ev["second"].fillna(0).astype(int)

        # normalize x,y
        if "x" not in ev.columns:
            ev["x"] = 0.0
        if "y" not in ev.columns:
            ev["y"] = 0.0
        ev["x_norm"] = ev["x"].fillna(0).astype(float) / 120.0
        ev["y_norm"] = ev["y"].fillna(0).astype(float) / 80.0

        # type -> id
        unk_id = self.type_to_id.get("<UNK>", 1)
        if "type" not in ev.columns:
            ev["type"] = "<UNK>"
        ev["type_id"] = ev["type"].astype(str).map(self.type_to_id).fillna(unk_id).astype(int)

        # team string
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

        # Şuttan önceki tüm eventler (hem own hem opponent)
        sub = ev[ev["t_sec"] < shot_t].copy()
        sub["same_team"] = (sub["team"].astype(str) == shot_team).astype(np.float32)

        # Daha stabil sıralama: period, t_sec, index
        sub = sub.sort_values(["period", "t_sec", "index"]).tail(self.cfg.seq_len)

        T = self.cfg.seq_len
        type_ids = np.zeros((T,), dtype=np.int64)
        xyt = np.zeros((T, 4), dtype=np.float32)   # x,y,t,same_team
        mask = np.zeros((T,), dtype=np.float32)

        if len(sub) > 0:
            sub = sub.reset_index(drop=True)
            start = T - len(sub)

            type_ids[start:] = sub["type_id"].to_numpy()

            # şuta göre göreli zaman farkı (dakika ölçeği), 0..1 normalize
            dt = (shot_t - sub["t_sec"].to_numpy()).astype(np.float32)
            t_norm = np.clip(dt / 60.0, 0.0, 200.0) / 200.0

            xyt[start:, 0] = sub["x_norm"].to_numpy(dtype=np.float32)
            xyt[start:, 1] = sub["y_norm"].to_numpy(dtype=np.float32)
            xyt[start:, 2] = t_norm
            xyt[start:, 3] = sub["same_team"].to_numpy(dtype=np.float32)

            mask[start:] = 1.0

        # static shot features
        static = np.array([
            float(row["distance"]),
            float(row["angle"]),
            int(row.get("under_pressure", 0)),
            int(row.get("shot_one_on_one", 0)),
            int(row.get("shot_first_time", 0)),
            int(row.get("shot_aerial_won", 0)),
        ], dtype=np.float32)

        # kabaca normalize
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
# Model
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
        """
        mask: (B,T) 1 valid, 0 pad
        son valid adımın hidden'ını alıyoruz (padding'le karışmasın)
        """
        emb = self.type_emb(type_ids)            # (B,T,E)
        seq = torch.cat([emb, xyt], dim=-1)      # (B,T,E+4)
        out, _ = self.rnn(seq)                   # (B,T,H)

        # last valid index per batch
        lengths = mask.sum(dim=1).long()         # (B,)
        idx = (lengths - 1).clamp(min=0)         # (B,)
        b = torch.arange(out.size(0), device=out.device)
        h_last = out[b, idx, :]                  # (B,H)

        z = torch.cat([h_last, static], dim=-1)
        return self.head(z)


# -----------------------------
# Eval loop
# -----------------------------
@torch.no_grad()
def evaluate(model, loader):
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

    # bazı test splitlerde tek sınıf olabilir (çok küçük veri vs.)
    # AUC böyle bir durumda hata verir, koruyalım:
    auc = None
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float("nan")

    return {
        "logloss": float(log_loss(y_true, y_prob)),
        "auc": float(auc),
        "brier": brier_score(y_true, y_prob),
        "ece": ece_score(y_true, y_prob, n_bins=10),
    }


# -----------------------------
# Train
# -----------------------------
def main():
    set_seed(42)

    # GRU/LSTM seç
    cfg = SeqConfig(seq_len=10, epochs=8, rnn_type="gru")

    shots = pd.read_csv("data/shots.csv").dropna(subset=["x", "y", "is_goal"]).copy()

    # Koordinat standardizasyonu gerekiyorsa enable=True yap
    shots = maybe_standardize_coords(shots, enable=False)

    # distance/angle + flags
    shots = compute_distance_angle(shots)
    shots = add_simple_flags(shots)

    # match bazlı split: train / test
    match_ids = shots["match_id"].astype(int).unique()
    train_m, test_m = train_test_split(match_ids, test_size=0.2, random_state=42)

    # train içinden val ayır
    train_m, val_m = train_test_split(train_m, test_size=0.2, random_state=42)

    os.makedirs("checkpoints", exist_ok=True)

    # split kaydet (eval aynı split'i kullanabilsin)
    split_path = "checkpoints/split.json"
    with open(split_path, "w") as f:
        json.dump(
            {
                "train_m": list(map(int, train_m)),
                "val_m": list(map(int, val_m)),
                "test_m": list(map(int, test_m)),
            },
            f,
            indent=2,
        )
    print("Saved split:", split_path)

    # vocab'u train maçlarından çıkar (tüm train)
    type_to_id = build_type_vocab_from_events(train_m, max_matches=None)
    print("Vocab size:", len(type_to_id))

    # vocab kaydet (eval birebir aynı vocab'ı yükleyebilsin)
    vocab_path = "checkpoints/type_vocab.json"
    with open(vocab_path, "w") as f:
        json.dump(type_to_id, f, indent=2)
    print("Saved vocab:", vocab_path)

    train_ds = ShotSeqDataset(shots, train_m, type_to_id, cfg)
    val_ds = ShotSeqDataset(shots, val_m, type_to_id, cfg)
    test_ds = ShotSeqDataset(shots, test_m, type_to_id, cfg)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    model = GRULSTMxG(n_types=len(type_to_id), cfg=cfg).to(DEVICE)

    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val = float("inf")
    best_path = f"checkpoints/seq_{cfg.rnn_type.lower()}_best.pt"

    # -----------------------------
    # History (learning curves)
    # -----------------------------
    history = {
        "epoch": [],
        "train_bce": [],
        "val_logloss": [],
        "val_auc": [],
        "test_logloss": [],
        "test_auc": [],
    }

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total = 0.0

        for type_ids, xyt, mask, static, y in train_loader:
            type_ids = type_ids.to(DEVICE)
            xyt = xyt.to(DEVICE)
            mask = mask.to(DEVICE)
            static = static.to(DEVICE)
            y = y.to(DEVICE)

            optim.zero_grad()
            logits = model(type_ids, xyt, mask, static)
            loss = loss_fn(logits, y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optim.step()
            total += loss.item() * y.size(0)

        train_bce = total / max(1, len(train_ds))
        val_metrics = evaluate(model, val_loader)
        test_metrics = evaluate(model, test_loader)

        # best checkpoint by val logloss
        if val_metrics["logloss"] < best_val:
            best_val = val_metrics["logloss"]
            torch.save(model.state_dict(), best_path)

        print(
            f"Epoch {epoch:02d} | "
            f"train_bce={train_bce:.4f} | "
            f"val_logloss={val_metrics['logloss']:.4f} val_auc={val_metrics['auc']:.4f} "
            f"val_brier={val_metrics['brier']:.4f} val_ece={val_metrics['ece']:.4f} | "
            f"test_logloss={test_metrics['logloss']:.4f} test_auc={test_metrics['auc']:.4f} "
            f"test_brier={test_metrics['brier']:.4f} test_ece={test_metrics['ece']:.4f} | "
            f"best_val={best_val:.4f}"
        )

        # history update
        history["epoch"].append(epoch)
        history["train_bce"].append(train_bce)
        history["val_logloss"].append(val_metrics["logloss"])
        history["val_auc"].append(val_metrics["auc"])
        history["test_logloss"].append(test_metrics["logloss"])
        history["test_auc"].append(test_metrics["auc"])

    # -----------------------------
    # Plot learning curves (TRAIN BITTİKTEN SONRA)
    # -----------------------------
    os.makedirs("results", exist_ok=True)

    # Loss curves
    plt.figure()
    plt.plot(history["epoch"], history["train_bce"], marker="o")
    plt.plot(history["epoch"], history["val_logloss"], marker="o")
    plt.plot(history["epoch"], history["test_logloss"], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Learning Curves (Loss) - {cfg.rnn_type.upper()}")
    plt.legend(["train_bce", "val_logloss", "test_logloss"])
    plt.tight_layout()
    out1 = f"results/learning_curve_loss_{cfg.rnn_type.lower()}.png"
    plt.savefig(out1, dpi=200)
    print("Saved:", out1)

    # AUC curves
    plt.figure()
    plt.plot(history["epoch"], history["val_auc"], marker="o")
    plt.plot(history["epoch"], history["test_auc"], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.title(f"Learning Curves (AUC) - {cfg.rnn_type.upper()}")
    plt.legend(["val_auc", "test_auc"])
    plt.tight_layout()
    out2 = f"results/learning_curve_auc_{cfg.rnn_type.lower()}.png"
    plt.savefig(out2, dpi=200)
    print("Saved:", out2)

    # last checkpoint da kaydet
    last_path = f"checkpoints/seq_{cfg.rnn_type.lower()}_last.pt"
    torch.save(model.state_dict(), last_path)

    print("Saved best:", best_path)
    print("Saved last:", last_path)



if __name__ == "__main__":
    main()
