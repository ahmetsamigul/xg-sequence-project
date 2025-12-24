# src/train_sequence.py
import os
import math
import numpy as np
import pandas as pd
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss

from features import compute_distance_angle, add_simple_flags

EVENT_DIR = "data/events"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def brier_score(y_true, y_prob):
    y_true = np.asarray(y_true, dtype=np.float32)
    y_prob = np.asarray(y_prob, dtype=np.float32)
    return np.mean((y_prob - y_true) ** 2)


@dataclass
class SeqConfig:
    seq_len: int = 10          # şuttan önceki kaç event
    batch_size: int = 64
    epochs: int = 6
    lr: float = 1e-3
    hidden_size: int = 64
    rnn_type: str = "gru"      # "gru" veya "lstm"
    type_emb_dim: int = 16
    grad_clip: float = 1.0


def build_type_vocab_from_events(match_ids, max_matches=120):
    """
    Neden:
    - Önceki sürümde vocab çok küçüktü, çoğu event type 0 (PAD/unknown) oluyordu.
    - Bu da sequence bağlamını zayıflatıp AUC'yi düşürebilir.
    - Vocab'u sadece TRAIN maçlarından çıkararak leakage'i önlüyoruz.
    """
    type_to_id = {"<PAD>": 0}
    seen = set()

    use_ids = list(match_ids)[:max_matches]
    for mid in use_ids:
        path = os.path.join(EVENT_DIR, f"events_{int(mid)}.csv")
        if not os.path.exists(path):
            continue

        ev = pd.read_csv(path, usecols=["type"])
        for t in ev["type"].astype(str).unique():
            if t not in seen:
                seen.add(t)
                type_to_id[t] = len(type_to_id)

    return type_to_id


class ShotSeqDataset(Dataset):
    """
    Her örnek:
      - type_ids: (T,)
      - xyt: (T,3)  -> x_norm, y_norm, t_norm
      - mask: (T,)  -> padding mask (şu an modelde doğrudan kullanılmıyor ama ileride kullanılabilir)
      - static: (6,) -> distance, angle, flags...
      - y: (1,) -> goal label
    """
    def __init__(self, shots_df, match_ids, type_to_id, cfg: SeqConfig):
        self.cfg = cfg
        self.type_to_id = type_to_id

        df = shots_df.copy()
        df = df[df["match_id"].astype(int).isin(match_ids)].reset_index(drop=True)
        self.df = df

        self._event_cache = {}  # match_id -> events df (hız için)

    def __len__(self):
        return len(self.df)

    def _load_events(self, match_id: int) -> pd.DataFrame:
        if match_id in self._event_cache:
            return self._event_cache[match_id]

        path = os.path.join(EVENT_DIR, f"events_{match_id}.csv")
        ev = pd.read_csv(path)

        # zaman saniye cinsinden
        ev["t_sec"] = ev["minute"].fillna(0).astype(int) * 60 + ev["second"].fillna(0).astype(int)

        # normalize x,y (StatsBomb: 120x80)
        ev["x_norm"] = ev["x"].fillna(0).astype(float) / 120.0
        ev["y_norm"] = ev["y"].fillna(0).astype(float) / 80.0

        # type string -> id (yoksa 0'a düşer)
        ev["type_id"] = ev["type"].astype(str).map(self.type_to_id).fillna(0).astype(int)

        self._event_cache[match_id] = ev
        return ev

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        match_id = int(row["match_id"])
        team = str(row["team"])
        shot_t = int(row["minute"]) * 60 + int(row["second"])

        ev = self._load_events(match_id)

        # Aynı takım + şuttan önce
        sub = ev[(ev["team"].astype(str) == team) & (ev["t_sec"] < shot_t)]
        sub = sub.sort_values("t_sec").tail(self.cfg.seq_len)

        T = self.cfg.seq_len
        type_ids = np.zeros((T,), dtype=np.int64)
        xyt = np.zeros((T, 3), dtype=np.float32)
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

            mask[start:] = 1.0

        # shot static features
        static = np.array([
            float(row["distance"]),
            float(row["angle"]),
            int(row.get("under_pressure", 0)),
            int(row.get("shot_one_on_one", 0)),
            int(row.get("shot_first_time", 0)),
            int(row.get("shot_aerial_won", 0)),
        ], dtype=np.float32)

        # normalize (kabaca)
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


class GRULSTMxG(nn.Module):
    def __init__(self, n_types: int, cfg: SeqConfig):
        super().__init__()
        self.cfg = cfg

        self.type_emb = nn.Embedding(n_types, cfg.type_emb_dim)
        step_in = cfg.type_emb_dim + 3  # emb + x,y,t

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
        emb = self.type_emb(type_ids)       # (B,T,E)
        seq = torch.cat([emb, xyt], dim=-1) # (B,T,E+3)
        out, _ = self.rnn(seq)              # (B,T,H)
        h_last = out[:, -1, :]              # (B,H)  (tail/padding yaklaşımıyla basit)
        z = torch.cat([h_last, static], dim=-1)
        return self.head(z)


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
        prob = torch.sigmoid(logits).cpu().numpy().reshape(-1)

        ys.append(y.numpy().reshape(-1))
        ps.append(prob)

    y_true = np.concatenate(ys)
    y_prob = np.concatenate(ps)

    return {
        "logloss": log_loss(y_true, y_prob),
        "auc": roc_auc_score(y_true, y_prob),
        "brier": brier_score(y_true, y_prob),
    }


def main():
    # Buradan GRU/LSTM seçebilirsin:
    # rnn_type="gru" veya "lstm"
    cfg = SeqConfig(seq_len=10, epochs=6, rnn_type="lstm")

    shots = pd.read_csv("data/shots.csv").dropna(subset=["x", "y", "is_goal"]).copy()

    # distance/angle + flags ekle
    shots = compute_distance_angle(shots)
    shots = add_simple_flags(shots)

    # match bazlı split
    match_ids = shots["match_id"].astype(int).unique()
    train_m, test_m = train_test_split(match_ids, test_size=0.2, random_state=42)

    # vocab'u train maçlarından çıkar (leakage olmasın)
    type_to_id = build_type_vocab_from_events(train_m, max_matches=120)
    print("Vocab size:", len(type_to_id))

    train_ds = ShotSeqDataset(shots, train_m, type_to_id, cfg)
    test_ds = ShotSeqDataset(shots, test_m, type_to_id, cfg)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    model = GRULSTMxG(n_types=len(type_to_id), cfg=cfg).to(DEVICE)

    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.BCEWithLogitsLoss()

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

            # exploding gradient'e karşı
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            optim.step()
            total += loss.item() * y.size(0)

        train_bce = total / len(train_ds)
        test_metrics = evaluate(model, test_loader)

        print(
            f"Epoch {epoch:02d} | train_bce={train_bce:.4f} | "
            f"test_logloss={test_metrics['logloss']:.4f} | "
            f"test_auc={test_metrics['auc']:.4f} | "
            f"test_brier={test_metrics['brier']:.4f}"
        )

    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = f"checkpoints/seq_{cfg.rnn_type.lower()}.pt"
    torch.save(model.state_dict(), ckpt_path)
    print("Saved:", ckpt_path)


if __name__ == "__main__":
    main()
