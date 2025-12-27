# src/gradio_app.py
import os
import json
import numpy as np
import pandas as pd
import torch
import gradio as gr

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

from features import compute_distance_angle, add_simple_flags, one_hot_categoricals
from train_sequence import SeqConfig, GRULSTMxG, ShotSeqDataset, DEVICE

CHECKPOINT_DIR = "checkpoints"
DATA_SHOTS = "data/shots.csv"

def load_split(path=os.path.join(CHECKPOINT_DIR, "split.json")):
    with open(path, "r") as f:
        sp = json.load(f)
    return np.array(sp["train_m"], dtype=int), np.array(sp.get("val_m", []), dtype=int), np.array(sp["test_m"], dtype=int)

def load_vocab(path=os.path.join(CHECKPOINT_DIR, "type_vocab.json")):
    with open(path, "r") as f:
        type_to_id = json.load(f)
    type_to_id = {str(k): int(v) for k, v in type_to_id.items()}
    return type_to_id

def safe_load_model(rnn_type: str):
    best_path = os.path.join(CHECKPOINT_DIR, f"seq_{rnn_type}_best.pt")
    last_path = os.path.join(CHECKPOINT_DIR, f"seq_{rnn_type}_last.pt")

    ckpt = best_path if os.path.exists(best_path) else last_path if os.path.exists(last_path) else None
    return ckpt

@torch.no_grad()
def predict_seq(model, sample):
    type_ids, xyt, mask, static, y = sample
    type_ids = type_ids.unsqueeze(0).to(DEVICE)
    xyt = xyt.unsqueeze(0).to(DEVICE)
    mask = mask.unsqueeze(0).to(DEVICE)
    static = static.unsqueeze(0).to(DEVICE)

    logits = model(type_ids, xyt, mask, static)
    prob = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)[0]
    return float(prob)

def build_tabular_model(shots_df):
    """
    Calibrated Logistic Regression baseline (train matches üzerinden)
    """
    train_m, _, test_m = load_split()
    match_ids = shots_df["match_id"].astype(int).to_numpy()
    train_mask = np.isin(match_ids, train_m)

    df = shots_df.copy()
    df = compute_distance_angle(df)
    df = add_simple_flags(df)
    df = one_hot_categoricals(df)

    y = df["is_goal"].astype(int).to_numpy()

    drop_cols = {"is_goal", "id", "player", "team", "minute", "second", "match_id"}
    X = df[[c for c in df.columns if c not in drop_cols]].copy()

    X_train, y_train = X.loc[train_mask], y[train_mask]

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

    return model, X, y

def make_shot_label(row, idx):
    # kısa, sunumda okunur label
    mid = int(row["match_id"])
    mm = int(row["minute"])
    ss = int(row["second"])
    team = str(row.get("team", ""))
    player = str(row.get("player", ""))
    is_goal = int(row.get("is_goal", 0))
    return f"[{idx}] match={mid} | {mm:02d}:{ss:02d} | {team} | {player} | goal={is_goal}"

def main():
    if not os.path.exists(DATA_SHOTS):
        raise FileNotFoundError("data/shots.csv bulunamadı. Önce: python src/data_make.py")

    # shots yükle
    shots = pd.read_csv(DATA_SHOTS).dropna(subset=["x", "y", "is_goal"]).copy()

    # sequence için train pipeline ile aynı featurelar
    shots_seq = compute_distance_angle(shots)
    shots_seq = add_simple_flags(shots_seq)

    # split + vocab
    split_path = os.path.join(CHECKPOINT_DIR, "split.json")
    vocab_path = os.path.join(CHECKPOINT_DIR, "type_vocab.json")
    if not os.path.exists(split_path) or not os.path.exists(vocab_path):
        raise FileNotFoundError("checkpoints/split.json veya checkpoints/type_vocab.json yok. Önce train_sequence.py çalıştır.")

    type_to_id = load_vocab(vocab_path)

    # Dataset: tüm maçları al (demo için)
    all_match_ids = shots_seq["match_id"].astype(int).unique()
    cfg_gru = SeqConfig(seq_len=10, epochs=1, rnn_type="gru")
    cfg_lstm = SeqConfig(seq_len=10, epochs=1, rnn_type="lstm")

    ds_gru = ShotSeqDataset(shots_seq, all_match_ids, type_to_id, cfg_gru)
    ds_lstm = ShotSeqDataset(shots_seq, all_match_ids, type_to_id, cfg_lstm)

    # modelleri yükle (varsa)
    models = {}
    for rnn_type, cfg in [("gru", cfg_gru), ("lstm", cfg_lstm)]:
        ckpt = safe_load_model(rnn_type)
        if ckpt:
            m = GRULSTMxG(n_types=len(type_to_id), cfg=cfg).to(DEVICE)
            m.load_state_dict(torch.load(ckpt, map_location=DEVICE))
            m.eval()
            models[rnn_type] = (m, ckpt)
        else:
            models[rnn_type] = (None, None)

    # tabular modeli eğit (bir kere)
    tab_model, tab_X, tab_y = build_tabular_model(shots)

    # dropdown seçenekleri
    # demo için çok büyümesin diye random 300 şut seçelim
    shots_small = shots.reset_index(drop=True).sample(n=min(300, len(shots)), random_state=42).reset_index(drop=True)
    # label -> global index eşleşmesi için orijinal indexi de tutalım:
    # en garantisi: orijinal shots_seq dataframe indexini taşıyalım
    shots_small["__orig_idx__"] = shots_small.index

    # burada basitçe ds index olarak shots_small satır numarasını kullanacağız
    # ama ds, shots_seq'in kendi sırasına göre; o yüzden mapping kuruyoruz:
    # En pratik: shots_seq resetlenmiş bir dataframe; aynı okuma ile aynı sırada.
    # Biz dropdown için shots_seq üzerinden seçelim:
    shots_seq_reset = shots_seq.reset_index(drop=True)

    # 300 index seç
    chosen_idx = np.random.RandomState(42).choice(len(shots_seq_reset), size=min(300, len(shots_seq_reset)), replace=False)
    chosen_idx = chosen_idx.tolist()

    labels = [make_shot_label(shots_seq_reset.iloc[i], i) for i in chosen_idx]
    label_to_idx = {labels[k]: chosen_idx[k] for k in range(len(labels))}

    def predict(shot_choice):
        idx = label_to_idx[shot_choice]
        row = shots_seq_reset.iloc[idx]

        # tabular proba (aynı index sırası shots üzerinde birebir olmayabilir; bu yüzden tabular için row bazlı feature üretelim)
        one = pd.DataFrame([shots.iloc[idx]]) if idx < len(shots) else pd.DataFrame([shots.iloc[0]])
        one = compute_distance_angle(one)
        one = add_simple_flags(one)
        one = one_hot_categoricals(one)

        drop_cols = {"is_goal", "id", "player", "team", "minute", "second", "match_id"}
        X_one = one[[c for c in one.columns if c not in drop_cols]].copy()

        # tab_X'in kolonları ile hizala (eksik kolonlar 0)
        for c in tab_X.columns:
            if c not in X_one.columns:
                X_one[c] = 0
        X_one = X_one[tab_X.columns]

        tab_p = float(tab_model.predict_proba(X_one)[:, 1][0])

        # sequence proba
        out_lines = []
        info = {
            "match_id": int(row["match_id"]),
            "time": f"{int(row['minute']):02d}:{int(row['second']):02d}",
            "team": str(row.get("team","")),
            "player": str(row.get("player","")),
            "is_goal": int(row["is_goal"]),
            "distance": float(row["distance"]),
            "angle(rad)": float(row["angle"]),
        }

        # decode last events type ids (PAD hariç)
        id_to_type = {v:k for k,v in type_to_id.items()}

        # GRU
        gru_model, gru_ckpt = models["gru"]
        if gru_model is not None:
            sample = ds_gru[idx]
            gru_p = predict_seq(gru_model, sample)
            type_ids = sample[0].numpy().tolist()
            ctx = [id_to_type.get(t, "<UNK>") for t in type_ids if t != 0]
        else:
            gru_p = None
            ctx = []

        # LSTM
        lstm_model, lstm_ckpt = models["lstm"]
        if lstm_model is not None:
            sample2 = ds_lstm[idx]
            lstm_p = predict_seq(lstm_model, sample2)
        else:
            lstm_p = None

        # output markdown
        md = []
        md.append("### Shot Info")
        md.append(f"- **Match**: {info['match_id']}")
        md.append(f"- **Time**: {info['time']}")
        md.append(f"- **Team / Player**: {info['team']} / {info['player']}")
        md.append(f"- **Label (is_goal)**: {info['is_goal']}")
        md.append(f"- **Distance**: {info['distance']:.2f} (raw), **Angle**: {info['angle(rad)']:.3f} rad")

        md.append("\n### xG Predictions")
        md.append(f"- **Tabular (Calibrated LR)**: `{tab_p:.3f}`")

        if gru_p is not None:
            md.append(f"- **GRU (best)**: `{gru_p:.3f}`  _(ckpt: {os.path.basename(gru_ckpt)})_")
        else:
            md.append("- **GRU**: _(checkpoint not found)_")

        if lstm_p is not None:
            md.append(f"- **LSTM (best)**: `{lstm_p:.3f}`  _(ckpt: {os.path.basename(lstm_ckpt)})_")
        else:
            md.append("- **LSTM**: _(checkpoint not found)_")

        if ctx:
            md.append("\n### Pre-shot Context (last events, same team)")
            md.append("`" + "  |  ".join(ctx[-10:]) + "`")

        return "\n".join(md)

    with gr.Blocks(title="xG Demo (Tabular vs GRU/LSTM)") as demo:
        gr.Markdown("# xG Demo: Tabular vs Sequence (GRU/LSTM)\nChoose a real shot from the dataset and compare xG predictions.")
        shot_dd = gr.Dropdown(choices=labels, value=labels[0], label="Select a shot (real example)")
        btn = gr.Button("Predict xG")
        out = gr.Markdown()

        btn.click(fn=predict, inputs=[shot_dd], outputs=[out])

        gr.Markdown(
            "### Notes\n"
            "- GRU/LSTM use the last `seq_len` pre-shot events from the same team.\n"
            "- Tabular baseline uses distance/angle + simple flags and is calibrated.\n"
        )

    demo.launch(share=True)

if __name__ == "__main__":
    main()
