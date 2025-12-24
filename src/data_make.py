# src/data_make.py
import os
import pandas as pd
from statsbombpy import sb
from tqdm import tqdm

OUT_DIR = "data"
OUT_CSV = os.path.join(OUT_DIR, "shots.csv")

def pick_competition():
    """
    Neden var?
    - StatsBomb open data içinde bir sürü lig/sezon var.
    - Biz önce kullanıcıya görünen bir liste verip bir tane seçiyoruz.
    """
    comps = sb.competitions()
    # En çok kullanılan birkaç competition'ı filtreleyip göstermek istersen burada filtre koyabilirsin.
    return comps

def load_shots_from_match(match_id: int) -> pd.DataFrame:
    """
    Neden bu fonksiyon?
    - Her match için events çekiyoruz.
    - Sadece Shot eventlerini ayıklayıp standardize ediyoruz.
    """
    events = sb.events(match_id=match_id)

    # Shot olaylarını seç
    shots = events[events["type"] == "Shot"].copy()
    if shots.empty:
        return shots

    # Label: goal mı?
    # shot_outcome name == "Goal" ise 1, değilse 0
    shots["is_goal"] = (shots["shot_outcome"] == "Goal").astype(int)

    # Konum (x,y) çıkar
    # StatsBomb location genelde [x, y] formatında
    loc = shots["location"].apply(lambda v: v if isinstance(v, list) and len(v) == 2 else [None, None])
    shots["x"] = loc.apply(lambda t: t[0])
    shots["y"] = loc.apply(lambda t: t[1])

    # Temel kolonları seç (başlangıç için yeterli)
    keep = [
        "match_id",
        "id",
        "minute",
        "second",
        "team",
        "player",
        "x",
        "y",
        "shot_body_part",
        "shot_type",
        "shot_first_time",
        "shot_one_on_one",
        "shot_aerial_won",
        "under_pressure",
        "is_goal",
    ]

    # Bazı kolonlar her maçta olmayabilir; var olanları al
    keep_existing = [c for c in keep if c in shots.columns]
    shots = shots[keep_existing].copy()

    return shots

def build_dataset(competition_id: int, season_id: int, max_matches: int | None = None) -> pd.DataFrame:
    """
    Neden match-bazlı çekiyoruz?
    - Sonradan train/val/test split'i match bazlı yapmak için match_id'yi tutacağız.
    """
    matches = sb.matches(competition_id=competition_id, season_id=season_id)
    match_ids = matches["match_id"].astype(int).tolist()

    if max_matches is not None:
        match_ids = match_ids[:max_matches]

    all_shots = []
    for mid in tqdm(match_ids, desc="Downloading matches"):
        try:
            df = load_shots_from_match(mid)
            if not df.empty:
                all_shots.append(df)
        except Exception as e:
            print(f"[WARN] match_id={mid} failed: {e}")

    if not all_shots:
        return pd.DataFrame()

    shots = pd.concat(all_shots, ignore_index=True)
    return shots

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    comps = pick_competition()
    # Kullanıcıya birkaç satır gösterelim
    print("Available competitions (showing first 15 rows):")
    print(comps[["competition_id", "season_id", "competition_name", "season_name"]].head(15))

    # Burayı şimdilik sabit seçelim: istersen sonra input ile seçtiririz
    # ÖRNEK: La Liga 2020/21 vs değişebilir. Çıktıdan bakıp güncelle.
    competition_id = int(input("competition_id gir: ").strip())
    season_id = int(input("season_id gir: ").strip())

    shots = build_dataset(competition_id, season_id, max_matches=None)

    if shots.empty:
        print("No shots found. Try different competition/season.")
        return

    shots.to_csv(OUT_CSV, index=False)
    print(f"Saved: {OUT_CSV} | rows={len(shots)}")
    print("Columns:", list(shots.columns))

if __name__ == "__main__":
    main()
