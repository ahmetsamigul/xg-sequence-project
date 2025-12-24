# src/data_make.py
import os
import pandas as pd
from statsbombpy import sb
from tqdm import tqdm
from utils_sb import sb_to_name

OUT_DIR = "data"
OUT_CSV = os.path.join(OUT_DIR, "shots.csv")

def pick_competition():
    comps = sb.competitions()
    return comps

def load_shots_from_match(match_id: int) -> pd.DataFrame:
    events = sb.events(match_id=match_id)

    # type -> name
    events["type_name"] = events["type"].apply(sb_to_name)

    # sadece Shot
    shots = events[events["type_name"] == "Shot"].copy()
    if shots.empty:
        return shots

    # outcome -> name, label
    if "shot_outcome" in shots.columns:
        shots["shot_outcome_name"] = shots["shot_outcome"].apply(sb_to_name)
        shots["is_goal"] = (shots["shot_outcome_name"] == "Goal").astype(int)
    else:
        shots["is_goal"] = 0

    # location -> x,y
    loc = shots["location"].apply(lambda v: v if isinstance(v, list) and len(v) == 2 else [None, None])
    shots["x"] = loc.apply(lambda t: t[0])
    shots["y"] = loc.apply(lambda t: t[1])

    # kategorikler -> name
    for c in ["shot_body_part", "shot_type", "team", "player"]:
        if c in shots.columns:
            shots[c] = shots[c].apply(sb_to_name)

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

    keep_existing = [c for c in keep if c in shots.columns]
    shots = shots[keep_existing].copy()
    return shots

def build_dataset(competition_id: int, season_id: int, max_matches: int | None = None) -> pd.DataFrame:
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

    return pd.concat(all_shots, ignore_index=True)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    comps = pick_competition()
    print("Available competitions (showing first 15 rows):")
    print(comps[["competition_id", "season_id", "competition_name", "season_name"]].head(15))

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
