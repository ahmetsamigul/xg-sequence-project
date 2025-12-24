# src/events_make.py
import os
import pandas as pd
from statsbombpy import sb
from tqdm import tqdm

OUT_DIR = "data/events"

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    shots = pd.read_csv("data/shots.csv")
    match_ids = sorted(shots["match_id"].astype(int).unique().tolist())

    print(f"Matches to download events for: {len(match_ids)}")

    for mid in tqdm(match_ids, desc="Downloading events"):
        out_path = os.path.join(OUT_DIR, f"events_{mid}.csv")
        if os.path.exists(out_path):
            continue

        ev = sb.events(match_id=mid)

        # Sadece işimize yarayacak kolonları tutalım (hafifletmek için)
        keep = [
            "match_id", "id", "index", "period", "timestamp",
            "minute", "second",
            "type", "team", "player",
            "location",
            "possession", "possession_team",
        ]
        keep_existing = [c for c in keep if c in ev.columns]
        ev = ev[keep_existing].copy()

        # location -> x,y
        loc = ev["location"].apply(lambda v: v if isinstance(v, list) and len(v) == 2 else [None, None])
        ev["x"] = loc.apply(lambda t: t[0])
        ev["y"] = loc.apply(lambda t: t[1])
        ev = ev.drop(columns=["location"], errors="ignore")

        ev.to_csv(out_path, index=False)

    print("Done. Events saved under:", OUT_DIR)

if __name__ == "__main__":
    main()
