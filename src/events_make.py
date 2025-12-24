# src/events_make.py
import os
import numpy as np
import pandas as pd
from statsbombpy import sb
from tqdm import tqdm
from utils_sb import sb_to_name

OUT_DIR = "data/events"

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    shots = pd.read_csv("data/shots.csv")
    match_ids = sorted(shots["match_id"].astype(int).unique().tolist())

    print(f"Matches to download events for: {len(match_ids)}")

    for mid in tqdm(match_ids, desc="Downloading events"):
        out_path = os.path.join(OUT_DIR, f"events_{mid}.csv")

        ev = sb.events(match_id=mid)

        keep = [
            "match_id", "id", "index", "period", "timestamp",
            "minute", "second",
            "type", "team", "player",
            "location",
            "possession", "possession_team",
        ]
        keep_existing = [c for c in keep if c in ev.columns]
        ev = ev[keep_existing].copy()

        # type/team/player/possession_team -> name
        if "type" in ev.columns:
            ev["type"] = ev["type"].apply(sb_to_name)

        for col in ["team", "player", "possession_team"]:
            if col in ev.columns:
                ev[col] = ev[col].apply(sb_to_name)

        # location -> x,y (location yoksa güvenli davran)
        if "location" in ev.columns:
            loc = ev["location"].apply(lambda v: v if isinstance(v, list) and len(v) == 2 else [None, None])
            ev["x"] = loc.apply(lambda t: t[0])
            ev["y"] = loc.apply(lambda t: t[1])
            ev = ev.drop(columns=["location"], errors="ignore")
        else:
            ev["x"] = np.nan
            ev["y"] = np.nan

        ev.to_csv(out_path, index=False)

    print("Done. Events saved under:", OUT_DIR)

if __name__ == "__main__":
    main()
