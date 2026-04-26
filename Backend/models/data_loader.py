import json
import os
import pandas as pd
from feature_engineering import extract_primary_position, extract_position_for_ml

def load_all_matches(data_path: str) -> pd.DataFrame:
    """
    Citește toate fișierele de meciuri și returnează
    un DataFrame cu un rând per jucător per meci.
    """
    records = []

    for filename in os.listdir(data_path):
        if not filename.endswith(".json"):
            continue
        if filename == "players (1).json":
            continue


        filepath = os.path.join(data_path, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            for player_raw in data["players"]:

                # Ignoră jucătorii care nu au jucat
                if player_raw["total"]["minutesOnField"] == 0:
                    continue

                flat = {
                    "playerId":       player_raw["playerId"],
                    "matchId":        player_raw["matchId"],
                    "minutesOnField": player_raw["total"]["minutesOnField"],
                    "position":       extract_primary_position(player_raw["positions"]),
                    "position_ml":    extract_position_for_ml(player_raw["positions"]),
                }

                # Average (per 90 min)
                for key, value in player_raw["average"].items():
                    flat[f"avg_{key}"] = value

                # Percent (eficiență)
                for key, value in player_raw["percent"].items():
                    flat[f"pct_{key}"] = value

                records.append(flat)

        except Exception as e:
            print(f"[WARN] Eroare la {filename}: {e}")

    df = pd.DataFrame(records)
    print(f"[OK] {len(df)} înregistrări jucător-meci încărcate.")
    return df


def aggregate_players(df: pd.DataFrame, min_minutes: int = 90) -> pd.DataFrame:
    """
    Agregează statisticile unui jucător din toate meciurile.
    
    Pentru avg_ : face media ponderată cu minutele jucate
    Pentru pct_ : face media simplă (sunt deja procente)
    """
    # Filtrează meciurile cu prea puține minute
    df = df[df["minutesOnField"] >= min_minutes].copy()

    avg_cols = [c for c in df.columns if c.startswith("avg_")]
    pct_cols = [c for c in df.columns if c.startswith("pct_")]

    results = []

    for player_id, group in df.groupby("playerId"):
        row = {
            "playerId":       player_id,
            "totalMinutes":   group["minutesOnField"].sum(),
            "totalMatches":   len(group),
            # Poziția dominantă = cea mai frecventă
            "position":       group["position"].mode()[0],
            "position_ml":    group["position_ml"].mode()[0],
        }

        # Media ponderată pentru avg_ (ponderat cu minutele)
        weights = group["minutesOnField"]
        for col in avg_cols:
            row[col] = (group[col] * weights).sum() / weights.sum()

        # Media simplă pentru pct_
        for col in pct_cols:
            row[col] = group[col].mean()

        results.append(row)

    df_agg = pd.DataFrame(results)
    print(f"[OK] {len(df_agg)} jucători unici după agregare.")
    return df_agg