import json
import os
import pickle

import pandas as pd

from data_loader import load_all_matches, aggregate_players
from feature_engineering import get_features_for_position, handle_missing_values, FEATURES_PER_POSITION
from normalization import normalize_position


# ─────────────────────────────────────────────
# PASUL 0 — Creezi folderele de output
# ─────────────────────────────────────────────
os.makedirs("saved_models", exist_ok=True)
os.makedirs("saved_data", exist_ok=True)


# ─────────────────────────────────────────────
# PASUL 1 — Încarci și agregezi meciurile
# ─────────────────────────────────────────────
df_matches = load_all_matches("../../Date - meciuri/")
df_players = aggregate_players(df_matches, min_minutes=45)


# ─────────────────────────────────────────────
# PASUL 2 — Construiești lookup-ul din players (1).json
# wyId  →  shortName  (pentru afișare)
# shortName → wyId   (pentru căutare din UI)
# ─────────────────────────────────────────────

PLAYERS_JSON_PATH = "../../Date - meciuri/players (1).json"

with open(PLAYERS_JSON_PATH, "r", encoding="utf-8") as f:
    players_raw = json.load(f)

id_to_name = {}
name_to_id = {}

for player in players_raw["players"]:
    wy_id = player["wyId"]
    short_name = player["shortName"].strip()
    id_to_name[wy_id] = short_name
    # În caz de duplicate de shortName, păstrezi primul
    if short_name not in name_to_id:
        name_to_id[short_name] = wy_id

lookup = {
    "id_to_name": id_to_name,
    "name_to_id": name_to_id,
}

with open("saved_data/players_lookup.pkl", "wb") as f:
    pickle.dump(lookup, f)

print(f"[OK] Lookup salvat: {len(id_to_name)} jucători indexați.")


# ─────────────────────────────────────────────
# PASUL 3 — Adaugi coloana 'name' pe df_players
# (din wyId → shortName, util pentru similarity engine)
# ─────────────────────────────────────────────
df_players["name"] = df_players["playerId"].map(id_to_name)


# ─────────────────────────────────────────────
# PASUL 4 — Procesezi fiecare poziție și salvezi
# ─────────────────────────────────────────────
for pozitie in FEATURES_PER_POSITION.keys():

    df_pozitie = df_players[df_players["position_ml"] == pozitie].copy()

    if len(df_pozitie) == 0:
        print(f"[SKIP] Niciun jucător pentru {pozitie}")
        continue

    df_pozitie = get_features_for_position(df_pozitie, pozitie)
    df_pozitie = handle_missing_values(df_pozitie)

    features = [c for c in df_pozitie.columns if c != "playerId"]

    # Normalizezi + salvezi scaler-ul (deja făceai asta)
    df_scaled = normalize_position(df_pozitie, pozitie, features)

    # ── NOU: adaugi coloana 'name' pe df_scaled pentru lookup rapid ──
    df_scaled["name"] = df_scaled["playerId"].map(id_to_name)

    # ── NOU: salvezi DataFrame-ul normalizat pe disc ──
    df_scaled.to_pickle(f"saved_data/normalized_{pozitie}.pkl")

    print(f"[OK] {pozitie}: {len(df_scaled)} jucători procesați și salvați.")

print("\n[DONE] Pipeline complet. Fișiere salvate în saved_data/ și saved_models/")