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

sintetici_in_matches = df_players[df_players["playerId"] >= 9_000_001]
print(f"[DEBUG] Jucători sintetici după agregare: {len(sintetici_in_matches)}")
print(sintetici_in_matches[["playerId", "totalMinutes"]].head())
# ─────────────────────────────────────────────
# PASUL 2 — Construiești lookup-ul din players (1).json
# ─────────────────────────────────────────────
PLAYERS_JSON_PATH = "../../Date - meciuri/players (1).json"

with open(PLAYERS_JSON_PATH, "r", encoding="utf-8") as f:
    players_raw = json.load(f)

id_to_name = {}
name_to_id = {}

for player in players_raw["players"]:
    wy_id      = player["wyId"]
    short_name = player["shortName"].strip()
    id_to_name[wy_id] = short_name
    if short_name not in name_to_id:
        name_to_id[short_name] = wy_id

print(f"[OK] {len(id_to_name)} jucători reali indexați.")

# ── Adaugă jucătorii sintetici în lookup ──
SYNTHETIC_JSON_PATH = "../../Date - meciuri/players_synthetic.json"
if os.path.exists(SYNTHETIC_JSON_PATH):
    with open(SYNTHETIC_JSON_PATH, "r", encoding="utf-8") as f:
        synthetic_raw = json.load(f)
    for player in synthetic_raw["players"]:
        wy_id      = player["wyId"]
        short_name = player["shortName"].strip()
        id_to_name[wy_id] = short_name
        if short_name not in name_to_id:
            name_to_id[short_name] = wy_id
    print(f"[OK] {len(synthetic_raw['players'])} jucători sintetici adăugați în lookup.")
else:
    print("[WARN] players_synthetic.json nu a fost găsit.")

# Salvezi lookup-ul complet
lookup = {
    "id_to_name": id_to_name,
    "name_to_id": name_to_id,
}

with open("saved_data/players_lookup.pkl", "wb") as f:
    pickle.dump(lookup, f)

print(f"[OK] Lookup salvat: {len(id_to_name)} jucători indexați total.")


# ─────────────────────────────────────────────
# PASUL 3 — Adaugi coloana 'name' pe df_players
# ─────────────────────────────────────────────
df_players["name"] = df_players["playerId"].map(id_to_name)

n_fara_nume = df_players["name"].isna().sum()
if n_fara_nume > 0:
    print(f"[WARN] {n_fara_nume} jucători fără nume — excluși din pkl-uri.")

df_players = df_players[df_players["name"].notna()].copy()


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

    df_scaled = normalize_position(df_pozitie, pozitie, features)

    df_scaled["name"] = df_scaled["playerId"].map(id_to_name)

    df_scaled.to_pickle(f"saved_data/normalized_{pozitie}.pkl")

    print(f"[OK] {pozitie}: {len(df_scaled)} jucători procesați și salvați.")

print("\n[DONE] Pipeline complet. Fișiere salvate în saved_data/ și saved_models/")