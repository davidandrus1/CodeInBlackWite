import json
import os
import pickle
import pandas as pd

from data_loader import load_all_matches, aggregate_players
from feature_engineering import get_features_for_position, handle_missing_values, FEATURES_PER_POSITION
from normalization import normalize_position

BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # /Backend/models/
BACKEND_DIR = os.path.dirname(BASE_DIR)                 # /Backend/
ROOT_DIR = os.path.dirname(BACKEND_DIR)                 # / (Root)

DATA_DIR = os.path.join(ROOT_DIR, "Data", "Date - meciuri")
PLAYERS_JSON_PATH = os.path.join(DATA_DIR, "players (1).json")

SAVED_MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
SAVED_DATA_DIR = os.path.join(BASE_DIR, "saved_data")

os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
os.makedirs(SAVED_DATA_DIR, exist_ok=True)


df_matches = load_all_matches(DATA_DIR)
df_players = aggregate_players(df_matches, min_minutes=45)

PLAYERS_JSON_PATH = os.path.join(DATA_DIR, "players (1).json")

with open(PLAYERS_JSON_PATH, "r", encoding="utf-8") as f:
    players_raw = json.load(f)

id_to_name = {}
name_to_id = {}

for player in players_raw["players"]:
    wy_id = player.get("wyId", player.get("id"))
    
    # 🚨 THE FIX: Extract the FULL NAME instead of shortName
    first_name = player.get('firstName', '')
    last_name = player.get('lastName', '')
    
    if first_name or last_name:
        full_name = f"{first_name} {last_name}".strip()
    else:
        full_name = player.get('name', player.get('shortName', 'Unknown')).strip()
        
    id_to_name[wy_id] = full_name
    if full_name not in name_to_id:
        name_to_id[full_name] = wy_id

lookup = {
    "id_to_name": id_to_name,
    "name_to_id": name_to_id,
}

with open(os.path.join(SAVED_DATA_DIR, "players_lookup.pkl"), "wb") as f:
    pickle.dump(lookup, f)

print(f"[OK] Lookup salvat: {len(id_to_name)} jucători indexați cu NUME COMPLET.")

df_players["name"] = df_players["playerId"].map(id_to_name)

for pozitie in FEATURES_PER_POSITION.keys():
    df_pozitie = df_players[df_players["position_ml"] == pozitie].copy()

    if len(df_pozitie) == 0:
        continue

    df_pozitie = get_features_for_position(df_pozitie, pozitie)
    df_pozitie = handle_missing_values(df_pozitie)

    features = [c for c in df_pozitie.columns if c != "playerId"]

    df_scaled = normalize_position(
        df_pozitie, 
        pozitie, 
        features, 
        save_path=os.path.join(SAVED_MODELS_DIR, "")
    )

    df_scaled["name"] = df_scaled["playerId"].map(id_to_name)

    save_path = os.path.join(SAVED_DATA_DIR, f"normalized_{pozitie}.pkl")
    df_scaled.to_pickle(save_path)

    print(f"[OK] {pozitie}: {len(df_scaled)} jucători procesați și salvați cu nume complet.")

print("\n[DONE] Pipeline complet. Fișiere salvate în saved_data/ și saved_models/")