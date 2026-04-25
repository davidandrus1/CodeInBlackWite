import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from feature_engineering import TOP_FEATURES_PER_POSITION


# ─────────────────────────────────────────────
# HELPERS — încarcă date de pe disc
# ─────────────────────────────────────────────

def _load_normalized(position: str) -> pd.DataFrame:
    """Încarcă DataFrame-ul normalizat pentru o poziție."""
    path = f"saved_data/normalized_{position}.pkl"
    try:
        return pd.read_pickle(path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Fișierul {path} nu există. Rulează train.py mai întâi."
        )


def _load_lookup() -> dict:
    """Încarcă dicționarul name → playerId."""
    with open("saved_data/players_lookup.pkl", "rb") as f:
        return pickle.load(f)


# ─────────────────────────────────────────────
# FUNCȚIA PRINCIPALĂ
# ─────────────────────────────────────────────

def compute_similarity(name_a: str, name_b: str, position: str) -> dict:
    """
    Calculează similaritatea dintre doi jucători de aceeași poziție.

    Returns un dict cu:
        - similarity_score: float 0-100
        - radar_values_a: dict {feature_label: valoare normalizată}
        - radar_values_b: dict {feature_label: valoare normalizată}
        - labels: lista de etichete pentru axele spider chart
        - error: str sau None
    """

    # 1. Încarcă datele
    try:
        df = _load_normalized(position)
    except FileNotFoundError as e:
        return {"error": str(e)}

    # 2. Găsește jucătorii în DataFrame
    row_a = df[df["name"] == name_a]
    row_b = df[df["name"] == name_b]

    if row_a.empty:
        return {"error": f"Jucătorul '{name_a}' nu a fost găsit în datele pentru {position}."}
    if row_b.empty:
        return {"error": f"Jucătorul '{name_b}' nu a fost găsit în datele pentru {position}."}

    row_a = row_a.iloc[0]
    row_b = row_b.iloc[0]

    # 3. Features disponibile pentru această poziție
    top_features = TOP_FEATURES_PER_POSITION.get(position, [])

    # Verifică că toate featurile există în DataFrame
    available = [f for f in top_features if f in df.columns]
    if len(available) == 0:
        return {"error": f"Nicio feature TOP disponibilă pentru poziția {position}."}

    # 4. Extrage vectorii completi pentru cosine similarity
    all_features = [c for c in df.columns if c not in ["playerId", "name"]]
    vec_a = row_a[all_features].values.reshape(1, -1)
    vec_b = row_b[all_features].values.reshape(1, -1)

    # 5. Calculează scorul global (cosine similarity → 0-100)
    score = float(cosine_similarity(vec_a, vec_b)[0][0]) * 100
    score = round(min(max(score, 0), 100), 1)

    # 6. Valorile pentru spider chart (doar TOP 4 features)
    labels = [_feature_to_label(f) for f in available]

    radar_a = {_feature_to_label(f): round(float(row_a[f]), 3) for f in available}
    radar_b = {_feature_to_label(f): round(float(row_b[f]), 3) for f in available}

    return {
        "error": None,
        "similarity_score": score,
        "labels": labels,
        "radar_values_a": radar_a,
        "radar_values_b": radar_b,
        "player_a": name_a,
        "player_b": name_b,
        "position": position,
    }


# ─────────────────────────────────────────────
# HELPER — nume frumos pentru axele radarului
# ─────────────────────────────────────────────

def _feature_to_label(feature: str) -> str:
    """Transformă 'avg_keyPasses' → 'Key Passes' pentru UI."""
    label = feature.replace("avg_", "").replace("pct_", "% ")
    # camelCase → cuvinte separate
    result = ""
    for char in label:
        if char.isupper():
            result += " " + char
        else:
            result += char
    return result.strip().title()


# ─────────────────────────────────────────────
# FUNCȚIE UTILITAR — jucătorii disponibili
# pentru o poziție (pentru dropdown UI)
# ─────────────────────────────────────────────

def get_players_for_position(position: str) -> list:
    """Returnează lista de nume disponibile pentru o poziție."""
    try:
        df = _load_normalized(position)
        return sorted(df["name"].dropna().unique().tolist())
    except FileNotFoundError:
        return []