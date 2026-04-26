import os
import json
import pickle
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity
from functools import lru_cache # 🚨 ADDED FOR I/O OPTIMIZATION

from models.feature_engineering import (
    TOP_FEATURES_PER_POSITION,
    STYLE_FEATURES_PER_POSITION,
    QUALITY_FEATURES_PER_POSITION,
    QUALITY_WEIGHTS_PER_POSITION,
    PERFORMANCE_FEATURES_PER_POSITION,
    SURPRISE_FEATURES,
    FIELD_POSITIONS,
)

# ─────────────────────────────────────────────
# LOADERS (CACHED TO PREVENT I/O BOTTLENECKS)
# ─────────────────────────────────────────────

def _base_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))

@lru_cache(maxsize=16) # 🚨 FIX: Cache dataframes in memory so we don't read from disk 150 times!
def _load_normalized(position: str) -> pd.DataFrame:
    path = os.path.join(_base_dir(), "saved_data", f"normalized_{position}.pkl")
    try:
        return pd.read_pickle(path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Fișierul pentru {position} nu există. Rulează train.py."
        )

@lru_cache(maxsize=1) # 🚨 FIX: Cache the massive JSON file in memory
def _load_physical_data() -> dict:
    json_path = os.path.join(
        _base_dir(), "..", "Date - meciuri", "players (1).json"
    )
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            p["wyId"]: {
                "height": p.get("height", 0),
                "weight": p.get("weight", 0),
            }
            for p in data["players"]
        }
    except Exception:
        return {}


# ─────────────────────────────────────────────
# HELPERS (DUPLICATES REMOVED)
# ─────────────────────────────────────────────

def _to_short(name: str) -> str:
    """🚨 FIX: Disabled name abbreviation to prevent mismatch bugs."""
    return name.strip()

def _find_player(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """🚨 FIX: Safe, case-insensitive match for the full name."""
    return df[df["name"].str.lower() == name.lower()]

def _get_player_id(df: pd.DataFrame, name: str):
    row = _find_player(df, name)
    if row.empty:
        return None
    return int(row.iloc[0]["playerId"])

def _extract_features(row: pd.Series, features: list) -> np.ndarray:
    return np.array([float(row.get(f, 0.0)) for f in features])


# ─────────────────────────────────────────────
# METODE DE CALCUL
# ─────────────────────────────────────────────

def _compute_fizic_gaussian(player_id_a, player_id_b, physical: dict):
    data_a = physical.get(player_id_a, {})
    data_b = physical.get(player_id_b, {})
    h_a, w_a = data_a.get("height", 0), data_a.get("weight", 0)
    h_b, w_b = data_b.get("height", 0), data_b.get("weight", 0)

    if h_a == 0 or h_b == 0 or w_a == 0 or w_b == 0:
        return None

    HEIGHT_RANGE = 40.0
    WEIGHT_RANGE = 45.0
    SIGMA = 0.3

    h_diff = abs(h_a - h_b) / HEIGHT_RANGE
    w_diff = abs(w_a - w_b) / WEIGHT_RANGE

    h_sim = np.exp(-(h_diff ** 2) / (2 * SIGMA ** 2))
    w_sim = np.exp(-(w_diff ** 2) / (2 * SIGMA ** 2))

    return round((h_sim + w_sim) / 2 * 100, 1)

def _compute_pearson(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    if len(vec_a) < 2 or np.std(vec_a) == 0 or np.std(vec_b) == 0:
        return 50.0  

    try:
        r, _ = pearsonr(vec_a, vec_b)
        score = (float(r) + 1) / 2 * 100
        return round(min(max(score, 0), 100), 1)
    except Exception:
        return 50.0

def _compute_weighted_euclidean(
    row_a: pd.Series, row_b: pd.Series, features: list, weights: dict
) -> float:
    total_weight = 0.0
    weighted_sq_diff = 0.0

    for feat in features:
        w = float(weights.get(feat, 1.0))
        val_a = float(row_a.get(feat, 0.0))
        val_b = float(row_b.get(feat, 0.0))
        weighted_sq_diff += w * (val_a - val_b) ** 2
        total_weight += w

    if total_weight == 0:
        return 0.0

    max_dist = np.sqrt(total_weight)
    dist = np.sqrt(weighted_sq_diff)
    
    # 🚨 FIX: Safe math clamping to prevent negative scores if un-normalized data slips in
    score = max(0.0, 1.0 - (dist / max_dist)) 
    return round(score * 100, 1)

# ─────────────────────────────────────────────
# RAPORT TEXT
# ─────────────────────────────────────────────

def _generate_report(
    name_a: str,
    name_b: str,
    row_a: pd.Series,
    row_b: pd.Series,
    position: str,
    scores: dict,
) -> dict:

    style_feats   = STYLE_FEATURES_PER_POSITION.get(position, [])
    quality_feats = QUALITY_FEATURES_PER_POSITION.get(position, [])
    perf_feats    = PERFORMANCE_FEATURES_PER_POSITION.get(position, [])

    similarities = []
    differences  = []

    for feat in [f for f in style_feats + quality_feats if f in row_a.index]:
        val_a = float(row_a.get(feat, 0))
        val_b = float(row_b.get(feat, 0))
        diff  = abs(val_a - val_b)
        label = _feature_to_label(feat)

        if diff < 0.12:
            level = "excelent" if max(val_a, val_b) > 0.7 else "similar"
            similarities.append((diff, f"✅ {label} — ambii la nivel {level}"))
        elif diff > 0.35:
            better = name_a if val_a > val_b else name_b
            differences.append((diff, f"❌ {label} — {better} este superior"))

    similarities.sort(key=lambda x: x[0])
    differences.sort(key=lambda x: x[0], reverse=True)

    # Performanță — afișat separat
    perf_comparison = []
    for feat in [f for f in perf_feats if f in row_a.index]:
        val_a = float(row_a.get(feat, 0))
        val_b = float(row_b.get(feat, 0))
        label = _feature_to_label(feat)
        better = name_a if val_a > val_b else name_b
        perf_comparison.append(
            f"📊 {label}: {name_a} {round(val_a, 2)} "
            f"vs {name_b} {round(val_b, 2)} — {better} produce mai mult"
        )

    # Fizic
    fizic = scores.get("fizic")
    if fizic is not None:
        if fizic >= 80:
            fizic_text = f"🏋️ Profil fizic aproape identic (scor: {fizic}/100)."
        elif fizic >= 60:
            fizic_text = f"🏋️ Profil fizic similar (scor: {fizic}/100)."
        else:
            fizic_text = f"🏋️ Diferențe fizice notabile (scor: {fizic}/100)."
    else:
        fizic_text = "🏋️ Date fizice indisponibile pentru unul sau ambii jucători."

    return {
        "similarities": [s[1] for s in similarities[:4]],
        "differences":  [d[1] for d in differences[:4]],
        "performance":  perf_comparison,
        "fizic":        fizic_text,
    }


# ─────────────────────────────────────────────
# FUNCȚIA PRINCIPALĂ
# ─────────────────────────────────────────────

def compute_similarity(name_a: str, name_b: str, position: str) -> dict:
    """
    Calculează similaritatea completă între doi jucători.

    Formula: 40% Fizic + 40% Stil + 20% Calitate

    Metode:
        Fizic    → Gaussian Similarity (înălțime + greutate)
        Stil     → Pearson Correlation (profilul de joc)
        Calitate → Weighted Euclidean (eficiență ponderată)

    Fără date fizice: 60% Stil + 40% Calitate
    """
    try:
        df = _load_normalized(position)
    except FileNotFoundError as e:
        return {"error": str(e)}

    rows_a = _find_player(df, name_a)
    rows_b = _find_player(df, name_b)

    if rows_a.empty:
        return {"error": f"'{name_a}' nu a fost găsit pentru poziția {position}."}
    if rows_b.empty:
        return {"error": f"'{name_b}' nu a fost găsit pentru poziția {position}."}

    row_a = rows_a.iloc[0]
    row_b = rows_b.iloc[0]

    # ── Fizic (40%) — Gaussian ──
    physical    = _load_physical_data()
    player_id_a = _get_player_id(df, name_a)
    player_id_b = _get_player_id(df, name_b)
    fizic_score = _compute_fizic_gaussian(player_id_a, player_id_b, physical)

    # ── Stil (40%) — Pearson Correlation ──
    style_feats = [
        f for f in STYLE_FEATURES_PER_POSITION.get(position, [])
        if f in df.columns
    ]
    stil_score = _compute_pearson(
        _extract_features(row_a, style_feats),
        _extract_features(row_b, style_feats),
    )

    # ── Calitate (20%) — Weighted Euclidean ──
    quality_feats = [
        f for f in QUALITY_FEATURES_PER_POSITION.get(position, [])
        if f in df.columns
    ]
    quality_weights = QUALITY_WEIGHTS_PER_POSITION.get(position, {})
    calitate_score = _compute_weighted_euclidean(
        row_a, row_b, quality_feats, quality_weights
    )

    # ── Scor Final ──
    if fizic_score is not None:
        final_score = round(
            0.40 * fizic_score + 0.40 * stil_score + 0.20 * calitate_score, 1
        )
    else:
        final_score = round(0.60 * stil_score + 0.40 * calitate_score, 1)

    # ── Spider Chart ──
    top_features = [
        f for f in TOP_FEATURES_PER_POSITION.get(position, [])
        if f in df.columns
    ]
    labels  = [_feature_to_label(f) for f in top_features]
    radar_a = {_feature_to_label(f): round(float(row_a.get(f, 0)), 3) for f in top_features}
    radar_b = {_feature_to_label(f): round(float(row_b.get(f, 0)), 3) for f in top_features}

    # ── Raport ──
    scores_dict = {
        "fizic":    fizic_score,
        "stil":     stil_score,
        "calitate": calitate_score,
        "final":    final_score,
    }
    report = _generate_report(name_a, name_b, row_a, row_b, position, scores_dict)

    return {
        "error":            None,
        "similarity_score": final_score,
        "scores": {
            "fizic":    fizic_score,
            "stil":     stil_score,
            "calitate": calitate_score,
        },
        "labels":         labels,
        "radar_values_a": radar_a,
        "radar_values_b": radar_b,
        "player_a":       name_a,
        "player_b":       name_b,
        "position":       position,
        "report":         report,
    }


# ─────────────────────────────────────────────
# UTILITARE UI
# ─────────────────────────────────────────────

def _feature_to_label(feature: str) -> str:
    label = feature.replace("avg_", "").replace("pct_", "% ")
    result = ""
    for char in label:
        if char.isupper():
            result += " " + char
        else:
            result += char
    return result.strip().title()


def _position_label(position: str) -> str:
    labels = {
        "fundas_central":    "fundași centrali",
        "fundas_lateral":    "fundași laterali",
        "mijlocas_defensiv": "mijlocași defensivi",
        "mijlocas_central":  "mijlocași centrali",
        "mijlocas_ofensiv":  "mijlocași ofensivi",
        "atacant_lateral":   "atacanți laterali",
        "atacant_central":   "atacanți centrali",
    }
    return labels.get(position, position)


def get_players_for_position(position: str) -> list:
    try:
        df = _load_normalized(position)
        return sorted(df["name"].dropna().unique().tolist())
    except FileNotFoundError:
        return []


def get_player_position_ml(player_name: str) -> str:
    saved_data_path = os.path.join(_base_dir(), "saved_data")
    short_name = _to_short(player_name)
    for filename in os.listdir(saved_data_path):
        if not filename.startswith("normalized_") or not filename.endswith(".pkl"):
            continue
        df = pd.read_pickle(os.path.join(saved_data_path, filename))
        if player_name in df["name"].values or short_name in df["name"].values:
            return filename.replace("normalized_", "").replace(".pkl", "")
    return ""


def get_player_position_by_id(player_id) -> str:
    """Găsește position_ml după Player ID — fără probleme cu diacritice."""
    saved_data_path = os.path.join(_base_dir(), "saved_data")
    try:
        pid = int(player_id)
    except (ValueError, TypeError):
        return ""
    for filename in os.listdir(saved_data_path):
        if not filename.startswith("normalized_") or not filename.endswith(".pkl"):
            continue
        df = pd.read_pickle(os.path.join(saved_data_path, filename))
        if pid in df["playerId"].values:
            return filename.replace("normalized_", "").replace(".pkl", "")
    return ""


def get_player_name_by_id(player_id) -> str:
    """Returnează shortName din pkl după Player ID."""
    saved_data_path = os.path.join(_base_dir(), "saved_data")
    try:
        pid = int(player_id)
    except (ValueError, TypeError):
        return ""
    for filename in os.listdir(saved_data_path):
        if not filename.startswith("normalized_") or not filename.endswith(".pkl"):
            continue
        df = pd.read_pickle(os.path.join(saved_data_path, filename))
        row = df[df["playerId"] == pid]
        if not row.empty:
            return str(row.iloc[0]["name"])
    return ""


def get_players_for_position_excluding(position: str, exclude_ids: list) -> list:
    """Returnează jucătorii pentru o poziție, excluzând ID-urile date."""
    try:
        df = _load_normalized(position)
        exclude = [int(x) for x in exclude_ids if str(x).strip().isdigit()]
        df_filtered = df[~df["playerId"].isin(exclude)]
        return sorted(df_filtered["name"].dropna().unique().tolist())
    except FileNotFoundError:
        return []