import pandas as pd

POSITION_MAPPING = {
    "gk":   "portar",
    "cb":   "fundas_central",
    "rcb":  "fundas_central_dreapta",
    "lcb":  "fundas_central_stanga",
    "rb":   "fundas_lateral_dreapta",
    "lb":   "fundas_lateral_stanga",
    "rwb":  "fundas_lateral_dreapta",
    "rcb3": "fundas_central_dreapta",  
    "lcb3": "fundas_central_stanga", 
    "lwb":  "fundas_lateral_stanga",
    "rb5":  "fundas_lateral_dreapta",  
    "lb5":  "fundas_lateral_stanga",
    "dmf":  "mijlocas_defensiv",
    "rdmf": "mijlocas_defensiv_dreapta",
    "ldmf": "mijlocas_defensiv_stanga",
    "cmf":  "mijlocas_central",
    "rcmf": "mijlocas_central_dreapta",
    "lcmf": "mijlocas_central_stanga",
    "amf":  "mijlocas_ofensiv",
    "ramf": "mijlocas_ofensiv_dreapta",
    "lamf": "mijlocas_ofensiv_stanga",
    "rw":   "atacant_lateral_dreapta",
    "lw":   "atacant_lateral_stanga",
    "rwf":  "atacant_lateral_dreapta",
    "lwf":  "atacant_lateral_stanga",
    "cf":   "atacant_central",
    "ss":   "atacant_central",
}

def extract_primary_position(positions: list) -> str:
    """Poziție completă cu lateralitate — pentru UI."""
    if not positions:
        return "unknown"

    sorted_positions = sorted(positions, key=lambda x: x["percent"], reverse=True)
    primary_code = sorted_positions[0]["position"]["code"].lower()

    return POSITION_MAPPING.get(primary_code, "unknown")


def extract_position_for_ml(positions: list) -> str:
    """Poziție grupată fără lateralitate — pentru ML."""
    full_position = extract_primary_position(positions)

    return (full_position
            .replace("_stanga", "")
            .replace("_dreapta", "")
            )


FEATURES_PER_POSITION = {

    "portar": [
        "avg_gkSaves",
        "pct_gkSaves",
        "avg_gkConcededGoals",       # invers — mai mic e mai bine
        "avg_gkExits",
        "pct_gkSuccessfulExits",
        "avg_gkAerialDuels",
        "pct_gkAerialDuelsWon",
        "avg_goalKicksShort",
        "avg_goalKicksLong",
        "pct_successfulGoalKicks",
    ],

    "fundas_central": [
        "avg_aerialDuels",
        "pct_fieldAerialDuelsWon",
        "avg_defensiveDuels",
        "pct_defensiveDuelsWon",
        "avg_interceptions",
        "avg_clearances",
        "avg_shotsBlocked",
        "avg_progressivePasses",
        "pct_successfulProgressivePasses",
        "avg_dangerousOwnHalfLosses",   # invers
    ],

    "fundas_lateral": [
        "avg_crosses",
        "pct_successfulCrosses",
        "avg_dribbles",
        "pct_successfulDribbles",
        "avg_defensiveDuels",
        "pct_defensiveDuelsWon",
        "avg_interceptions",
        "avg_progressiveRun",
        "pct_newDuelsWon",
        "avg_ballRecoveries",
        "avg_opponentHalfRecoveries",
    ],

    "mijlocas_defensiv": [
        "avg_interceptions",
        "avg_ballRecoveries",
        "avg_counterpressingRecoveries",
        "avg_defensiveDuels",
        "pct_defensiveDuelsWon",
        "avg_aerialDuels",
        "pct_aerialDuelsWon",
        "avg_passes",
        "pct_successfulPasses",
        "avg_progressivePasses",
        "pct_successfulProgressivePasses",
        "avg_dangerousOwnHalfLosses",   # invers
    ],

   "mijlocas_central": [
    "avg_passes",
    "pct_successfulPasses",
    "avg_keyPasses",
    "pct_successfulKeyPasses",
    "avg_progressivePasses",
    "pct_successfulProgressivePasses",
    "avg_interceptions",
    "avg_ballRecoveries",
    "avg_counterpressingRecoveries",
    "pct_newDuelsWon",
    "avg_dribbles",
    "pct_successfulDribbles",
    ],

    "mijlocas_ofensiv": [
    "avg_keyPasses",
    "pct_successfulKeyPasses",
    "avg_xgAssist",
    "avg_shotAssists",
    "avg_dribbles",
    "pct_successfulDribbles",
    "avg_progressiveRun",
    "avg_shots",
    "avg_shotsOnTarget",
    "avg_xgShot",
    "avg_touchInBox",
    "avg_passesToFinalThird",
    "pct_successfulPassesToFinalThird",
        ],

    "atacant_lateral": [
        "avg_dribbles",
        "pct_successfulDribbles",
        "avg_accelerations",
        "avg_progressiveRun",
        "avg_crosses",
        "pct_successfulCrosses",
        "avg_shots",
        "avg_shotsOnTarget",
        "avg_xgShot",
        "avg_xgAssist",
        "avg_goals",
        "avg_assists",
        "avg_opponentHalfRecoveries",
    ],

    "atacant_central": [
        "avg_goals",
        "avg_xgShot",
        "pct_goalConversion",
        "avg_shots",
        "avg_shotsOnTarget",
        "pct_shotsOnTarget",
        "avg_headShots",
        "avg_aerialDuels",
        "pct_fieldAerialDuelsWon",
        "avg_linkupPlays",
        "pct_successfulLinkupPlays",
        "avg_touchInBox",
        "avg_dribbles",
        "avg_xgAssist",
    ],
}


def get_features_for_position(df: pd.DataFrame, position: str) -> pd.DataFrame:
    """
    Returnează DataFrame-ul cu doar coloanele relevante
    pentru poziția specificată.
    """
    if position not in FEATURES_PER_POSITION:
        raise ValueError(f"Poziție necunoscută: {position}. "
                         f"Alege din: {list(FEATURES_PER_POSITION.keys())}")

    features = FEATURES_PER_POSITION[position]

    # Păstrează și ID-ul
    cols = ["playerId"] + features

    # Verifică dacă toate coloanele există
    missing = [c for c in features if c not in df.columns]
    if missing:
        print(f"[WARN] Coloane lipsă pentru {position}: {missing}")
        cols = ["playerId"] + [c for c in features if c in df.columns]

    return df[cols].copy()


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Înlocuiește valorile lipsă cu 0.
    (Dacă un jucător nu are date la o metrică, înseamnă că nu a făcut acea acțiune)
    """
    return df.fillna(0)