import pandas as pd

# ─────────────────────────────────────────────
# POSITION MAPPINGS
# ─────────────────────────────────────────────

POSITION_MAPPING = {
    "gk":   "portar",
    "cb":   "fundas_central",
    "rcb":  "fundas_central",
    "lcb":  "fundas_central",
    "rb":   "fundas_lateral",
    "lb":   "fundas_lateral",
    "rwb":  "fundas_lateral",
    "lcb3": "fundas_central",
    "rcb3": "fundas_central",
    "lwb":  "fundas_lateral",
    "rb5":  "fundas_lateral",
    "lb5":  "fundas_lateral",
    "dmf":  "mijlocas_defensiv",
    "rdmf": "mijlocas_defensiv",
    "ldmf": "mijlocas_defensiv",
    "cmf":  "mijlocas_central",
    "rcmf": "mijlocas_central",
    "lcmf": "mijlocas_central",
    "amf":  "mijlocas_ofensiv",
    "ramf": "mijlocas_ofensiv",
    "lamf": "mijlocas_ofensiv",
    "rw":   "atacant_lateral",
    "lw":   "atacant_lateral",
    "rwf":  "atacant_lateral",
    "lwf":  "atacant_lateral",
    "cf":   "atacant_central",
    "ss":   "atacant_central",
}


def extract_primary_position(positions: list) -> str:
    if not positions:
        return "unknown"
    sorted_positions = sorted(positions, key=lambda x: x["percent"], reverse=True)
    primary_code = sorted_positions[0]["position"]["code"].lower()
    return POSITION_MAPPING.get(primary_code, "unknown")


def extract_position_for_ml(positions: list) -> str:
    return extract_primary_position(positions)


# ─────────────────────────────────────────────
# FEATURES PER POSITION — pentru antrenare
# Include TOT ce avem nevoie pentru Stil +
# Calitate + Performanță + Spider Chart
# ─────────────────────────────────────────────

FEATURES_PER_POSITION = {

    "portar": [
        # Stil
        "avg_gkSaves", "avg_gkExits",
        "avg_goalKicksLong", "avg_goalKicksShort",
        "avg_interceptions",
        # Calitate
        "pct_gkSaves", "pct_gkSuccessfulExits",
        "avg_xgSave", "pct_successfulGoalKicks",
        "pct_gkAerialDuelsWon", "pct_successfulPasses",
        # Performanță (separat)
        "avg_gkConcededGoals",
    ],

    "fundas_central": [
        # Stil
        "avg_interceptions", "avg_clearances",
        "avg_progressivePasses", "avg_shotsBlocked",
        "avg_defensiveDuels",
        # Calitate
        "pct_defensiveDuelsWon", "pct_fieldAerialDuelsWon",
        "pct_successfulLongPasses", "pct_successfulProgressivePasses",
        # Performanță (separat)
        "avg_dangerousOwnHalfLosses", "avg_aerialDuels",
    ],

    "fundas_lateral": [
        # Stil
        "avg_defensiveDuels", "avg_crosses",
        "avg_progressiveRun", "avg_interceptions",
        "avg_ballRecoveries", "avg_dribbles", "avg_accelerations",
        # Calitate
        "pct_successfulCrosses", "pct_successfulPassesToFinalThird",
        "pct_successfulDribbles", "pct_defensiveDuelsWon",
        # Performanță (separat)
        "avg_assists", "avg_opponentHalfRecoveries",
    ],

    "mijlocas_defensiv": [
        # Stil
        "avg_interceptions", "avg_ballRecoveries",
        "avg_counterpressingRecoveries", "avg_progressivePasses",
        "avg_verticalPasses", "avg_looseBallDuels",
        # Calitate
        "pct_defensiveDuelsWon", "pct_successfulPasses",
        "pct_successfulLongPasses",
        # Performanță (separat)
        "avg_passes", "avg_aerialDuels",
    ],

    "mijlocas_central": [
        # Stil
        "avg_keyPasses", "avg_smartPasses", "avg_shotAssists",
        "avg_dribbles", "avg_touchInBox",
        "avg_interceptions", "avg_ballRecoveries",
        # Calitate
        "pct_successfulDribbles", "pct_successfulPassesToFinalThird",
        "avg_xgAssist", "pct_offensiveDuelsWon",
        "pct_successfulKeyPasses",
        # Performanță (separat)
        "avg_goals", "avg_assists", "avg_passes",
    ],

    "mijlocas_ofensiv": [
        # Stil
        "avg_keyPasses", "avg_smartPasses", "avg_shotAssists",
        "avg_dribbles", "avg_touchInBox", "avg_progressiveRun",
        # Calitate
        "pct_successfulDribbles", "pct_successfulPassesToFinalThird",
        "avg_xgAssist", "avg_xgShot", "pct_successfulKeyPasses",
        # Performanță (separat)
        "avg_goals", "avg_assists",
    ],

    "atacant_lateral": [
        # Stil
        "avg_dribbles", "avg_crosses", "avg_progressiveRun",
        "avg_accelerations", "avg_opponentHalfRecoveries", "avg_shots",
        # Calitate
        "pct_successfulDribbles", "pct_successfulCrosses",
        "avg_xgShot", "avg_xgAssist", "pct_successfulForwardPasses",
        # Performanță (separat)
        "avg_goals", "avg_assists",
    ],

    "atacant_central": [
        # Stil
        "avg_shots", "avg_touchInBox", "avg_offensiveDuels",
        "avg_aerialDuels", "avg_linkupPlays",
        # Calitate
        "pct_shotsOnTarget", "avg_xgShot", "pct_goalConversion",
        "pct_fieldAerialDuelsWon", "pct_successfulLinkupPlays",
        # Performanță (separat)
        "avg_goals",
    ],
}


# ─────────────────────────────────────────────
# STYLE FEATURES — CUM joacă
# Volume, tendințe, comportament pe teren
# ─────────────────────────────────────────────

STYLE_FEATURES_PER_POSITION = {

    "portar": [
        "avg_gkSaves", "avg_gkExits",
        "avg_goalKicksLong", "avg_goalKicksShort",
        "avg_interceptions",
    ],

    "fundas_central": [
        "avg_interceptions", "avg_clearances",
        "avg_progressivePasses", "avg_shotsBlocked",
        "avg_defensiveDuels",
    ],

    "fundas_lateral": [
        "avg_defensiveDuels", "avg_crosses",
        "avg_progressiveRun", "avg_interceptions",
        "avg_ballRecoveries", "avg_dribbles", "avg_accelerations",
    ],

    "mijlocas_defensiv": [
        "avg_interceptions", "avg_ballRecoveries",
        "avg_counterpressingRecoveries", "avg_progressivePasses",
        "avg_verticalPasses", "avg_looseBallDuels",
    ],

    "mijlocas_central": [
        "avg_keyPasses", "avg_smartPasses", "avg_shotAssists",
        "avg_dribbles", "avg_touchInBox",
        "avg_interceptions", "avg_ballRecoveries",
    ],

    "mijlocas_ofensiv": [
        "avg_keyPasses", "avg_smartPasses", "avg_shotAssists",
        "avg_dribbles", "avg_touchInBox", "avg_progressiveRun",
    ],

    "atacant_lateral": [
        "avg_dribbles", "avg_crosses", "avg_progressiveRun",
        "avg_accelerations", "avg_opponentHalfRecoveries", "avg_shots",
    ],

    "atacant_central": [
        "avg_shots", "avg_touchInBox", "avg_offensiveDuels",
        "avg_aerialDuels", "avg_linkupPlays",
    ],
}


# ─────────────────────────────────────────────
# QUALITY FEATURES — CÂT DE BINE face ce face
# Eficiență, precizie, metrici xG
# Intră în formula de similaritate (20%)
# ─────────────────────────────────────────────

QUALITY_FEATURES_PER_POSITION = {

    "portar": [
        "pct_gkSaves", "pct_gkSuccessfulExits",
        "avg_xgSave", "pct_successfulGoalKicks",
        "pct_gkAerialDuelsWon", "pct_successfulPasses",
    ],

    "fundas_central": [
        "pct_defensiveDuelsWon", "pct_fieldAerialDuelsWon",
        "pct_successfulLongPasses", "pct_successfulProgressivePasses",
    ],

    "fundas_lateral": [
        "pct_successfulCrosses", "pct_successfulPassesToFinalThird",
        "pct_successfulDribbles", "pct_defensiveDuelsWon",
    ],

    "mijlocas_defensiv": [
        "pct_defensiveDuelsWon", "pct_successfulPasses",
        "pct_successfulLongPasses",
    ],

    "mijlocas_central": [
        "pct_successfulDribbles", "pct_successfulPassesToFinalThird",
        "avg_xgAssist", "pct_offensiveDuelsWon",
        "pct_successfulKeyPasses",
    ],

    "mijlocas_ofensiv": [
        "pct_successfulDribbles", "pct_successfulPassesToFinalThird",
        "avg_xgAssist", "avg_xgShot", "pct_successfulKeyPasses",
    ],

    "atacant_lateral": [
        "pct_successfulDribbles", "pct_successfulCrosses",
        "avg_xgShot", "avg_xgAssist", "pct_successfulForwardPasses",
    ],

    "atacant_central": [
        "pct_shotsOnTarget", "avg_xgShot", "pct_goalConversion",
        "pct_fieldAerialDuelsWon", "pct_successfulLinkupPlays",
    ],
}


# ─────────────────────────────────────────────
# PERFORMANCE FEATURES — CÂT produce
# Afișat SEPARAT în raport, NU în scorul
# de similaritate
# ─────────────────────────────────────────────

PERFORMANCE_FEATURES_PER_POSITION = {

    "portar": [
        "avg_gkConcededGoals",
    ],

    "fundas_central": [
        "avg_dangerousOwnHalfLosses", "avg_aerialDuels",
    ],

    "fundas_lateral": [
        "avg_assists", "avg_opponentHalfRecoveries",
    ],

    "mijlocas_defensiv": [
        "avg_passes", "avg_aerialDuels",
    ],

    "mijlocas_central": [
        "avg_goals", "avg_assists", "avg_passes",
    ],

    "mijlocas_ofensiv": [
        "avg_goals", "avg_assists",
    ],

    "atacant_lateral": [
        "avg_goals", "avg_assists",
    ],

    "atacant_central": [
        "avg_goals",
    ],
}


# ─────────────────────────────────────────────
# QUALITY WEIGHTS — importanța fiecărui metric
# de calitate per poziție (pentru Weighted Euclidean)
# ─────────────────────────────────────────────

QUALITY_WEIGHTS_PER_POSITION = {

    "portar": {
        "pct_gkSaves":            3,
        "pct_gkSuccessfulExits":  2,
        "avg_xgSave":             3,
        "pct_successfulGoalKicks": 1,
        "pct_gkAerialDuelsWon":   2,
        "pct_successfulPasses":   1,
    },

    "fundas_central": {
        "pct_defensiveDuelsWon":          3,
        "pct_fieldAerialDuelsWon":        2,
        "pct_successfulLongPasses":       2,
        "pct_successfulProgressivePasses": 2,
    },

    "fundas_lateral": {
        "pct_successfulCrosses":         3,
        "pct_successfulPassesToFinalThird": 2,
        "pct_successfulDribbles":        2,
        "pct_defensiveDuelsWon":         3,
    },

    "mijlocas_defensiv": {
        "pct_defensiveDuelsWon":  3,
        "pct_successfulPasses":   3,
        "pct_successfulLongPasses": 2,
    },

    "mijlocas_central": {
        "pct_successfulDribbles":          2,
        "pct_successfulPassesToFinalThird": 3,
        "avg_xgAssist":                    3,
        "pct_offensiveDuelsWon":           1,
        "pct_successfulKeyPasses":         3,
    },

    "mijlocas_ofensiv": {
        "pct_successfulDribbles":          2,
        "pct_successfulPassesToFinalThird": 2,
        "avg_xgAssist":                    3,
        "avg_xgShot":                      3,
        "pct_successfulKeyPasses":         3,
    },

    "atacant_lateral": {
        "pct_successfulDribbles":    3,
        "pct_successfulCrosses":     2,
        "avg_xgShot":                3,
        "avg_xgAssist":              3,
        "pct_successfulForwardPasses": 1,
    },

    "atacant_central": {
        "pct_shotsOnTarget":        2,
        "avg_xgShot":               3,
        "pct_goalConversion":       3,
        "pct_fieldAerialDuelsWon":  2,
        "pct_successfulLinkupPlays": 1,
    },
}


# ─────────────────────────────────────────────
# SURPRISE FEATURES — cross-poziționale
# Normalizate față de TOȚI jucătorii de câmp
# Afișat SEPARAT în raport
# ─────────────────────────────────────────────

SURPRISE_FEATURES = [
    "avg_progressiveRun",
    "avg_smartPasses",
    "avg_interceptions",
    "avg_dribbles",
    "avg_counterpressingRecoveries",
    "avg_touchInBox",
    "avg_xgAssist",
    "avg_shotAssists",
    "avg_aerialDuels",
    "avg_keyPasses",
]

FIELD_POSITIONS = [
    "fundas_central", "fundas_lateral",
    "mijlocas_defensiv", "mijlocas_central", "mijlocas_ofensiv",
    "atacant_lateral", "atacant_central",
]


# ─────────────────────────────────────────────
# TOP FEATURES — Spider Chart (6-8 axe)
# ─────────────────────────────────────────────

TOP_FEATURES_PER_POSITION = {

    "portar": [
        "avg_gkSaves", "pct_gkSaves",
        "avg_gkExits", "pct_gkSuccessfulExits",
        "pct_gkAerialDuelsWon", "pct_successfulGoalKicks",
        "avg_goalKicksLong", "avg_goalKicksShort",
    ],

    "fundas_central": [
        "avg_defensiveDuels", "pct_defensiveDuelsWon",
        "avg_interceptions", "avg_aerialDuels",
        "pct_fieldAerialDuelsWon", "avg_progressivePasses",
        "pct_successfulProgressivePasses",
    ],

    "fundas_lateral": [
        "avg_crosses", "pct_successfulCrosses",
        "avg_progressiveRun", "avg_defensiveDuels",
        "pct_defensiveDuelsWon", "avg_interceptions",
        "pct_successfulPassesToFinalThird",
    ],

    "mijlocas_defensiv": [
        "avg_interceptions", "avg_ballRecoveries",
        "avg_counterpressingRecoveries", "pct_defensiveDuelsWon",
        "avg_progressivePasses", "pct_successfulPasses",
        "avg_verticalPasses",
    ],

    "mijlocas_central": [
        "avg_keyPasses", "avg_smartPasses",
        "avg_touchInBox", "avg_interceptions",
        "pct_successfulKeyPasses", "avg_xgAssist",
        "pct_successfulPassesToFinalThird",
    ],

    "mijlocas_ofensiv": [
        "avg_keyPasses", "avg_smartPasses",
        "avg_xgAssist", "avg_xgShot",
        "avg_touchInBox", "avg_dribbles",
        "pct_successfulPassesToFinalThird", "avg_shotAssists",
    ],

    "atacant_lateral": [
        "avg_dribbles", "pct_successfulDribbles",
        "avg_progressiveRun", "avg_xgShot",
        "avg_xgAssist", "avg_crosses",
        "avg_opponentHalfRecoveries",
    ],

    "atacant_central": [
        "avg_shots", "avg_xgShot",
        "avg_touchInBox", "avg_aerialDuels",
        "pct_shotsOnTarget", "pct_goalConversion",
        "avg_linkupPlays", "pct_fieldAerialDuelsWon",
    ],
}


# ─────────────────────────────────────────────
# UTILITY FUNCTIONS
# ─────────────────────────────────────────────

def get_features_for_position(df: pd.DataFrame, position: str) -> pd.DataFrame:
    if position not in FEATURES_PER_POSITION:
        raise ValueError(f"Poziție necunoscută: {position}.")
    features = FEATURES_PER_POSITION[position]
    cols = ["playerId"] + features
    missing = [c for c in features if c not in df.columns]
    if missing:
        print(f"[WARN] Coloane lipsă pentru {position}: {missing}")
        cols = ["playerId"] + [c for c in features if c in df.columns]
    return df[cols].copy()


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    return df.fillna(0)