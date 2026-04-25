from data_loader import load_all_matches, aggregate_players
from feature_engineering import get_features_for_position, handle_missing_values, FEATURES_PER_POSITION
from normalization import normalize_position

# 1. Încarci toate meciurile
df_matches = load_all_matches("../../Date - meciuri/")

# 2. Agregezi per jucător
df_players = aggregate_players(df_matches, min_minutes=45)

# 3. Pentru fiecare poziție: extragi, curăți, normalizezi
for pozitie in FEATURES_PER_POSITION.keys():
    
    df_pozitie = df_players[df_players["position_ml"] == pozitie].copy()
    
    if len(df_pozitie) == 0:
        print(f"[SKIP] Niciun jucător pentru {pozitie}")
        continue
    
    df_pozitie = get_features_for_position(df_pozitie, pozitie)
    df_pozitie = handle_missing_values(df_pozitie)
    
    features = [c for c in df_pozitie.columns if c != "playerId"]
    
    normalize_position(df_pozitie, pozitie, features)
    
    print(f"[OK] {pozitie}: {len(df_pozitie)} jucători procesați")