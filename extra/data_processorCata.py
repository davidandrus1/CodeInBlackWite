import os
import json
import pandas as pd

def normalize_romanian(text):
    """Replaces Romanian diacritics with the standard alphabet for easy searching."""
    if not isinstance(text, str):
        return "Unknown"
    # Includes both standard comma-below and older cedilla versions
    mapping = {
        'ă': 'a', 'â': 'a', 'î': 'i', 'ș': 's', 'ț': 't',
        'Ă': 'A', 'Â': 'A', 'Î': 'I', 'Ș': 'S', 'Ț': 'T',
        'ş': 's', 'ţ': 't', 'Ş': 'S', 'Ţ': 'T' 
    }
    for k, v in mapping.items():
        text = text.replace(k, v)
    return text

def load_players(filepath):
    """Parses static player profiles and calculates age."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    players_data = data.get('players', []) if isinstance(data, dict) else data
    
    player_list = []
    for p in players_data:
        p_id = p.get('wyId', p.get('id', p.get('playerId')))
        
        # 1. Get the Full Name
        first_name = p.get('firstName', '')
        last_name = p.get('lastName', '')
        if first_name or last_name:
            raw_name = f"{first_name} {last_name}".strip()
        else:
            raw_name = p.get('name', p.get('shortName', 'Unknown'))
            
        # 2. Normalize for the search bar
        search_friendly_name = normalize_romanian(raw_name)
        
        role = p.get('role', {})
        position = role.get('code2', role.get('name', 'Unknown'))
        
        birth_date = p.get('birthDate')
        age = None
        if birth_date:
            try:
                birth_year = int(birth_date.split('-')[0])
                age = 2026 - birth_year # Base year for U Cluj Hackathon constraints
            except:
                pass
                
        player_list.append({
            'player_id': str(p_id),
            'name': search_friendly_name, # Used for the dropdown search
            'original_name': raw_name,    # Preserved for the beautiful UI display
            'position': position,
            'age': age
        })
        
    return pd.DataFrame(player_list)

def load_match_stats(directory_path):
    """Dynamically extracts ALL performance events for every player."""
    all_stats = []
    
    for filename in os.listdir(directory_path):
        if filename.endswith("_players_stats.json"):
            filepath = os.path.join(directory_path, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                match_players = data.get('players', []) if isinstance(data, dict) else data
                
                for player_stat in match_players:
                    p_id = str(player_stat.get('playerId', player_stat.get('id', '')))
                    stats = player_stat.get('total', player_stat)
                    
                    mins_played = stats.get('minutesOnField', 0)
                    if mins_played == 0:
                        continue 
                    
                    row = {
                        'player_id': p_id,
                        'minutes_played': mins_played
                    }
                    
                    for key, value in stats.items():
                        if isinstance(value, (int, float)) and key not in ['minutesOnField', 'id', 'matchId', 'playerId']:
                            row[key] = value
                            
                    all_stats.append(row)
            except Exception as e:
                print(f"Error parsing {filename}: {e}")
                
    return pd.DataFrame(all_stats)

def process_data(players_file, stats_dir):
    """Merges profiles and dynamically calculates p90 for every metric."""
    df_players = load_players(players_file)
    df_stats = load_match_stats(stats_dir)
    
    # Group by player_id and sum all numeric columns directly!
    df_agg_stats = df_stats.groupby('player_id').sum().reset_index()
    
    # Change 'inner' to 'left' merge so players with 0 minutes aren't deleted
    df_final = pd.merge(df_players, df_agg_stats, on='player_id', how='left')
    
    # Fill their missing minutes with 0
    df_final['minutes_played'] = df_final['minutes_played'].fillna(0)
    
    # Calculate p90 Metrics dynamically for all fields SAFELY
    metric_cols = [col for col in df_stats.columns if col != 'player_id']
    stats_to_convert = [col for col in metric_cols if col != 'minutes_played']
    
    for col in stats_to_convert:
        # Fill missing raw stats with 0 for the bench players
        df_final[col] = df_final[col].fillna(0)
        
        # Safe Math: Only calculate p90 if they actually played, otherwise set to 0.0
        df_final[f'{col}_p90'] = df_final.apply(
            lambda row: (row[col] / row['minutes_played'] * 90) if row['minutes_played'] > 0 else 0.0, 
            axis=1
        ).round(2)
        
    # ==========================================
    # 💰 TRANSFERMARKT & U CLUJ OVERRIDE INTEGRATION
    # ==========================================
    try:
        # 1. Load the Global Transfermarkt dataset (for the rest of the league)
        df_tm = pd.read_csv("players.csv", low_memory=False)
        df_tm = df_tm[['name', 'market_value_in_eur']].dropna()
        df_tm['match_name'] = df_tm['name'].apply(lambda x: normalize_romanian(str(x)).lower())
        df_final['match_name'] = df_final['name'].str.lower() 
        df_final = pd.merge(df_final, df_tm[['match_name', 'market_value_in_eur']], on='match_name', how='left')
        df_final = df_final.drop(columns=['match_name'])
    except FileNotFoundError:
        print("⚠️ players.csv not found. General market values will be empty.")
        df_final['market_value_in_eur'] = pd.NA

    try:
        # 2. THE HARD OVERRIDE: Inject accurate U Cluj Squad values
        df_ucluj = pd.read_csv("u_cluj_current_squad.csv")
        
        # Clean the "€1,500,000" strings into pure numbers
        def clean_currency(val):
            if pd.isna(val) or val == 'N/A':
                return pd.NA
            return float(str(val).replace('€', '').replace(',', ''))
            
        df_ucluj['exact_value'] = df_ucluj['Market Value (€)'].apply(clean_currency)
        df_ucluj['Player ID'] = df_ucluj['Player ID'].astype(str)
        
        # Merge exactly on Wyscout ID
        df_final = pd.merge(df_final, df_ucluj[['Player ID', 'exact_value']], left_on='player_id', right_on='Player ID', how='left')
        
        # If 'exact_value' exists, override whatever Kaggle gave us!
        df_final['market_value_in_eur'] = df_final['exact_value'].combine_first(df_final['market_value_in_eur'])
        df_final = df_final.drop(columns=['Player ID', 'exact_value'])
        
    except FileNotFoundError:
        print("⚠️ u_cluj_current_squad.csv not found. Could not apply squad overrides.")
        
    # Fill remaining missing values with 0 so the Streamlit UI formatting doesn't crash
    df_final['market_value_in_eur'] = df_final['market_value_in_eur'].fillna(0)
    
    # FIX FOR STREAMLIT CACHING
    for col in df_final.columns:
        if df_final[col].dtype == 'object':
            df_final[col] = df_final[col].astype(str)
            
    return df_final

if __name__ == "__main__":
    df_master = process_data("Date - meciuri/players (1).json", "Date - meciuri")
    print("\n✅ DATA PIPELINE COMPLETE!")