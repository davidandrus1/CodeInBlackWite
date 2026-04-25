import os
import json
import pandas as pd
from models.feature_engineering import extract_position_for_ml, POSITION_MAPPING

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
        
        first_name = p.get('firstName', '')
        last_name = p.get('lastName', '')
        if first_name or last_name:
            raw_name = f"{first_name} {last_name}".strip()
        else:
            raw_name = p.get('name', p.get('shortName', 'Unknown'))
            
        search_friendly_name = normalize_romanian(raw_name)
        
        role = p.get('role', {})
        position = role.get('code2', role.get('name', 'Unknown'))
        
        birth_date = p.get('birthDate')
        age = None
        if birth_date:
            try:
                birth_year = int(birth_date.split('-')[0])
                age = 2026 - birth_year
            except:
                age = None
                
        player_list.append({
            'player_id': str(p_id),
            'name': search_friendly_name,
            'original_name': raw_name,
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
    """Merges profiles, calculates p90s, and adds Transfermarkt values."""
    df_players = load_players(players_file)
    df_stats = load_match_stats(stats_dir)
    
    # Group by player_id and sum all numeric columns directly!
    df_agg_stats = df_stats.groupby('player_id').sum().reset_index()
    
    # Merge with Player Profiles
    df_final = pd.merge(df_players, df_agg_stats, on='player_id', how='inner')
    
    # Filter out players with too few minutes
    df_final = df_final[df_final['minutes_played'] >= 200].copy()
    
    # Calculate p90 Metrics dynamically for all fields
    metric_cols = [col for col in df_stats.columns if col != 'player_id']
    stats_to_convert = [col for col in metric_cols if col != 'minutes_played']
    
    for col in stats_to_convert:
        df_final[f'{col}_p90'] = (df_final[col] / df_final['minutes_played']) * 90
        df_final[f'{col}_p90'] = df_final[f'{col}_p90'].round(2)
        
    # ==========================================
    # 💰 NEW: TRANSFERMARKT INTEGRATION
    # ==========================================
    try:
        # Load the Transfermarkt dataset
        df_tm = pd.read_csv("players.csv", low_memory=False)
        
        # Keep only what we need to save memory
        df_tm = df_tm[['name', 'market_value_in_eur']].dropna()
        
        # Create a matching column by normalizing and lowercasing both datasets
        # This fixes issues where Wyscout says "Andrei Șut" and TM says "Andrei Sut"
        df_tm['match_name'] = df_tm['name'].apply(lambda x: normalize_romanian(str(x)).lower())
        df_final['match_name'] = df_final['name'].str.lower() 
        
        # Merge the datasets!
        df_final = pd.merge(df_final, df_tm[['match_name', 'market_value_in_eur']], on='match_name', how='left')
        
        # Drop the temporary match_name column
        df_final = df_final.drop(columns=['match_name'])
        
        # If a player wasn't found in Transfermarkt, give them a default value of €250k
        df_final['market_value_in_eur'] = df_final['market_value_in_eur'].fillna(250000)
        
    except FileNotFoundError:
        print("⚠️ players.csv not found. Using default €250k market values.")
        df_final['market_value_in_eur'] = 250000
    
    # FIX FOR STREAMLIT CACHING
    for col in df_final.columns:
        if df_final[col].dtype == 'object':
            df_final[col] = df_final[col].astype(str)
            
    return df_final

if __name__ == "__main__":
    df_master = process_data("Date - meciuri/players (1).json", "Date - meciuri")
    print("\n✅ DATA PIPELINE COMPLETE!")