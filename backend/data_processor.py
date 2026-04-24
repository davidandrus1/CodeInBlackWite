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
    
    metric_cols = [col for col in df_stats.columns if col != 'player_id']
    agg_funcs = {col: 'sum' for col in metric_cols}
    df_agg_stats = df_stats.groupby('player_id').agg(agg_funcs).reset_index()
    
    df_final = pd.merge(df_players, df_agg_stats, on='player_id', how='inner')
    df_final = df_final[df_final['minutes_played'] >= 200].copy()
    
    stats_to_convert = [col for col in metric_cols if col != 'minutes_played']
    for col in stats_to_convert:
        df_final[f'{col}_p90'] = (df_final[col] / df_final['minutes_played']) * 90
        df_final[f'{col}_p90'] = df_final[f'{col}_p90'].round(2)
        
    return df_final

if __name__ == "__main__":
    df_master = process_data("Date - meciuri/players (1).json", "Date - meciuri")
    print("\n✅ DATA PIPELINE COMPLETE!")