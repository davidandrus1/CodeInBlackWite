import os
import json
import pandas as pd
import numpy as np
import difflib

def normalize_romanian(text):
    if not isinstance(text, str): return "Unknown"
    mapping = {'ă': 'a', 'â': 'a', 'î': 'i', 'ș': 's', 'ț': 't', 'Ă': 'A', 'Â': 'A', 'Î': 'I', 'Ș': 'S', 'Ț': 'T', 'ş': 's', 'ţ': 't', 'Ş': 'S', 'Ţ': 'T'}
    for k, v in mapping.items(): text = text.replace(k, v)
    return text

def load_players(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    players_data = data.get('players', []) if isinstance(data, dict) else data
    
    player_list = []
    for p in players_data:
        p_id = p.get('wyId', p.get('id', p.get('playerId')))
        raw_name = f"{p.get('firstName', '')} {p.get('lastName', '')}".strip() or p.get('name', p.get('shortName', 'Unknown'))
        role = p.get('role', {})
        position = role.get('code2') or role.get('name') or ""
        position = position.strip()
        birth_date = p.get('birthDate')
        age = 25
        if birth_date:
            try:
                age = 2026 - int(str(birth_date).split('-')[0])
            except (ValueError, IndexError):
                age = 25 
                
        player_list.append({
            'player_id': str(p_id),
            'name': normalize_romanian(raw_name), 
            'original_name': raw_name,    
            'position': position,
            'age': age,
            'foot': str(p.get('foot') or 'unknown').capitalize(),
            'height': p.get('height', 0),
            'weight': p.get('weight', 0)
        })
        
    return pd.DataFrame(player_list)

def load_match_stats(directory_path):
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
                    if mins_played == 0: continue 
                    
                    row = {'player_id': p_id, 'minutes_played': mins_played}
                    for key, value in stats.items():
                        if isinstance(value, (int, float)) and key not in ['minutesOnField', 'id', 'matchId', 'playerId']:
                            row[key] = value
                    all_stats.append(row)
            except Exception as e: 
                pass
    return pd.DataFrame(all_stats)

def add_percentage_metrics(df):
    def safe_pct(num, den):
        return (num / den * 100) if pd.notnull(num) and pd.notnull(den) and den > 0 else 0.0

    df['pct_gkSaves'] = df.apply(lambda r: safe_pct(r.get('gkSaves'), r.get('gkShotsAgainst')), axis=1)
    df['pct_gkSuccessfulExits'] = df.apply(lambda r: safe_pct(r.get('gkSuccessfulExits'), r.get('gkExits')), axis=1)
    df['pct_gkAerialDuelsWon'] = df.apply(lambda r: safe_pct(r.get('gkAerialDuelsWon'), r.get('gkAerialDuels')), axis=1)
    df['pct_successfulGoalKicks'] = df.apply(lambda r: safe_pct(r.get('successfulGoalKicks'), r.get('goalKicks')), axis=1)
    df['pct_defensiveDuelsWon'] = df.apply(lambda r: safe_pct(r.get('defensiveDuelsWon'), r.get('defensiveDuels')), axis=1)
    df['pct_fieldAerialDuelsWon'] = df.apply(lambda r: safe_pct(r.get('fieldAerialDuelsWon'), r.get('fieldAerialDuels')), axis=1)
    df['pct_successfulProgressivePasses'] = df.apply(lambda r: safe_pct(r.get('successfulProgressivePasses'), r.get('progressivePasses')), axis=1)
    df['pct_successfulCrosses'] = df.apply(lambda r: safe_pct(r.get('successfulCrosses'), r.get('crosses')), axis=1)
    df['pct_successfulPassesToFinalThird'] = df.apply(lambda r: safe_pct(r.get('successfulPassesToFinalThird'), r.get('passesToFinalThird')), axis=1)
    df['pct_successfulPasses'] = df.apply(lambda r: safe_pct(r.get('successfulPasses'), r.get('passes')), axis=1)
    df['pct_successfulKeyPasses'] = df.apply(lambda r: safe_pct(r.get('successfulKeyPasses'), r.get('keyPasses')), axis=1)
    df['pct_successfulDribbles'] = df.apply(lambda r: safe_pct(r.get('successfulDribbles'), r.get('dribbles')), axis=1)
    df['pct_shotsOnTarget'] = df.apply(lambda r: safe_pct(r.get('shotsOnTarget'), r.get('shots')), axis=1)
    df['pct_goalConversion'] = df.apply(lambda r: safe_pct(r.get('goals'), r.get('shots')), axis=1)
    
    return df

def calculate_growth_potential(df):
    df['safe_age'] = pd.to_numeric(df['age'], errors='coerce').fillna(25)
    
    def get_age_score(age):
        if age <= 21: return 1.0     
        elif age <= 24: return 0.95  
        elif age <= 28: return 0.90  
        elif age <= 32: return 0.75  
        else: return 0.40            
        
    df['Age_Score'] = df['safe_age'].apply(get_age_score)
    
    pos_metrics = {
        'GK': ["gkSaves_p90", "pct_gkSaves", "gkExits_p90", "pct_gkSuccessfulExits", "pct_gkAerialDuelsWon"],
        'DF': ["defensiveDuels_p90", "pct_defensiveDuelsWon", "interceptions_p90", "fieldAerialDuels_p90", "pct_fieldAerialDuelsWon", "progressivePasses_p90"],
        'MD': ["interceptions_p90", "recoveries_p90", "pct_successfulPasses", "keyPasses_p90", "pct_successfulPassesToFinalThird", "progressiveRun_p90"],
        'FW': ["xgShot_p90", "shots_p90", "touchInBox_p90", "pct_shotsOnTarget", "pct_goalConversion", "dribbles_p90"]
    }
    
    df['Performance_Index'] = 0.0
    for pos, metrics in pos_metrics.items():
        pos_mask = (df['position'] == pos) & (df['minutes_played'] >= 200)
        if not pos_mask.any(): continue
        
        pos_df = df[pos_mask]
        player_ratios = pd.Series(0.0, index=df.index)
        valid_metrics_count = 0
        
        for metric in metrics:
            if metric in df.columns:
                metric_avg = pos_df[metric].mean()
                if metric_avg > 0:
                    ratio = (df[metric] / metric_avg).clip(upper=2.0)
                    player_ratios += ratio
                    valid_metrics_count += 1
                    
        if valid_metrics_count > 0:
            final_pos_score = (player_ratios / valid_metrics_count) * 50
            df.loc[df['position'] == pos, 'Performance_Index'] = final_pos_score.loc[df['position'] == pos]
            
    df['Growth_Potential'] = (df['Performance_Index'] * df['Age_Score']).clip(lower=0, upper=100).round(1)
    return df

def process_data(players_file, stats_dir):
    df_players = load_players(players_file)
    df_stats = load_match_stats(stats_dir)
    df_agg_stats = df_stats.groupby('player_id').sum().reset_index()
    
    df_final = pd.merge(df_players, df_agg_stats, on='player_id', how='left')
    df_final['minutes_played'] = df_final['minutes_played'].fillna(0)
    
    df_final = add_percentage_metrics(df_final)
    
    metric_cols = [col for col in df_stats.columns if col != 'player_id']
    stats_to_convert = [col for col in metric_cols if col != 'minutes_played']
    for col in stats_to_convert:
        df_final[col] = df_final[col].fillna(0)
        df_final[f'{col}_p90'] = df_final.apply(lambda row: (row[col] / row['minutes_played'] * 90) if row['minutes_played'] > 0 else 0.0, axis=1).round(2)
        
    # ==========================================
    # 🚨 DYNAMIC PATHING ENGINE (ADDED HERE)
    # ==========================================
    BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.dirname(BACKEND_DIR)
    DATA_DIR = os.path.join(ROOT_DIR, "Data")  # 🚨 Capital 'D' for your folder!
    
    try:
        # 🚨 Tell Pandas to look inside the Data folder
        players_csv_path = os.path.join(DATA_DIR, "players.csv")
        df_tm = pd.read_csv(players_csv_path, low_memory=False)
        df_tm = df_tm[['name', 'market_value_in_eur']].dropna()
        
        def sanitize_name(name_str):
            clean = normalize_romanian(str(name_str)).lower()
            clean = clean.replace('-', ' ')
            clean = ' '.join(clean.split())
            return clean

        df_tm['match_name'] = df_tm['name'].apply(sanitize_name)
        df_tm = df_tm.drop_duplicates(subset=['match_name'], keep='first')
        
        df_final['match_name'] = df_final['name'].apply(sanitize_name)
        
        tm_valid_names = df_tm['match_name'].tolist()
        
        def find_best_match(name):
            if name in tm_valid_names:
                return name
            
            matches = difflib.get_close_matches(name, tm_valid_names, n=1, cutoff=0.8)
            if matches:
                return matches[0]
                
            return name
            
        df_final['smart_match_name'] = df_final['match_name'].apply(find_best_match)
        df_final = pd.merge(df_final, df_tm[['match_name', 'market_value_in_eur']], left_on='smart_match_name', right_on='match_name', how='left')
        df_final = df_final.drop(columns=['match_name_x', 'match_name_y', 'smart_match_name'], errors='ignore')
        
    except Exception as e:
        df_final['market_value_in_eur'] = pd.NA

    try:
        # 🚨 Tell Pandas to look inside the Data folder
        ucluj_csv_path = os.path.join(DATA_DIR, "u_cluj_current_squad.csv")
        df_ucluj = pd.read_csv(ucluj_csv_path)
        def clean_currency(val):
            if pd.isna(val) or val == 'N/A': return pd.NA
            return float(str(val).replace('€', '').replace(',', ''))
        df_ucluj['exact_value'] = df_ucluj['Market Value (€)'].apply(clean_currency)
        df_ucluj['Player ID'] = df_ucluj['Player ID'].astype(str)
        df_final = pd.merge(df_final, df_ucluj[['Player ID', 'exact_value']], left_on='player_id', right_on='Player ID', how='left')
        df_final['market_value_in_eur'] = df_final['exact_value'].combine_first(df_final['market_value_in_eur'])
        df_final = df_final.drop(columns=['Player ID', 'exact_value'])
    except: pass
        
    df_final['market_value_in_eur'] = df_final['market_value_in_eur'].fillna(0)
    df_final = calculate_growth_potential(df_final)
    
    for col in df_final.columns:
        if df_final[col].dtype == 'object': df_final[col] = df_final[col].astype(str)
            
    return df_final