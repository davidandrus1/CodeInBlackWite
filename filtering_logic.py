import pandas as pd

def apply_scouting_filters(df, name_query, age_range, position, max_value, height_range, foot, min_minutes):
    """
    Filters the master dataframe based on user inputs.
    """
    filtered_df = df.copy()
    
    # 1. Age Filter
    filtered_df['safe_age'] = pd.to_numeric(filtered_df['age'], errors='coerce').fillna(99)
    filtered_df = filtered_df[(filtered_df['safe_age'] >= age_range[0]) & (filtered_df['safe_age'] <= age_range[1])]
    
    # 2. Maximum Market Value Filter
    if 'market_value_in_eur' in filtered_df.columns:
        filtered_df['safe_value'] = pd.to_numeric(filtered_df['market_value_in_eur'], errors='coerce').fillna(0)
        filtered_df = filtered_df[filtered_df['safe_value'] <= max_value]
        
    # 3. Height Filter (Assuming height is stored in meters, e.g., 1.85. If in cm, adjust logic)
    if 'height' in filtered_df.columns:
        filtered_df['safe_height'] = pd.to_numeric(filtered_df['height'], errors='coerce').fillna(0)
        # We only filter if the player has a height listed to avoid dropping players with missing data
        # Alternatively, change this to strictly drop players outside the range.
        filtered_df = filtered_df[
            (filtered_df['safe_height'] == 0) | 
            ((filtered_df['safe_height'] >= height_range[0]) & (filtered_df['safe_height'] <= height_range[1]))
        ]

    # 4. Preferred Foot Filter
    if foot != "Any" and 'foot' in filtered_df.columns:
        # Case insensitive match
        filtered_df = filtered_df[filtered_df['foot'].astype(str).str.lower() == foot.lower()]
        
    # 5. Min Minutes Played Filter
    if 'minutes_played' in filtered_df.columns:
        filtered_df['safe_mins'] = pd.to_numeric(filtered_df['minutes_played'], errors='coerce').fillna(0)
        filtered_df = filtered_df[filtered_df['safe_mins'] >= min_minutes]
        
    # 6. Position Filter
    if position != "Any Position":
        filtered_df = filtered_df[filtered_df['position'] == position]
        
    # 7. Name Filter
    if name_query:
        filtered_df = filtered_df[filtered_df['original_name'].astype(str).str.contains(name_query, case=False, na=False)]

    # Clean up temporary columns used for safe filtering
    cols_to_drop = [col for col in ['safe_age', 'safe_value', 'safe_height', 'safe_mins'] if col in filtered_df.columns]
    filtered_df = filtered_df.drop(columns=cols_to_drop)

    return filtered_df