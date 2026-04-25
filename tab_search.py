import streamlit as st
import pandas as pd
import google.generativeai as genai
import json
from st_keyup import st_keyup  

def format_currency(val):
    try:
        v = float(val)
        return f"€{int(v):,}" if v > 0 else "Not Available"
    except: 
        return "Not Available"

def apply_scouting_filters(df, name_query, age_range, position, max_value, height_range, foot, min_minutes):
    filtered_df = df.copy()
    
    if name_query:
        filtered_df = filtered_df[filtered_df['original_name'].astype(str).str.contains(name_query, case=False, na=False)]
        if filtered_df.empty:
            return filtered_df

    filtered_df['safe_age'] = pd.to_numeric(filtered_df['age'], errors='coerce').fillna(99)
    filtered_df = filtered_df[(filtered_df['safe_age'] >= age_range[0]) & (filtered_df['safe_age'] <= age_range[1])]
    
    if 'market_value_in_eur' in filtered_df.columns:
        filtered_df['safe_value'] = pd.to_numeric(filtered_df['market_value_in_eur'], errors='coerce').fillna(0)
        filtered_df = filtered_df[filtered_df['safe_value'] <= max_value]
        
    if 'height' in filtered_df.columns:
        filtered_df['safe_height'] = pd.to_numeric(filtered_df['height'], errors='coerce').fillna(0)
        filtered_df = filtered_df[
            (filtered_df['safe_height'] == 0) | 
            ((filtered_df['safe_height'] >= height_range[0]) & (filtered_df['safe_height'] <= height_range[1]))
        ]

    if foot != "Any" and 'foot' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['foot'].astype(str).str.lower() == foot.lower()]
        
    if 'minutes_played' in filtered_df.columns:
        filtered_df['safe_mins'] = pd.to_numeric(filtered_df['minutes_played'], errors='coerce').fillna(0)
        filtered_df = filtered_df[filtered_df['safe_mins'] >= min_minutes]
        
    if position != "All":
        filtered_df = filtered_df[filtered_df['position'] == position]
        
    cols_to_drop = [col for col in ['safe_age', 'safe_value', 'safe_height', 'safe_mins'] if col in filtered_df.columns]
    return filtered_df.drop(columns=cols_to_drop)

def render_search_tab(df_master, u_cluj_names):
    if not u_cluj_names.empty and 'Player ID' in u_cluj_names.columns:
        u_cluj_ids = u_cluj_names['Player ID'].astype(str).tolist()
        scouting_pool = df_master[~df_master['player_id'].astype(str).isin(u_cluj_ids)]
    else:
        scouting_pool = df_master

    left_panel, right_panel = st.columns([1, 3], gap="large")

    with left_panel:
        st.subheader("Search Filters")
        api_key = st.text_input("Gemini API Key:", type="password")
        st.divider()
        
        default_search = st.session_state.get('search_target_name', '')
        
        search_name = st_keyup("Player Name:", value=default_search, debounce=300)
        
        if st.button("Clear Search", use_container_width=True):
            st.session_state.search_target_name = ""
            st.rerun()

        raw_positions = df_master['position'].dropna().unique().tolist()
        valid_positions = [pos for pos in raw_positions if str(pos).strip() != ""]
        valid_positions.insert(0, "All")
        selected_pos = st.selectbox("Position:", valid_positions)
        
        age_range = st.slider("Age Range:", min_value=16, max_value=40, value=(16, 40))
        max_value = st.number_input("Max Market Value (€):", min_value=0, value=2000000, step=100000)
        height_range = st.slider("Height (cm):", min_value=150, max_value=220, value=(150, 220), step=1)
        
        foot_options = ["Any", "Right", "Left", "Both"]
        selected_foot = st.selectbox("Preferred Foot:", foot_options, index=0)
        min_minutes = st.number_input("Min Minutes Played:", min_value=0, value=200, step=50)

        filtered_df = apply_scouting_filters(
            scouting_pool, search_name, age_range, selected_pos, max_value, height_range, selected_foot, min_minutes
        )

        if not filtered_df.empty and 'Growth_Potential' in filtered_df.columns:
            filtered_df = filtered_df.sort_values(by='Growth_Potential', ascending=False)

        st.divider()

        if 'ai_scout_report' not in st.session_state:
            st.session_state.ai_scout_report = ""
        if 'last_filtered_count' not in st.session_state:
            st.session_state.last_filtered_count = 0

        if st.session_state.last_filtered_count != len(filtered_df):
            st.session_state.ai_scout_report = ""
            st.session_state.last_filtered_count = len(filtered_df)

        if st.button("Generate AI Scout Report", type="primary", use_container_width=True):
            if not api_key:
                st.warning("Please provide a Gemini API Key.")
            else:
                safe_values = pd.to_numeric(filtered_df['market_value_in_eur'], errors='coerce').fillna(0)
                ai_pool = filtered_df[safe_values > 0]
                
                if len(ai_pool) < 3:
                    st.warning("Not enough players with known market values to generate a top 3 report.")
                else:
                    with st.spinner("Analyzing profiles..."):
                        top_candidates = ai_pool.head(5)
                        cols = ['original_name', 'age', 'position', 'foot', 'Growth_Potential', 'Performance_Index', 'market_value_in_eur']
                        candidates_data = top_candidates[[c for c in cols if c in top_candidates.columns]].to_dict(orient='records')
                        
                        genai.configure(api_key=api_key)
                        model = genai.GenerativeModel('gemini-2.5-flash')
                        
                        prompt = f"""Act as the Chief Scout for U Cluj. Review these candidate profiles: {json.dumps(candidates_data, ensure_ascii=False)}. 
                        Select exactly 3 players to recommend to the board and write a ruthless, concise scouting summary for each.
                        
                        Your selection criteria MUST be:
                        1. The 'Future Star': Select 1 player primarily for their high Growth Potential.
                        2 & 3. The 'Value Performers': Select 2 players by balancing their Performance Index against their Market Value.
                        
                        Format your response as 3 distinct bullet points. Be direct, analytical, and professional."""
                        
                        st.session_state.ai_scout_report = model.generate_content(prompt).text

        if st.session_state.ai_scout_report:
            st.info(st.session_state.ai_scout_report)
            
            if st.button("Clear Report"):
                st.session_state.ai_scout_report = ""
                st.rerun()

    with right_panel:
        if filtered_df.empty:
            st.warning("No players found matching these strict criteria.")
        else:
            st.subheader(f"Database Results ({len(filtered_df)})")
            
            display_df = filtered_df.copy()
            if 'market_value_in_eur' in display_df.columns:
                display_df['Market Value'] = display_df['market_value_in_eur'].apply(format_currency)
            
            display_cols = ['original_name', 'age', 'position', 'height', 'foot', 'Market Value', 'Growth_Potential']
            display_cols = [c for c in display_cols if c in display_df.columns]
            
            search_table = st.dataframe(
                display_df[display_cols],
                column_config={"original_name": "Player", "Market Value": "Value"},
                use_container_width=True, hide_index=True, selection_mode="single-row", on_select="rerun"
            )

            selected_row = search_table.get("selection", {}).get("rows", [])
            
            if selected_row or (search_name and len(filtered_df) == 1):
                idx = selected_row[0] if selected_row else 0
                selected_player = filtered_df.iloc[idx]
                
                st.markdown(f"### Deep Dive Tactical Stats: {selected_player['original_name']}")
                
                sc1, sc2 = st.columns(2)
                growth_val = selected_player.get('Growth_Potential', 0)
                sc1.metric("Growth Potential", f"{float(growth_val):.1f}")
                sc2.metric("Minutes Played", selected_player.get('minutes_played', 0))
                
                st.markdown("#### Role-Specific Key Metrics")
                pos = selected_player.get('position', '')
                
                pos_metrics = {
                    'GK': [
                        ("Saves p90", "gkSaves_p90", False), 
                        ("Save %", "pct_gkSaves", True), 
                        ("Exits p90", "gkExits_p90", False), 
                        ("Exit Success %", "pct_gkSuccessfulExits", True), 
                        ("Aerial Win %", "pct_gkAerialDuelsWon", True)
                    ],
                    'DF': [
                        ("Def Duels p90", "defensiveDuels_p90", False), 
                        ("Def Duel Win %", "pct_defensiveDuelsWon", True), 
                        ("Intercept p90", "interceptions_p90", False), 
                        ("Aerial Duels p90", "fieldAerialDuels_p90", False), 
                        ("Aerial Win %", "pct_fieldAerialDuelsWon", True), 
                        ("Prog Passes p90", "progressivePasses_p90", False)
                    ],
                    'MD': [
                        ("Intercept p90", "interceptions_p90", False), 
                        ("Recoveries p90", "recoveries_p90", False), 
                        ("Pass %", "pct_successfulPasses", True), 
                        ("Key Passes p90", "keyPasses_p90", False), 
                        ("Final 3rd Pass %", "pct_successfulPassesToFinalThird", True), 
                        ("Prog Runs p90", "progressiveRun_p90", False)
                    ],
                    'FW': [
                        ("xG p90", "xgShot_p90", False), 
                        ("Shots p90", "shots_p90", False), 
                        ("Touches in Box", "touchInBox_p90", False), 
                        ("Shot on Target %", "pct_shotsOnTarget", True), 
                        ("Goal Conversion %", "pct_goalConversion", True), 
                        ("Dribbles p90", "dribbles_p90", False)
                    ]
                }
                
                metrics_to_show = pos_metrics.get(pos, [])
                
                if metrics_to_show:
                    cols = st.columns(3) 
                    for i, (label, key, is_pct) in enumerate(metrics_to_show):
                        col = cols[i % 3]
                        val = selected_player.get(key, 0)
                        
                        if is_pct:
                            col.metric(label, f"{float(val):.1f}%")
                        else:
                            col.metric(label, f"{float(val):.2f}")
                else:
                    st.info(f"No specific tactical metrics mapped for position: {pos}")