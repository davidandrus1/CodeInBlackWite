import streamlit as st
import pandas as pd
import google.generativeai as genai
import json

def apply_scouting_filters(df, name_query, age_range, position, max_value, height_range, foot, min_minutes):
    """Filters database. Features a 'Bypass' if a specific name is clicked/searched."""
    filtered_df = df.copy()
    
    # 🚨 FILTER BYPASS: If a specific name is searched, show them no matter what!
    if name_query:
        name_match = filtered_df[filtered_df['original_name'].astype(str).str.contains(name_query, case=False, na=False)]
        if not name_match.empty:
            return name_match # Return the player immediately, ignoring the sliders!

    # Normal Numerical Filters (Only apply if no name is searched)
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

def render_search_tab(df_master):
    left_panel, right_panel = st.columns([1, 3], gap="large")

    with left_panel:
        st.subheader("⚙️ Search Filters")
        api_key = st.text_input("Gemini API:", type="password")
        st.divider()
        
        # Pull from session state to auto-fill when redirected from a Card!
        default_search = st.session_state.get('search_target_name', '')
        search_name = st.text_input("Player Name:", value=default_search)
        
        if st.button("Clear Search", use_container_width=True):
            st.session_state.search_target_name = ""
            st.rerun()

        valid_positions = df_master['position'].dropna().unique().tolist()
        valid_positions.insert(0, "All")
        selected_pos = st.selectbox("Position:", valid_positions)
        
        age_range = st.slider("Age Range:", min_value=16, max_value=40, value=(16, 40))
        max_value = st.number_input("Max Market Value (€):", min_value=0, value=2000000, step=100000)
        height_range = st.slider("Height (cm):", min_value=150, max_value=220, value=(150, 220), step=1)
        
        foot_options = ["Any", "Right", "Left", "Both"]
        selected_foot = st.selectbox("Preferred Foot:", foot_options, index=0)
        min_minutes = st.number_input("Min Minutes Played:", min_value=0, value=200, step=50)

    with right_panel:
        filtered_df = apply_scouting_filters(
            df_master, search_name, age_range, selected_pos, max_value, height_range, selected_foot, min_minutes
        )

        if filtered_df.empty:
            st.warning("No players found matching these strict criteria.")
        else:
            if 'Growth_Potential' in filtered_df.columns:
                filtered_df = filtered_df.sort_values(by='Growth_Potential', ascending=False)
            
            # --- AI SCOUTING REPORT ---
            if st.button("🤖 Generate AI Scout Report for Top 3", type="primary"):
                if not api_key:
                    st.warning("⚠️ Please provide a Gemini API Key.")
                else:
                    with st.spinner("Analyzing physical and tactical profiles..."):
                        top_candidates = filtered_df.head(3)
                        cols = ['original_name', 'age', 'position', 'foot', 'Growth_Potential', 'market_value_in_eur']
                        candidates_data = top_candidates[[c for c in cols if c in top_candidates.columns]].to_dict(orient='records')
                        
                        genai.configure(api_key=api_key)
                        model = genai.GenerativeModel('gemini-2.5-flash')
                        prompt = f"""Act as the Chief Scout for U Cluj. Analyze these candidates: {json.dumps(candidates_data)}. 
                        Write a ruthless 3-sentence tactical summary identifying the best target based on Growth_Potential, physical traits, and Value."""
                        st.info(model.generate_content(prompt).text)

            st.divider()

            # --- INTERACTIVE TABLE ---
            st.subheader(f"📋 Database Results ({len(filtered_df)})")
            
            display_cols = ['original_name', 'age', 'position', 'height', 'foot', 'market_value_in_eur', 'Growth_Potential']
            display_cols = [c for c in display_cols if c in filtered_df.columns]
            
            search_table = st.dataframe(
                filtered_df[display_cols],
                column_config={"original_name": "Player", "market_value_in_eur": st.column_config.NumberColumn("Value", format="€%d")},
                use_container_width=True, hide_index=True, selection_mode="single-row", on_select="rerun"
            )

            # --- DEEP DIVE TACTICAL STATS ---
            selected_row = search_table.get("selection", {}).get("rows", [])
            
            # Auto-open stats if redirected or clicked
            if selected_row or (search_name and len(filtered_df) == 1):
                idx = selected_row[0] if selected_row else 0
                selected_player = filtered_df.iloc[idx]
                
                st.markdown(f"### 🔎 Deep Dive Tactical Stats: {selected_player['original_name']}")
                sc1, sc2, sc3, sc4 = st.columns(4)
                
                growth_val = selected_player.get('Growth_Potential', 0)
                sc1.metric("⭐ Growth Potential", f"{float(growth_val):.1f}")
                sc2.metric("⏱️ Minutes Played", selected_player.get('minutes_played', 0))
                
                pos = selected_player.get('position', '')
                if pos == 'FW':
                    sc3.metric("🎯 Shots p90", selected_player.get('shots_p90', 0))
                    sc4.metric("⚽ xG p90", selected_player.get('xgShot_p90', 0))
                elif pos == 'MD':
                    sc3.metric("👟 Pass %", f"{selected_player.get('pct_successfulPasses', 0):.1f}%")
                    sc4.metric("⏩ Prog. Passes p90", selected_player.get('progressivePasses_p90', 0))
                else:
                    sc3.metric("🛡️ Def Duels p90", selected_player.get('defensiveDuels_p90', 0))
                    sc4.metric("✂️ Intercept p90", selected_player.get('interceptions_p90', 0))