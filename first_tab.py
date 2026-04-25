import streamlit as st
import pandas as pd

def render_first_tab(df_master, u_cluj_names):
    # Process U Cluj Roster
    u_cluj_ids = u_cluj_names['Player ID'].astype(str).tolist()
    roster_df = df_master[df_master['player_id'].astype(str).isin(u_cluj_ids)].copy()

    # ==========================================
    # 🎯 CUSTOM TACTICAL SORTING (FW -> MD -> DF -> GK)
    # ==========================================
    position_order = ['FW', 'MD', 'DF', 'GK']
    roster_df['position_cat'] = pd.Categorical(roster_df['position'], categories=position_order, ordered=True)
    roster_df = roster_df.sort_values(by=['position_cat', 'original_name'])
    
    st.subheader(f"U Cluj Roster ({len(roster_df)} Players Matched)")
    
    # Display the roster table
    event = st.dataframe(
        roster_df[['original_name', 'age', 'position', 'market_value_in_eur']], 
        column_config={
            "original_name": "Name",
            "age": "Age",
            "position": "Position",
            "market_value_in_eur": st.column_config.NumberColumn("Value", format="€%d")
        },
        use_container_width=True,
        hide_index=True,
        selection_mode="single-row",
        on_select="rerun"
    )

    # --- Selection Analysis ---
    selected_rows = event.get("selection", {}).get("rows", [])

    if selected_rows:
        selected_idx = selected_rows[0]
        target = roster_df.iloc[selected_idx]
        
        st.divider()
        st.header(f"Replacement Analysis: {target['original_name']}")
        
        # Exclude current roster
        replacements_pool = df_master[~df_master['original_name'].isin(u_cluj_names['Player ID'])]
        
        u21_df = replacements_pool[
            (replacements_pool['position'] == target['position']) & (replacements_pool['age'].astype(float) <= 21)
        ].sort_values('minutes_played', ascending=False).head(1)
        
        liga_df = replacements_pool[
            (replacements_pool['position'] == target['position'])
        ].sort_values('market_value_in_eur', ascending=False).head(5)

        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.markdown("### 👤 Selected")
            st.metric("Name", target['original_name'])
            # 🚨 FIX: Convert to float first, then int!
            st.metric("Value", f"€{int(float(target['market_value_in_eur'])):,}")

        with c2:
            st.markdown("### ✨ Top U21 Prospect")
            if not u21_df.empty:
                p = u21_df.iloc[0]
                st.markdown(f"""<div class="scout-card"><h3>{p['original_name']}</h3>
                    <p>Age: {int(float(p['age']))} | Mins: {int(float(p['minutes_played']))}</p>
                    <div class="value">€{int(float(p['market_value_in_eur'])):,}</div></div>""", unsafe_allow_html=True)

        with c3:
            st.markdown("### 🏟️ Liga Replacement 1")
            if not liga_df.empty:
                r = liga_df.iloc[0]
                st.markdown(f"""<div class="scout-card"><h3>{r['original_name']}</h3>
                    <p>Age: {int(float(r['age']))} | Mins: {int(float(r['minutes_played']))}</p>
                    <div class="value">€{int(float(r['market_value_in_eur'])):,}</div></div>""", unsafe_allow_html=True)

        st.markdown("### 📋 Additional Options")
        grid = st.columns(4)
        for i, (_, row) in enumerate(liga_df.iloc[1:5].iterrows()):
            with grid[i]:
                st.markdown(f"""<div class="scout-card"><h3>{row['original_name']}</h3>
                    <p>Age: {int(float(row['age']))}</p><div class="value">€{int(float(row['market_value_in_eur'])):,}</div></div>""", unsafe_allow_html=True)
    else:
        st.info("Select a player from the roster to start analysis.")