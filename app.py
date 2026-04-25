import streamlit as st
import pandas as pd
import numpy as np
from data_processor import process_data

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="U Cluj Scouting AI", 
    page_icon="🦅", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. Unified Styling (Matches your Wireframe) ---
st.markdown("""
    <style>
    .block-container { padding-top: 1.5rem; }
    .scout-card {
        background-color: #0e1117;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #00FFAA;
        text-align: center;
        height: 220px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .scout-card h3 { color: #00FFAA; font-size: 1.3rem; margin-bottom: 8px; }
    .scout-card p { color: #ffffff; margin: 2px 0; font-size: 0.9rem; }
    .scout-card .value { color: #ffffff; font-size: 1.1rem; font-weight: bold; margin-top: 12px; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. Data Loading ---
@st.cache_data
def load_all_data():
    # 1. Load the performance database from JSON
    df_perf = process_data("Date - meciuri/players (1).json", "Date - meciuri")
    
    # 2. Load your specific U Cluj Squad file
    # This file has one column: 'Player'
    u_cluj_list = pd.read_csv("u_cluj_current_squad.csv")
    
    return df_perf, u_cluj_list

with st.spinner('Loading Roster...'):
    df_master, u_cluj_names = load_all_data()

# Filter df_master to only include the names found in your CSV
# We use .isin() to match the 'Player' column from your CSV to 'original_name' in JSON

u_cluj_ids = u_cluj_names['Player ID'].astype(str).tolist()

# 2. Filter df_master using the correct column 'player_id', also forced to string
roster_df = df_master[df_master['player_id'].astype(str).isin(u_cluj_ids)].copy()

# --- 4. Main Interface ---
st.title("🦅 U Cluj - Roster & Replacement Finder")

tab1, tab2 = st.tabs(["📋 CURRENT SQUAD", "🔍 SEARCH DATABASE"])

with tab1:
    st.subheader(f"U Cluj Roster ({len(roster_df)} Players Matched)")
    
    # Display the roster table
    # Users can click a row to trigger the "Replacement Analysis"
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

# --- 5. Selection Analysis (The Wireframe Section) ---
selected_rows = event.get("selection", {}).get("rows", [])

if selected_rows:
    selected_idx = selected_rows[0]
    target = roster_df.iloc[selected_idx]
    
    st.divider()
    st.header(f"Replacement Analysis: {target['original_name']}")
    
    # Define replacements pool (Exclude U Cluj players so we find new signings)
    replacements_pool = df_master[~df_master['original_name'].isin(u_cluj_names['Player ID'])]
    
    # Logic for cards
    u21_df = replacements_pool[
        (replacements_pool['position'] == target['position']) & (replacements_pool['age'] <= 21)
    ].sort_values('minutes_played', ascending=False).head(1)
    
    liga_df = replacements_pool[
        (replacements_pool['position'] == target['position'])
    ].sort_values('market_value_in_eur', ascending=False).head(5)

    # Rendering the Layout
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown("### 👤 Selected")
        st.metric("Name", target['original_name'])
        st.metric("Value", f"€{int(target['market_value_in_eur']):,}")

    with c2:
        st.markdown("### ✨ Top U21 Prospect")
        if not u21_df.empty:
            p = u21_df.iloc[0]
            st.markdown(f"""<div class="scout-card"><h3>{p['original_name']}</h3>
                <p>Age: {int(p['age'])} | Mins: {int(p['minutes_played'])}</p>
                <div class="value">€{int(p['market_value_in_eur']):,}</div></div>""", unsafe_allow_html=True)

    with c3:
        st.markdown("### 🏟️ Liga Replacement 1")
        if not liga_df.empty:
            r = liga_df.iloc[0]
            st.markdown(f"""<div class="scout-card"><h3>{r['original_name']}</h3>
                <p>Age: {int(r['age'])} | Mins: {int(r['minutes_played'])}</p>
                <div class="value">€{int(r['market_value_in_eur']):,}</div></div>""", unsafe_allow_html=True)

    # Additional grid
    st.markdown("### 📋 Additional Options")
    grid = st.columns(4)
    for i, (_, row) in enumerate(liga_df.iloc[1:].iterrows()):
        with grid[i]:
            st.markdown(f"""<div class="scout-card"><h3>{row['original_name']}</h3>
                <p>Age: {int(row['age'])}</p><div class="value">€{int(row['market_value_in_eur']):,}</div></div>""", unsafe_allow_html=True)
else:
    st.info("Select a player from the roster to start analysis.")