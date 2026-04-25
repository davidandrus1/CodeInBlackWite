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

# --- 2. Unified Styling (Wireframe Aesthetic) ---
st.markdown("""
    <style>
    .block-container { padding-top: 1.5rem; }
    
    /* The Green Border Card Style from your U21 screenshot */
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
        transition: transform 0.2s;
    }
    .scout-card:hover {
        background-color: #161a24;
        transform: scale(1.02);
    }
    .scout-card h3 {
        color: #00FFAA;
        font-size: 1.4rem;
        margin-bottom: 10px;
        font-weight: 600;
    }
    .scout-card p {
        color: #ffffff;
        margin: 2px 0;
        font-size: 0.95rem;
    }
    .scout-card .value {
        color: #ffffff;
        font-size: 1.2rem;
        font-weight: bold;
        margin-top: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. Data Loading ---
@st.cache_data
def load_cached_data():
    # Note: Ensure paths match your local directory structure
    return process_data("Date - meciuri/players (1).json", "Date - meciuri")

with st.spinner('Accessing U Cluj Database...'):
    df_master = load_cached_data()

# --- 4. Header & Tabs ---
st.title("🦅 U Cluj - Roster & Replacement Finder")

tab1, tab2 = st.tabs(["📋 ROSTER OVERVIEW", "🔍 SEARCH DATABASE"])

with tab1:
    st.subheader("Current Squad")
    roster_display_cols = ['original_name', 'age', 'position', 'market_value_in_eur']
    
    # Selection trigger
    roster_event = st.dataframe(
        df_master.head(15), 
        column_config={
            "original_name": "Name",
            "age": "Age",
            "position": "Position",
            "market_value_in_eur": st.column_config.NumberColumn("Market Value", format="€%d")
        },
        use_container_width=True,
        hide_index=True,
        selection_mode="single-row",
        on_select="rerun"
    )

with tab2:
    st.info("Global Search and advanced filters can be implemented here.")

# --- 5. Analysis Section (Triggered by Selection) ---
selection = roster_event.get("selection", {}).get("rows", [])

if selection:
    selected_index = selection[0]
    target = df_master.iloc[selected_index]
    
    st.divider()
    st.header(f"Replacement Analysis: {target['original_name']}")
    
    # Filter Logic
    u21_prospects = df_master[
        (df_master['position'] == target['position']) & 
        (df_master['age'] <= 21) & 
        (df_master['original_name'] != target['original_name'])
    ].sort_values(by='minutes_played', ascending=False)

    liga_replacements = df_master[
        (df_master['position'] == target['position']) & 
        (df_master['original_name'] != target['original_name'])
    ].sort_values(by='market_value_in_eur', ascending=False).head(5)

    # --- ROW 1: Selected Player + Top U21 + Top Liga ---
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown("### 👤 Selected Player")
        st.metric("Name", target['original_name'])
        st.write(f"**Pos:** {target['position']} | **Age:** {target['age']}")
        st.metric("Current Value", f"€{int(target['market_value_in_eur']):,}")

    with c2:
        st.markdown("### ✨ Top U21 Prospect")
        if not u21_prospects.empty:
            u21 = u21_prospects.iloc[0]
            st.markdown(f"""
                <div class="scout-card">
                    <h3>{u21['original_name']}</h3>
                    <p>Age: {int(u21['age'])}</p>
                    <p>Mins: {int(u21['minutes_played'])}</p>
                    <div class="value">€{int(u21['market_value_in_eur']):,}</div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("No U21 prospects found.")

    with c3:
        st.markdown("### 🏟️ Top Superliga Option")
        if not liga_replacements.empty:
            rep1 = liga_replacements.iloc[0]
            st.markdown(f"""
                <div class="scout-card">
                    <h3>{rep1['original_name']}</h3>
                    <p>Age: {int(rep1['age'])}</p>
                    <p>Mins: {int(rep1['minutes_played'])}</p>
                    <div class="value">€{int(rep1['market_value_in_eur']):,}</div>
                </div>
            """, unsafe_allow_html=True)

    # --- ROW 2: Additional Market Options (All using the same card wireframe) ---
    st.markdown("### 📋 Additional Superliga Options")
    grid_cols = st.columns(4)
    
    # Taking the next 4 replacements to fill the grid
    other_reps = liga_replacements.iloc[1:5]
    
    for i, (idx, row) in enumerate(other_reps.iterrows()):
        with grid_cols[i]:
            st.markdown(f"""
                <div class="scout-card">
                    <h3 style="font-size:1.1rem;">{row['original_name']}</h3>
                    <p>Age: {int(row['age'])}</p>
                    <p>Mins: {int(row['minutes_played'])}</p>
                    <div class="value" style="font-size:1rem;">€{int(row['market_value_in_eur']):,}</div>
                </div>
            """, unsafe_allow_html=True)

else:
    st.info("💡 Select a player from the roster above to generate the replacement wireframe.")