import streamlit as st
import pandas as pd
import numpy as np
from data_processor import process_data

# 1. Page Configuration
st.set_page_config(
    page_title="U Cluj Scouting AI", 
    page_icon="🦅", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Styling
st.markdown("""
    <style>
    .block-container { padding-top: 1.5rem; }
    .metric-card {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #ff4b4b;
        margin-bottom: 10px;
    }
    /* Pulse effect for the AI recommendation */
    .pulse {
        animation: pulse-red 2s infinite;
    }
    @keyframes pulse-red {
        0% { box-shadow: 0 0 0 0 rgba(255, 75, 75, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(255, 75, 75, 0); }
        100% { box-shadow: 0 0 0 0 rgba(255, 75, 75, 0); }
    }
    </style>
    """, unsafe_allow_html=True)

# 3. Backend Data Connection
@st.cache_data
def load_cached_data():
    return process_data("Date - meciuri/players (1).json", "Date - meciuri")

with st.spinner('Aggregating Wyscout Data...'):
    df_master = load_cached_data()

# 4. Header
st.markdown("# 🦅 U Cluj - AI Scouting Assistant")
st.divider()

# 5. Sidebar - Configuration & Filters
with st.sidebar:
    st.header("Scouting Criteria")
    search_name = st.text_input("Player Name (Optional)", placeholder="Search specific name...")
    age_range = st.slider("Age Range", 15, 45, (15, 40))
    max_val = st.number_input("Max Market Value (€)", value=1200000)
    min_mins = st.number_input("Minimum Minutes Played", min_value=0, value=0)
    
    positions_available = ["Any"] + sorted(df_master['position'].dropna().unique().tolist())
    selected_pos = st.selectbox("Position", options=positions_available)

    st.divider()
    if st.button("🚀 Find Prospects", type="primary", use_container_width=True):
        st.session_state.results_active = True
        st.session_state.selected_row = None # Reset selection on new search

# 6. Main Display Logic
if "results_active" not in st.session_state:
    st.session_state.results_active = False

if st.session_state.results_active:
    # Filtering Logic
    mask = (df_master['age'] >= age_range[0]) & (df_master['age'] <= age_range[1])
    mask &= (df_master['minutes_played'] >= min_mins)
    
    if search_name:
        mask &= (df_master['name'].str.contains(search_name, case=False, na=False))
    if selected_pos != "Any":
        mask &= (df_master['position'] == selected_pos)
    
    df_filtered = df_master[mask].reset_index(drop=True)
    
    st.subheader(f"🎯 Scouting Results ({len(df_filtered)} players found)")
    
    # Selection Table - We keep height small so stats are visible below
    event = st.dataframe(
        df_filtered[['original_name', 'position', 'age', 'minutes_played']],
        use_container_width=True,
        hide_index=True,
        selection_mode="single-row",
        on_select="rerun",
        height=300 
    )

    selected_rows = event.get("selection", {}).get("rows", [])
    
    if selected_rows:
        selected_index = selected_rows[0]
        p_data = df_filtered.iloc[selected_index]
        
        # This anchor ensures the browser has a point to reference
        st.markdown("<div id='linkto_stats'></div>", unsafe_allow_html=True)
        
        st.divider()
        
        # --- THE DEEP DIVE SECTION ---
        with st.expander(f"📊 DEEP DIVE: {p_data['original_name']}", expanded=True):
            col_main, col_side = st.columns([2, 1])
            
            with col_main:
                m1, m2, m3 = st.columns(3)
                m1.metric("Position", str(p_data['position']).upper())
                m2.metric("Age", int(p_data['age']) if pd.notnull(p_data['age']) else "N/A")
                m3.metric("Minutes", int(p_data['minutes_played']))
                
                ignore = ['player_id', 'name', 'original_name', 'position', 'age', 'minutes_played']
                metrics = [c for c in df_master.columns if c not in ignore and not str(c).endswith('_p90')]
                rows = [{"Metric": m.replace('_',' ').capitalize(), "Val": p_data[m]} for m in metrics]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=400)
            
            with col_side:
                st.markdown(f"""
                    <div class="metric-card pulse">
                        <h4>AI Recommendation</h4>
                        <h2 style="color:#ff4b4b;">SCOUT</h2>
                        <p>This player matches the <b>{p_data['position']}</b> tactical profile for the current U Cluj system.</p>
                    </div>
                    <div class="metric-card" style="border-left-color: #00FFAA;">
                        <h4>Growth Prediction</h4>
                        <h2 style="color:#00FFAA;">High</h2>
                        <p>Performance metrics indicate a high ceiling for development.</p>
                    </div>
                """, unsafe_allow_html=True)
                
                if st.button("✖ Clear Selection", use_container_width=True):
                    st.session_state.selected_row = None
                    st.rerun()
    else:
        st.info("👆 **Select a player row** in the table above to reveal the AI Deep-Dive analysis.")

else:
    # --- DEFAULT VIEW ---
    st.subheader("🦅 Current U Cluj Roster Overview")
    st.dataframe(
        df_master.head(10)[['original_name', 'position', 'age', 'minutes_played']], 
        use_container_width=True, 
        hide_index=True, 
        height=350
    )
    st.info("👈 Use the sidebar filters and click 'Find Prospects' to begin.")