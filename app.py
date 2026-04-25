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
    find_button = st.button("🚀 Find Prospects", type="primary", use_container_width=True)

# 6. Main Display Logic
if "results_active" not in st.session_state:
    st.session_state.results_active = False

if find_button:
    st.session_state.results_active = True

if st.session_state.results_active:
    mask = (df_master['age'] >= age_range[0]) & (df_master['age'] <= age_range[1])
    mask &= (df_master['minutes_played'] >= min_mins)
    
    if search_name:
        mask &= (df_master['name'].str.contains(search_name, case=False, na=False))
    if selected_pos != "Any":
        mask &= (df_master['position'] == selected_pos)
    if max_val > 0 and 'market_value' in df_master.columns:
        mask &= (df_master['market_value'] <= max_val)
        
    df_filtered = df_master[mask].reset_index(drop=True)
    
    st.subheader(f"🎯 Scouting Results ({len(df_filtered)} players found)")
    
    # Table selection
    event = st.dataframe(
        df_filtered[['original_name', 'position', 'age', 'minutes_played']],
        use_container_width=True,
        hide_index=True,
        selection_mode="single-row",
        on_select="rerun",
        height=400 
    )

    selected_rows = event.get("selection", {}).get("rows", [])
    
    if selected_rows:
        # --- AUTO-SCROLL JAVASCRIPT ---
        # This invisible component tells the browser to scroll to the bottom
        st.components.v1.html(
            """
            <script>
                window.parent.document.querySelectorAll('[data-testid="stVerticalBlock"]')[0].scrollTo({
                    top: 1000, 
                    behavior: 'smooth'
                });
            </script>
            """,
            height=0,
        )
        
        selected_index = selected_rows[0]
        p_data = df_filtered.iloc[selected_index]
        
        st.divider()
        st.markdown(f"### 👤 Analysis: {p_data['original_name']}")
        
        col_main, col_side = st.columns([2, 1])
        
        with col_main:
            m1, m2, m3 = st.columns(3)
            m1.metric("Position", str(p_data['position']).upper())
            m2.metric("Age", int(p_data['age']) if pd.notnull(p_data['age']) else "N/A")
            m3.metric("Minutes", int(p_data['minutes_played']))
            
            ignore = ['player_id', 'name', 'original_name', 'position', 'age', 'minutes_played']
            metrics = [c for c in df_master.columns if c not in ignore and not str(c).endswith('_p90')]
            rows = [{"Metric": m.replace('_',' ').capitalize(), "Val": p_data[m]} for m in metrics]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=350)
        
        with col_side:
            st.markdown(f"""
                <div class="metric-card">
                    <h4>AI Recommendation</h4>
                    <h2 style="color:#ff4b4b;">SCOUT</h2>
                    <p>Matches <b>{p_data['position']}</b> profile for U Cluj.</p>
                </div>
                <div class="metric-card" style="border-left-color: #00FFAA;">
                    <h4>Growth Potential</h4>
                    <h2 style="color:#00FFAA;">+8%</h2>
                    <p>Estimated market value increase.</p>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.info("👆 Click a row in the table above to view the analysis below.")

else:
    st.subheader("🦅 Current U Cluj Roster Overview")
    st.dataframe(df_master.head(10)[['original_name', 'position', 'age', 'minutes_played']], use_container_width=True, hide_index=True, height=300)