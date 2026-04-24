import streamlit as st
import pandas as pd
from backend.data_processor import process_data

st.set_page_config(page_title="U Cluj Scouting AI", page_icon="🦅", layout="wide")

@st.cache_data
def load_cached_data():
    return process_data("Date - meciuri/players (1).json", "Date - meciuri")

with st.spinner('Aggregating All Wyscout Data...'):
    df_master = load_cached_data()

st.title("🦅 Universitatea Cluj - AI Scouting Tool")

# The dropdown will now use the clean names (e.g., "Andrei Sut") so standard typing works
player_names = df_master['name'].dropna().sort_values().unique().tolist()
selected_player = st.selectbox("🔍 Search Player Name:", options=["-- Select a Player --"] + player_names)

if selected_player != "-- Select a Player --":
    st.divider()
    player_data = df_master[df_master['name'] == selected_player].iloc[0]
    
    # --- TOP PROFILE HEADER ---
    # We use 'original_name' here so it displays beautifully with diacritics (e.g., "Andrei Șut")
    st.header(f"👤 {player_data['original_name']}") 
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Age", int(player_data['age']) if pd.notnull(player_data['age']) else "N/A")
    col2.metric("Position", str(player_data['position']).upper())
    col3.metric("Total Season Minutes", int(player_data['minutes_played']))
    
    st.divider()
    st.subheader("📊 Complete Performance Data")
    
    # --- DYNAMIC STATS TABLE ---
    # Added 'original_name' to the ignore list so it doesn't try to plot it as a stat
    ignore_cols = ['player_id', 'name', 'original_name', 'position', 'age', 'minutes_played']
    base_metrics = [col for col in df_master.columns if col not in ignore_cols and not str(col).endswith('_p90')]
    
    display_rows = []
    for metric in base_metrics:
        p90_col = f"{metric}_p90"
        p90_val = player_data[p90_col] if p90_col in df_master.columns else 0.0
        
        display_rows.append({
            "Metric": str(metric).replace('gk', 'GK ').replace('xg', 'xG ').capitalize(),
            "Total Value": round(player_data[metric], 2),
            "Per 90 Mins": p90_val
        })
        
    df_display = pd.DataFrame(display_rows)
    
    st.dataframe(
        df_display.sort_values(by="Metric"), 
        use_container_width=True, 
        hide_index=True,
        height=600 
    )

else:
    st.info("👈 Please search and select a player from the dropdown above to view their entire stat profile.")