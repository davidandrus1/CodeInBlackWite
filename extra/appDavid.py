import streamlit as st
import pandas as pd
from data_processor import process_data
# Import the custom functions we will build in the other files
from tab_squad import render_first_tab
from tab_search import render_second_tab

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="U Cluj Scouting AI", 
    page_icon="🦅", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. Unified Styling ---
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
    # Load the performance database
    df_perf = process_data("Date - meciuri/players (1).json", "Date - meciuri")
    
    # Load U Cluj Squad
    try:
        u_cluj_list = pd.read_csv("u_cluj_current_squad.csv")
    except:
        u_cluj_list = pd.DataFrame(columns=['Player ID']) # Fallback if file is missing
        
    return df_perf, u_cluj_list

with st.spinner('Loading Database...'):
    df_master, u_cluj_names = load_all_data()

# --- 4. Main Interface & Tabs ---
st.title("🦅 U Cluj - Roster & Replacement Finder")

# Create the Tabs
tab1, tab2 = st.tabs(["📋 CURRENT SQUAD", "🔍 SEARCH DATABASE"])

# Pass the data into the separate files to render!
with tab1:
    render_first_tab(df_master, u_cluj_names)

with tab2:
    render_second_tab(df_master)