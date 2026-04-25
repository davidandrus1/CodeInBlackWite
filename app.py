import streamlit as st
import pandas as pd
from data_processor import process_data
from tab_squad import render_squad_tab
from tab_search import render_search_tab

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="U Cluj Scouting AI", 
    page_icon="🦅", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. Session State Initialization ---
if 'active_page' not in st.session_state:
    st.session_state.active_page = "🔍 SEARCH DATABASE" # Search is now the default!
if 'search_target_name' not in st.session_state:
    st.session_state.search_target_name = ""

# --- 3. Unified Styling ---
st.markdown("""
    <style>
    .block-container { padding-top: 1.5rem; }
    
    /* 🎯 THE SEGMENTED CONTROL TOGGLE (Looks like Chrome/Modern Tabs) */
    div[role="radiogroup"] {
        display: flex;
        flex-direction: row;
        justify-content: center;
        background-color: #16181f;
        padding: 6px;
        border-radius: 12px;
        width: fit-content;
        margin: 0 auto 20px auto;
        border: 1px solid #2e303e;
    }
    div[role="radiogroup"] > label {
        background-color: transparent;
        padding: 10px 40px !important;
        border-radius: 8px;
        cursor: pointer;
        transition: all 0.3s ease-in-out;
    }
    div[role="radiogroup"] > label[data-checked="true"] {
        background-color: #00FFAA;
        box-shadow: 0px 4px 10px rgba(0, 255, 170, 0.2);
    }
    div[role="radiogroup"] > label[data-checked="true"] p {
        color: #000000 !important;
        font-weight: 800;
    }
    div[role="radiogroup"] [data-testid="stMarkdownContainer"] { margin-left: 0px !important; }
    div[role="radiogroup"] span[data-baseweb="radio"] { display: none !important; }
    
    /* Card Styles */
    .scout-card {
        background-color: #0e1117; padding: 20px; border-radius: 10px;
        border: 1px solid #00FFAA; text-align: center;
        display: flex; flex-direction: column; justify-content: center;
    }
    .scout-card p { color: #ffffff; margin: 4px 0; font-size: 0.95rem; }
    .scout-card .value { color: #ffffff; font-size: 1.2rem; font-weight: bold; margin-top: 10px; color: #00FFAA;}
    </style>
    """, unsafe_allow_html=True)

# --- 4. Data Loading ---
@st.cache_data
def load_all_data():
    df_perf = process_data("Date - meciuri/players (1).json", "Date - meciuri")
    try:
        u_cluj_list = pd.read_csv("u_cluj_current_squad.csv")
    except:
        u_cluj_list = pd.DataFrame(columns=['Player ID']) 
    return df_perf, u_cluj_list

with st.spinner('Loading Database...'):
    df_master, u_cluj_names = load_all_data()

st.markdown("<h1 style='text-align: center;'>🦅 U Cluj - AI Scouting Platform</h1>", unsafe_allow_html=True)

# --- 5. Main Frame Navigation ---
selected_page = st.radio(
    "Navigation", 
    ["🔍 SEARCH DATABASE", "📋 CURRENT SQUAD"], # Search is now the first tab!
    horizontal=True,
    label_visibility="collapsed",
    index=0 if st.session_state.active_page == "🔍 SEARCH DATABASE" else 1
)

if selected_page != st.session_state.active_page:
    st.session_state.active_page = selected_page
    st.rerun()

st.divider()

# --- 6. Render the Active Page ---
if st.session_state.active_page == "🔍 SEARCH DATABASE":
    render_search_tab(df_master)
else:
    render_squad_tab(df_master, u_cluj_names)