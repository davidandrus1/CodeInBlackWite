import sys
import os
import streamlit as st
import pandas as pd

FRONTEND_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(FRONTEND_DIR)

if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from Backend.data_processor import process_data
from tab_squad import render_squad_tab
from tab_search import render_search_tab

DATA_DIR = os.path.join(ROOT_DIR, "Data")
U_CLUJ_CSV = os.path.join(DATA_DIR, "u_cluj_current_squad.csv")

MECIURI_DIR = os.path.join(DATA_DIR, "Date - meciuri")
PLAYERS_JSON = os.path.join(MECIURI_DIR, "players (1).json")

st.set_page_config(
    page_title="U-Scout", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

if 'active_page' not in st.session_state:
    st.session_state.active_page = "SEARCH DATABASE"
if 'search_target_name' not in st.session_state:
    st.session_state.search_target_name = ""

st.markdown("""
    <style>
    .block-container { padding-top: 1.5rem; }
    
    div[role="radiogroup"] {
        display: flex;
        flex-direction: row;
        justify-content: center;
        background-color: #1a1a1a;
        padding: 6px;
        border-radius: 12px;
        width: fit-content;
        margin: 0 auto 20px auto;
        border: 1px solid #333333;
    }
    div[role="radiogroup"] > label {
        background-color: transparent;
        padding: 10px 40px !important;
        border-radius: 8px;
        cursor: pointer;
        transition: all 0.3s ease-in-out;
    }
    div[role="radiogroup"] > label[data-checked="true"] {
        background-color: #ffffff;
        box-shadow: 0px 4px 10px rgba(255, 255, 255, 0.1);
    }
    div[role="radiogroup"] > label[data-checked="true"] p {
        color: #000000 !important;
        font-weight: 800;
    }
    div[role="radiogroup"] [data-testid="stMarkdownContainer"] { margin-left: 0px !important; }
    div[role="radiogroup"] span[data-baseweb="radio"] { display: none !important; }
    
    div.stButton > button[kind="primary"] {
        background-color: #ffffff !important;
        border-color: #ffffff !important;
        color: #000000 !important;
        border-radius: 8px;
        font-weight: bold;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #e6e6e6 !important;
        border-color: #e6e6e6 !important;
    }

    .scout-card {
        background-color: #1a1a1a; padding: 20px; border-radius: 10px;
        border: 1px solid #ffffff; text-align: center;
        display: flex; flex-direction: column; justify-content: center;
    }
    .scout-card p { color: #ffffff; margin: 4px 0; font-size: 0.95rem; }
    .scout-card .value { color: #ffffff; font-size: 1.2rem; font-weight: bold; margin-top: 10px;}
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
@st.cache_data
def load_all_data():
    df_perf = process_data(PLAYERS_JSON, MECIURI_DIR)
    try:
        u_cluj_list = pd.read_csv(U_CLUJ_CSV)
    except:
        u_cluj_list = pd.DataFrame(columns=['Player ID']) 
    return df_perf, u_cluj_list

with st.spinner('Loading Database...'):
    df_master, u_cluj_names = load_all_data()

st.markdown("<h1 style='text-align: center;'>U-Scout</h1>", unsafe_allow_html=True)

selected_page = st.radio(
    "Navigation", 
    ["SEARCH DATABASE", "CURRENT SQUAD"],
    horizontal=True,
    label_visibility="collapsed",
    index=0 if st.session_state.active_page == "SEARCH DATABASE" else 1
)

if selected_page != st.session_state.active_page:
    st.session_state.active_page = selected_page
    st.rerun()

st.divider()

if st.session_state.active_page == "SEARCH DATABASE":
    render_search_tab(df_master, u_cluj_names)
else:
    render_squad_tab(df_master, u_cluj_names)