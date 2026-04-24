import streamlit as st
import pandas as pd
import google.generativeai as genai
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from data_processor import process_data

st.set_page_config(page_title="U Cluj Scouting AI", page_icon="🦅", layout="wide")

# --- SIDEBAR: API CONFIGURATION ---
with st.sidebar:
    st.header("⚙️ AI Settings")
    api_key = st.text_input("Enter Gemini API Key (for GenAI Reports):", type="password")
    st.markdown("*Get a free key at [Google AI Studio](https://aistudio.google.com/)*")
    st.divider()
    st.image("https://upload.wikimedia.org/wikipedia/ro/3/3e/U_Cluj.svg", width=150)

@st.cache_data
def load_cached_data():
    return process_data("Date - meciuri/players (1).json", "Date - meciuri")

with st.spinner('Aggregating All Wyscout Data...'):
    df_master = load_cached_data()

st.title("🦅 Universitatea Cluj - AI Scouting Tool")

st.markdown("### 🎛️ Scouting Filters")
col_f1, col_f2 = st.columns(2)
max_budget = col_f1.slider("💰 Maximum Budget (€)", min_value=0, max_value=1200000, value=400000, step=10000)

# Filter the master dataframe based on the slider BEFORE creating the dropdown
df_filtered = df_master[df_master['market_value_in_eur'] <= max_budget]

# --- SEARCH DROPDOWN ---
# Update the dropdown to only show players who fit the budget!
player_names = df_filtered['name'].dropna().sort_values().unique().tolist()
selected_player = st.selectbox("🔍 Search Player Name (Filtered by Budget):", options=["-- Select a Player --"] + player_names)

if selected_player != "-- Select a Player --":
    # Make sure to pull the data from the FILTERED dataframe now
    player_data = df_filtered[df_filtered['name'] == selected_player].iloc[0]
    
    st.header(f"👤 {player_data['original_name']}") 
    
    # --- UPDATE METRICS TO SHOW VALUE ---
    col1, col2, col3, col4 = st.columns(4) # Changed to 4 columns!
    col1.metric("Age", int(player_data['age']) if pd.notnull(player_data['age']) else "N/A")
    col2.metric("Position", str(player_data['position']).upper())
    col3.metric("Total Season Minutes", int(player_data['minutes_played']))
    # Format the market value to look nice (e.g., €1,200,000)
    formatted_value = f"€{int(player_data['market_value_in_eur']):,}"
    col4.metric("Market Value", formatted_value)
    st.divider()
    
    # ==========================================
    # 🤖 GEN AI SCOUT REPORT SECTION
    # ==========================================
    st.subheader("🤖 AI Scout Summary")
    if st.button("Generate Tactical Report with Gemini"):
        if not api_key:
            st.error("⚠️ Please paste your Gemini API Key in the sidebar first!")
        else:
            with st.spinner("Analyzing 80+ performance metrics..."):
                try:
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel('gemini-2.5-flash') 
                    stats_dict = player_data.drop(['player_id', 'name', 'original_name']).to_dict()
                    prompt = f"""
                    You are the Chief Data Scout for Universitatea Cluj in the Romanian SuperLiga. 
                    I am giving you the season performance data for a player named {player_data['original_name']}, 
                    who is {player_data['age']} years old and plays as a {player_data['position']}. 
                    
                    Here is his data dictionary (including raw counts and per-90 metrics):
                    {stats_dict}
                    
                    Write a punchy, professional, 3-sentence scouting report. 
                    Sentence 1: Highlight his biggest statistical strength on the pitch.
                    Sentence 2: Point out a statistical weakness or area for improvement.
                    Sentence 3: Conclude on whether his age and output suggest high growth potential for a club like U Cluj.
                    
                    Do NOT just read the numbers back to me. Talk like a tactical expert evaluating a player.
                    """
                    response = model.generate_content(prompt)
                    st.info(response.text)
                except Exception as e:
                    st.error(f"API Error: {e}")
    
    st.divider()
    st.subheader("📊 Complete Performance Data")
    
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
        height=400 
    )

    # ==========================================
    # 🎯 AI SIMILARITY & RADAR CHART
    # ==========================================
    st.divider()
    st.subheader("🎯 AI Replacement Engine")
    st.markdown("Find the most statistically similar **U24 player** in the SuperLiga to replace this veteran.")
    
    if st.button("Find Similar U24 Prospect"):
        with st.spinner("Calculating Cosine Similarity across SuperLiga..."):
            # Using exact Wyscout dynamic keys
            features = ['xgShot_p90', 'assists_p90', 'progressiveRun_p90', 'defensiveDuels_p90', 'defensiveDuelsWon_p90']
            
            # Verify features exist to prevent crashes
            available_features = [f for f in features if f in df_master.columns]
            
            if len(available_features) == 0:
                st.error("Error: Could not find necessary metrics for comparison.")
            else:
                df_sim = df_master.copy()
                scaler = MinMaxScaler()
                scaled_matrix = scaler.fit_transform(df_sim[available_features].fillna(0))
                
                target_idx = df_sim[df_sim['name'] == selected_player].index[0]
                
                # AI MATH: Calculate similarity between the target and EVERYONE else
                sim_scores = cosine_similarity([scaled_matrix[target_idx]], scaled_matrix)[0]
                df_sim['similarity_match'] = (sim_scores * 100).round(1)
                
                # Filter results: Must be Under 24, and cannot be the player we just searched for
                df_sim = df_sim[(df_sim['age'] <= 24) & (df_sim['name'] != selected_player)]
                
                if df_sim.empty:
                    st.warning("No U24 players found in the dataset with enough minutes to compare.")
                else:
                    top_match = df_sim.sort_values('similarity_match', ascending=False).iloc[0]
                    
                    st.success(f"**Top Match:** {top_match['original_name']} (Age: {int(top_match['age'])}, Position: {top_match['position']}) — **{top_match['similarity_match']}% Match**")
                    
                    # --- DRAW THE RADAR CHART ---
                    categories = ['xG Shot', 'Assists', 'Prog. Runs', 'Def. Duels', 'Duels Won']
                    
                    target_radar = scaled_matrix[target_idx]
                    match_radar = scaled_matrix[top_match.name]
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatterpolar(
                        r=target_radar, theta=categories, fill='toself', name=player_data['original_name']
                    ))
                    fig.add_trace(go.Scatterpolar(
                        r=match_radar, theta=categories, fill='toself', name=top_match['original_name']
                    ))
                    
                    fig.update_layout(
                        polar=dict(radialaxis=dict(visible=False)), 
                        showlegend=True, 
                        title="Playstyle Statistical Overlap"
                    )
                    
                    
                    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👈 Please search and select a player from the dropdown above to view their entire stat profile.")