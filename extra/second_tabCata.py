import streamlit as st
import pandas as pd
import google.generativeai as genai
from filtering_logic import apply_scouting_filters

def render_second_tab(df_master):
    # Left column (1 part), Right column (3 parts)
    left_panel, right_panel = st.columns([1, 3], gap="large")

    # ==========================================
    # 🎛️ LEFT PANEL: FILTERS
    # ==========================================
    with left_panel:
        st.subheader("⚙️ Search Filters")
        
        api_key = st.text_input("Gemini API Key:", type="password", help="Required for AI recommendations")
        st.divider()
        
        search_name = st.text_input("Player Name (Optional):")
        
        age_range = st.slider("Age Range:", min_value=16, max_value=40, value=(16, 40))
        
        # --- UPDATED POSITION FILTER LOGIC ---
        if 'position' in df_master.columns:
            raw_positions = df_master['position'].dropna().unique()
            valid_positions = [pos for pos in raw_positions if str(pos).strip() != ""]
        else:
            valid_positions = []
            
        valid_positions.insert(0, "Any Position")
        selected_pos = st.selectbox("Position:", valid_positions)
        # ---------------------------------------

        max_value = st.number_input("Max Market Value (€):", min_value=0, value=1200000, step=100000)

        height_range = st.slider("Height (m):", min_value=1.50, max_value=2.20, value=(1.50, 2.20), step=0.01)

        foot_options = ["Any", "Right", "Left", "Both"]
        selected_foot = st.selectbox("Preferred Foot:", foot_options, index=0)

        min_minutes = st.number_input("Min Minutes Played:", min_value=0, value=0, step=50)

    # ==========================================
    # 📊 RIGHT PANEL: RESULTS & AI
    # ==========================================
    with right_panel:
        # The table now filters automatically as you change inputs!
        filtered_df = apply_scouting_filters(
            df=df_master, 
            name_query=search_name, 
            age_range=age_range, 
            position=selected_pos, 
            max_value=max_value, 
            height_range=height_range, 
            foot=selected_foot, 
            min_minutes=min_minutes
        )

        if filtered_df.empty:
            st.warning("No players found matching these criteria.")
        else:
            # --- DATAFRAME SECTION ---
            st.subheader(f"📋 Full Results ({len(filtered_df)} players)")
            
            display_cols = [c for c in ['original_name', 'age', 'position', 'height', 'foot', 'market_value_in_eur', 'minutes_played'] if c in filtered_df.columns]
            display_df = filtered_df[display_cols].copy()
            
            st.dataframe(
                display_df,
                column_config={
                    "original_name": "Player Name",
                    "age": "Age",
                    "position": "Position",
                    "height": "Height",
                    "foot": "Foot",
                    "market_value_in_eur": st.column_config.NumberColumn("Market Value", format="€%d"),
                    "minutes_played": "Mins Played"
                },
                use_container_width=True,
                hide_index=True
            )

            st.divider()

            # --- AI GENERATION SECTION ---
            st.subheader("🤖 AI Top Recommendations")
            
            # The AI is safely protected behind its own button so it doesn't burn credits
            if st.button("Generate AI Scout Report", type="primary"):
                if not api_key:
                    st.warning("⚠️ Please provide a Gemini API Key in the left panel to generate AI insights.")
                else:
                    with st.spinner("Analyzing top candidates..."):
                        try:
                            genai.configure(api_key=api_key)
                            model = genai.GenerativeModel('gemini-2.5-flash')
                            
                            top_candidates = filtered_df.head(3)
                            cols_for_ai = [c for c in ['original_name', 'age', 'position', 'market_value_in_eur', 'minutes_played'] if c in top_candidates.columns]
                            candidates_data = top_candidates[cols_for_ai].to_dict(orient='records')
                            
                            prompt = f"""
                            You are the Head Scout. I have filtered the database and found these top candidates: {candidates_data}.
                            Write a 3-sentence summary highlighting the best option based on their age, market value, and minutes played. 
                            Be decisive and professional.
                            """
                            
                            response = model.generate_content(prompt)
                            st.info(response.text)
                        except Exception as e:
                            st.error(f"AI Generation Failed: {e}")