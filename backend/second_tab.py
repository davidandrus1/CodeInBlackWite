import streamlit as st
import pandas as pd
import google.generativeai as genai

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
        
        age_range = st.slider("Age Range:", min_value=17, max_value=40, value=(17, 40))
        
        # Safe extraction of positions
        valid_positions = df_master['position'].dropna().unique().tolist()
        valid_positions.insert(0, "All Positions")
        selected_pos = st.selectbox("Position:", valid_positions)
        
        search_pressed = st.button("Search Database", type="primary", use_container_width=True)

    # ==========================================
    # 📊 RIGHT PANEL: RESULTS & AI
    # ==========================================
    with right_panel:
        if search_pressed:
            with st.spinner("Searching database..."):
                filtered_df = df_master.copy()
                
                # Age Filter (Safe calculation)
                filtered_df['safe_age'] = pd.to_numeric(filtered_df['age'], errors='coerce').fillna(99)
                filtered_df = filtered_df[(filtered_df['safe_age'] >= age_range[0]) & (filtered_df['safe_age'] <= age_range[1])]
                
                # Position Filter
                if selected_pos != "All Positions":
                    filtered_df = filtered_df[filtered_df['position'] == selected_pos]
                    
                # Name Filter
                if search_name:
                    filtered_df = filtered_df[filtered_df['original_name'].str.contains(search_name, case=False, na=False)]

                if filtered_df.empty:
                    st.warning("No players found matching these criteria.")
                else:
                    # --- AI GENERATION SECTION ---
                    st.subheader("🤖 AI Top Recommendations")
                    
                    if not api_key:
                        st.warning("⚠️ Please provide a Gemini API Key in the left panel to generate AI insights.")
                    else:
                        try:
                            genai.configure(api_key=api_key)
                            model = genai.GenerativeModel('gemini-2.5-flash')
                            
                            top_candidates = filtered_df.head(3)
                            candidates_data = top_candidates[['original_name', 'age', 'position', 'market_value_in_eur', 'minutes_played']].to_dict(orient='records')
                            
                            prompt = f"""
                            You are the Head Scout. I have filtered the database and found these top candidates: {candidates_data}.
                            Write a 3-sentence summary highlighting the best option based on their age, market value, and minutes played. 
                            Be decisive and professional.
                            """
                            
                            response = model.generate_content(prompt)
                            st.info(response.text)
                        except Exception as e:
                            st.error(f"AI Generation Failed: {e}")
                    
                    # --- DATAFRAME SECTION ---
                    st.divider()
                    st.subheader(f"📋 Full Results ({len(filtered_df)} players)")
                    
                    display_df = filtered_df[['original_name', 'age', 'position', 'market_value_in_eur', 'minutes_played']].copy()
                    
                    st.dataframe(
                        display_df,
                        column_config={
                            "original_name": "Player Name",
                            "age": "Age",
                            "position": "Position",
                            "market_value_in_eur": st.column_config.NumberColumn("Market Value", format="€%d"),
                            "minutes_played": "Mins Played"
                        },
                        use_container_width=True,
                        hide_index=True
                    )
        else:
            st.info("👈 Adjust your filters on the left and click 'Search Database' to begin.")