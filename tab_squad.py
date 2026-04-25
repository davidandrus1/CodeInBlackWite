import streamlit as st
import pandas as pd
import plotly.graph_objects as go

try:
    from models.similarity import (
        compute_similarity, get_player_position_by_id, 
        get_player_name_by_id, get_players_for_position_excluding
    )
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

def format_currency(val):
    try:
        v = float(val)
        return f"€{int(v):,}" if v > 0 else "Not Available"
    except: return "Not Available"

def render_clickable_card(player_row, prefix_key, index_label="Potential Index", index_col="Growth_Potential", custom_index_str=None):
    name = player_row['original_name']
    
    if st.button(f"{name}", key=f"btn_{prefix_key}_{player_row['player_id']}", use_container_width=True, type="primary"):
        st.session_state.search_target_name = name
        st.session_state.active_page = "SEARCH DATABASE"
        st.rerun()
        
    if custom_index_str:
        index_html = f"<p>{index_label}: <b>{custom_index_str}</b></p>"
    else:
        val = float(player_row.get(index_col, 0))
        index_html = f"<p>{index_label}: <b>{val:.1f}</b></p>"

    st.markdown(f"""
        <div class="scout-card" style="margin-top: -15px; border-top-left-radius: 0; border-top-right-radius: 0; height: 140px;">
            <p>Age: <b>{int(float(player_row['age']))}</b> | Mins: <b>{int(float(player_row['minutes_played']))}</b></p>
            {index_html}
            <div class="value">{format_currency(player_row['market_value_in_eur'])}</div>
        </div>""", unsafe_allow_html=True)

def render_squad_tab(df_master, u_cluj_names):
    u_cluj_ids = u_cluj_names['Player ID'].astype(str).tolist()
    roster_df = df_master[df_master['player_id'].astype(str).isin(u_cluj_ids)].copy()

    position_order = ['FW', 'MD', 'DF', 'GK']
    roster_df['position_cat'] = pd.Categorical(roster_df['position'], categories=position_order, ordered=True)
    roster_df = roster_df.sort_values(by=['position_cat', 'original_name'])
    
    st.subheader(f"U Cluj Roster ({len(roster_df)} Players)")
    
    display_df = roster_df.copy()
    display_df['Display Value'] = display_df['market_value_in_eur'].apply(format_currency)
    
    event = st.dataframe(
        display_df[['original_name', 'age', 'position', 'Display Value']], 
        column_config={
            "original_name": "Name", "age": "Age", "position": "Pos.",
            "Display Value": "Market Value"
        },
        use_container_width=True, hide_index=True, selection_mode="single-row", on_select="rerun"
    )

    selected_rows = event.get("selection", {}).get("rows", [])
    if not selected_rows:
        st.info("Select a player from the roster to start analysis.")
        return

    target = roster_df.iloc[selected_rows[0]]
    st.divider()
    
    st.header(f"Replacement Analysis: {target['original_name']}")
    
    max_budget = st.slider("Set Maximum Transfer Budget (€)", min_value=0, max_value=5000000, value=2000000, step=100000)

    replacements_pool = df_master[~df_master['player_id'].astype(str).isin(u_cluj_ids)]
    
    budget_pool = replacements_pool[
        (pd.to_numeric(replacements_pool['market_value_in_eur'], errors='coerce').fillna(0) > 0) &
        (pd.to_numeric(replacements_pool['market_value_in_eur'], errors='coerce').fillna(0) <= max_budget)
    ]
    
    u21_df = budget_pool[
        (budget_pool['position'] == target['position']) & 
        (budget_pool['age'].astype(float) <= 21)
    ].sort_values('Growth_Potential', ascending=False).head(1)
    
    target_id = str(target['player_id'])
    position_ml = get_player_position_by_id(target_id) if ML_AVAILABLE else None
    name_a = get_player_name_by_id(target_id) if ML_AVAILABLE else None

    best_match_name = None
    best_match_score = -1
    
    if ML_AVAILABLE and name_a and position_ml:
        available_players = get_players_for_position_excluding(position_ml, u_cluj_ids)
        valid_budget_names = budget_pool['original_name'].tolist()
        available_players = [p for p in available_players if p in valid_budget_names]
        
        if available_players:
            with st.spinner("Analyzing profiles..."):
                progress_text = f"Processing {len(available_players)} profiles..."
                my_bar = st.progress(0, text=progress_text)
                for i, p_b in enumerate(available_players):
                    try:
                        res = compute_similarity(name_a, p_b, position_ml)
                        if not res.get("error"):
                            score = res["similarity_score"]
                            if score > best_match_score:
                                best_match_score = score
                                best_match_name = p_b
                    except:
                        continue
                    my_bar.progress((i + 1) / len(available_players), text=progress_text)
                my_bar.empty() 
                        
    liga_match_df = pd.DataFrame()
    if best_match_name:
        liga_match_df = budget_pool[budget_pool['original_name'] == best_match_name]
    if liga_match_df.empty:
        liga_match_df = budget_pool[
            budget_pool['position'] == target['position']
        ].sort_values('Performance_Index', ascending=False).head(1)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("### Selected Player")
        st.metric("Name", target['original_name'])
        st.metric("Value", format_currency(target['market_value_in_eur']))
        st.metric("Performance Index", f"{float(target.get('Performance_Index', 0)):.1f}")

    with c2:
        st.markdown("### Top U21 Prospect")
        if not u21_df.empty:
            render_clickable_card(u21_df.iloc[0], "u21", index_label="Potential Index", index_col="Growth_Potential")
        else:
            st.warning("No U21 prospects within budget found.")

    with c3:
        if not liga_match_df.empty:
            if best_match_score > 0:
                st.markdown("### Exact Tactical Match")
                render_clickable_card(liga_match_df.iloc[0], "liga1", index_label="ML Match", custom_index_str=f"{best_match_score}%")
            else:
                st.markdown("### Top Positional Performer")
                render_clickable_card(liga_match_df.iloc[0], "liga1", index_label="Performance", index_col="Performance_Index")
        else:
            st.markdown("### Exact Tactical Match")
            st.warning("No tactical matches within budget found.")

    st.divider()
    st.markdown("### Highest Performing League Options")
    
    excluded_names = []
    if not u21_df.empty: excluded_names.append(u21_df.iloc[0]['original_name'])
    if not liga_match_df.empty: excluded_names.append(liga_match_df.iloc[0]['original_name'])
    
    additional_df = budget_pool[
        (budget_pool['position'] == target['position']) & 
        (~budget_pool['original_name'].isin(excluded_names))
    ].sort_values('Performance_Index', ascending=False).head(4)

    if not additional_df.empty:
        grid = st.columns(4)
        for i, (_, row) in enumerate(additional_df.iterrows()):
            with grid[i]:
                render_clickable_card(row, f"add_{i}", index_label="Performance Index", index_col="Performance_Index")
    else:
        st.warning("No additional league options found within this budget.")

    st.divider()
    st.header("Player Compatibility Engine")
    if not ML_AVAILABLE:
        st.warning("ML Models are currently unavailable.")
        return
    if not position_ml or not name_a:
        st.warning(f"{target['original_name']} lacks advanced data.")
        return
    
    available_players_comp = get_players_for_position_excluding(position_ml, u_cluj_ids)
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Player A (Target)")
        st.info(f"**{name_a}** - {position_ml.replace('_', ' ').title()}")
    with col_b:
        st.markdown("#### Player B (Compare)")
        default_index = 0
        if best_match_name and best_match_name in available_players_comp:
            default_index = available_players_comp.index(best_match_name) + 1
        player_b = st.selectbox("Select Player B:", options=["-- Select Player --"] + available_players_comp, index=default_index, label_visibility="collapsed")

    if player_b != "-- Select Player --":
        if st.button("Calculate Compatibility", type="primary", use_container_width=True):
            with st.spinner("Processing..."):
                result = compute_similarity(name_a=name_a, name_b=player_b, position=position_ml)
                if result.get("error"):
                    st.error(result["error"])
                    return
                st.markdown("---")
                score1, score2, score3, score4 = st.columns(4)
                ov_score = result["similarity_score"]
                score1.metric(f"Overall Compatibility", f"{ov_score}/100")
                score2.metric("Physic Score", f"{result['scores']['fizic']}/100" if result['scores']['fizic'] else "N/A")
                score3.metric("Style Score", f"{result['scores']['stil']}/100")
                score4.metric("Quality Score", f"{result['scores']['calitate']}/100")
                st.markdown("<br>", unsafe_allow_html=True)
                text_col1, text_col2 = st.columns(2)
                with text_col1:
                    st.markdown("#### Common Things")
                    if result["report"]["similarities"]:
                        for s in result["report"]["similarities"]: st.success(s)
                    else:
                        st.info("No strong similarities found.")
                with text_col2:
                    st.markdown("#### Key Differences")
                    if result["report"]["differences"]:
                        for d in result["report"]["differences"]: st.error(d)
                    else:
                        st.info("No major discrepancies found.")
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("#### Tactical Footprint")
                labels = result["labels"]
                vals_a = list(result["radar_values_a"].values())
                vals_b = list(result["radar_values_b"].values())
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=vals_a + [vals_a[0]], theta=labels + [labels[0]],
                    fill="toself", name=name_a,
                    line=dict(color="#FFFB0C", width=3), fillcolor="rgba(255, 252, 56, 0.3)"
                ))
                fig.add_trace(go.Scatterpolar(
                    r=vals_b + [vals_b[0]], theta=labels + [labels[0]],
                    fill="toself", name=player_b,
                    line=dict(color="#888888", width=3), fillcolor="rgba(136, 136, 136, 0.3)"
                ))
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(size=10, color="gray"))),
                    showlegend=True, height=500, paper_bgcolor='rgba(0,0,0,0)', font=dict(color="white")
                )
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("#### Details per Attribute")
                breakdown_rows = []
                for label, val_a, val_b in zip(labels, vals_a, vals_b):
                    better_player = name_a if val_a > val_b else player_b if val_b > val_a else "Tie"
                    breakdown_rows.append({
                        "Attribute": label,
                        f"Player A ({name_a})": round(val_a, 3),
                        f"Player B ({player_b})": round(val_b, 3),
                        "Difference": round(abs(val_a - val_b), 3),
                        "Who is Better": better_player
                    })
                st.dataframe(pd.DataFrame(breakdown_rows), use_container_width=True, hide_index=True)