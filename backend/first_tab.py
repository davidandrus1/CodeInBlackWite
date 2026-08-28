import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from models.similarity import (
    compute_similarity,
    get_player_position_by_id,
    get_player_name_by_id,
    get_players_for_position_excluding,
    get_similar_better_players,
)

def _get_market_value(df_master, pid_str: str) -> float:
    row = df_master[df_master['player_id'].astype(str) == pid_str]
    if row.empty:
        return 0.0
    try:
        return float(row.iloc[0]['market_value_in_eur'])
    except (ValueError, TypeError):
        return 0.0

def render_first_tab(df_master, u_cluj_names):

    # ── Roster U Cluj ──
    u_cluj_ids = u_cluj_names['Player ID'].astype(str).tolist()
    roster_df = df_master[df_master['player_id'].astype(str).isin(u_cluj_ids)].copy()

    position_order = ['FW', 'MD', 'DF', 'GK']
    roster_df['position_cat'] = pd.Categorical(
        roster_df['position'], categories=position_order, ordered=True
    )
    roster_df = roster_df.sort_values(by=['position_cat', 'original_name'])

    # ── Session State ──
    if "roster_expanded" not in st.session_state:
        st.session_state.roster_expanded = True

    # ── Roster Table în Expander ──
    with st.expander(
        f"📋 U Cluj Roster ({len(roster_df)} Players) — "
        f"{'click to expand' if not st.session_state.roster_expanded else 'click to collapse'}",
        expanded=st.session_state.roster_expanded
    ):
        event = st.dataframe(
            roster_df[['original_name', 'age', 'position', 'market_value_in_eur']],
            column_config={
                "original_name": "Name",
                "age": "Age",
                "position": "Position",
                "market_value_in_eur": st.column_config.NumberColumn("Value", format="€%d")
            },
            use_container_width=True,
            hide_index=True,
            selection_mode="single-row",
            on_select="rerun"
        )
        selected_rows = event.get("selection", {}).get("rows", [])

        if selected_rows:
            st.session_state.roster_expanded = False
            st.session_state.selected_idx = selected_rows[0]

        if not selected_rows and "selected_idx" not in st.session_state:
            st.info("Select a player from the roster to start analysis.")
            return

    # ── Buton Select Different Player ──
    if not st.session_state.roster_expanded:
        if st.button("🔄 Select Different Player"):
            st.session_state.roster_expanded = True
            if "selected_idx" in st.session_state:
                del st.session_state.selected_idx
            st.rerun()

    if "selected_idx" not in st.session_state:
        return

    target      = roster_df.iloc[st.session_state.selected_idx]
    target_id   = str(target['player_id'])
    position_ml = get_player_position_by_id(target_id)
    name_a      = get_player_name_by_id(target_id)

    st.divider()
    st.header(f"Analysis: {target['original_name']}")

    # ==========================================
    # 🤖 SIMILAR & BETTER PLAYERS
    # ==========================================
    st.subheader("🤖 Similar & Better Players")
    st.markdown(
        "Jucători cu **stil similar** dar **performanță mai bună** — "
        "sortați de la cel mai bun la stânga."
    )

    if not position_ml:
        st.warning("Poziția ML nu a putut fi detectată pentru acest jucător.")
    else:
        with st.spinner("Calculând similaritate și performanță..."):
            better_players = get_similar_better_players(
                player_id=target_id,
                position_ml=position_ml,
                exclude_ids=u_cluj_ids,
                df_master=df_master,
                top_n=5,
                min_similarity=50.0,
            )

            better_players = [
            p for p in better_players
            if _get_market_value(df_master, str(p["playerId"])) < 1_000_000
            ]
 
            u21_players = get_similar_better_players(
                player_id=target_id,
                position_ml=position_ml,
                exclude_ids=u_cluj_ids,
                df_master=df_master,
                top_n=1,
                min_similarity=30.0,
                max_age=21,
            )

            u21_players = [
            p for p in u21_players
            if _get_market_value(df_master, str(p["playerId"])) < 1_000_000
            ]

        if not better_players and not u21_players:
            st.info("Nu s-au găsit jucători cu stil similar și performanță mai bună.")
        else:
            # ── Top players ──
            if better_players:
                cols = st.columns(len(better_players))
                for i, player in enumerate(better_players):
                    pid_str   = str(player["playerId"])
                    meta      = df_master[df_master['player_id'].astype(str) == pid_str]
                    orig_name = meta.iloc[0]['original_name'] if not meta.empty else player["name"]
                    age_str   = str(int(float(meta.iloc[0]['age']))) if not meta.empty and pd.notnull(meta.iloc[0]['age']) else "N/A"
                    val_str   = f"€{int(float(meta.iloc[0]['market_value_in_eur'])):,}" if not meta.empty and float(meta.iloc[0]['market_value_in_eur']) > 0 else "N/A"
                    diff      = round(player['performance_score'] - player['performance_target'], 1)

                    with cols[i]:
                        st.markdown(f"""
                            <div class="scout-card">
                                <h3>{orig_name}</h3>
                                <p>Age: {age_str}</p>
                                <p>🎯 Similarity: {player['similarity_score']}%</p>
                                <p>📈 +{diff}% performanță</p>
                                <div class="value">{val_str}</div>
                            </div>""", unsafe_allow_html=True)

            # ── Card U21 ──
            if u21_players:
                st.markdown("### ✨ Top U21 Similar")
                u21 = u21_players[0]
                pid_str   = str(u21["playerId"])
                meta      = df_master[df_master['player_id'].astype(str) == pid_str]
                orig_name = meta.iloc[0]['original_name'] if not meta.empty else u21["name"]
                age_str   = str(int(float(meta.iloc[0]['age']))) if not meta.empty and pd.notnull(meta.iloc[0]['age']) else "N/A"
                val_str   = f"€{int(float(meta.iloc[0]['market_value_in_eur'])):,}" if not meta.empty and float(meta.iloc[0]['market_value_in_eur']) > 0 else "N/A"
                diff      = round(u21['performance_score'] - u21['performance_target'], 1)

                col_u21, _ = st.columns([1, 3])
                with col_u21:
                    st.markdown(f"""
                        <div class="scout-card">
                            <h3>{orig_name}</h3>
                            <p>Age: {age_str} ⭐ U21</p>
                            <p>🎯 Similarity: {u21['similarity_score']}%</p>
                            <p>📈 +{diff}% performanță</p>
                            <div class="value">{val_str}</div>
                        </div>""", unsafe_allow_html=True)

    # ==========================================
    # ⚔️ COMPATIBILITY ENGINE
    # ==========================================
    st.divider()
    st.subheader("⚔️ Player Compatibility Engine")
    st.markdown(
        "Compară jucătorul selectat cu un jucător din ligă "
        "de **aceeași poziție** și vezi gradul de similaritate."
    )

    if not position_ml or not name_a:
        st.warning("Jucătorul selectat nu are date suficiente în sistemul ML.")
        return

    available_players = get_players_for_position_excluding(position_ml, u_cluj_ids)

    if not available_players:
        st.warning(f"Nu există jucători disponibili pentru poziția {position_ml}.")
        return

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Jucător A", target['original_name'])
        st.caption(f"Poziție ML: {position_ml}")
    with col_b:
        player_b = st.selectbox(
            "🔍 Alege Jucătorul B pentru comparație:",
            options=["-- Selectează --"] + available_players,
            key="compat_player_b"
        )

    if player_b == "-- Selectează --":
        return

    if st.button("⚡ Calculează Compatibilitatea", type="primary"):
        with st.spinner("Calculând similaritatea..."):
            result = compute_similarity(
                name_a=name_a,
                name_b=player_b,
                position=position_ml,
            )

        if result["error"]:
            st.error(result["error"])
            return

        score = result["similarity_score"]
        color = "🟢" if score >= 75 else "🟡" if score >= 50 else "🔴"
        st.markdown(f"## {color} Similarity Score: **{score}/100**")

        scores = result["scores"]
        c1, c2, c3 = st.columns(3)
        c1.metric("🏋️ Fizic", f"{scores['fizic']}/100" if scores['fizic'] else "N/A")
        c2.metric("🎨 Stil", f"{scores['stil']}/100")
        c3.metric("🎯 Calitate", f"{scores['calitate']}/100")

        report = result["report"]
        st.markdown("#### 📋 Raport Scout")
        st.info(report["fizic"])

        if report["similarities"]:
            st.markdown("**Puncte comune:**")
            for s in report["similarities"]:
                st.markdown(s)

        if report["differences"]:
            st.markdown("**Diferențe cheie:**")
            for d in report["differences"]:
                st.markdown(d)

        labels = result["labels"]
        vals_a = list(result["radar_values_a"].values())
        vals_b = list(result["radar_values_b"].values())

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=vals_a + [vals_a[0]],
            theta=labels + [labels[0]],
            fill="toself",
            name=target['original_name'],
            line=dict(color="#1f77b4", width=2),
            fillcolor="rgba(31, 119, 180, 0.2)",
        ))
        fig.add_trace(go.Scatterpolar(
            r=vals_b + [vals_b[0]],
            theta=labels + [labels[0]],
            fill="toself",
            name=player_b,
            line=dict(color="#ff7f0e", width=2),
            fillcolor="rgba(255, 127, 14, 0.2)",
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(size=9))),
            showlegend=True,
            title=dict(text=f"{target['original_name']} vs {player_b}", x=0.5),
            height=450,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### 📊 Detaliu per atribut")
        breakdown_rows = []
        for label, val_a, val_b in zip(labels, vals_a, vals_b):
            breakdown_rows.append({
                "Atribut":               label,
                target['original_name']: round(val_a, 3),
                player_b:                round(val_b, 3),
                "Diferență":             round(abs(val_a - val_b), 3),
                "Mai bun":               target['original_name'] if val_a > val_b else player_b,
            })
        st.dataframe(
            pd.DataFrame(breakdown_rows),
            use_container_width=True,
            hide_index=True,
        )