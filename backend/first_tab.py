import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from models.similarity import (
    compute_similarity,
    get_player_position_by_id,
    get_player_name_by_id,
    get_players_for_position_excluding,
)


def render_first_tab(df_master, u_cluj_names):

    # ── Roster U Cluj ──
    u_cluj_ids = u_cluj_names['Player ID'].astype(str).tolist()
    roster_df = df_master[df_master['player_id'].astype(str).isin(u_cluj_ids)].copy()

    position_order = ['FW', 'MD', 'DF', 'GK']
    roster_df['position_cat'] = pd.Categorical(
        roster_df['position'], categories=position_order, ordered=True
    )
    roster_df = roster_df.sort_values(by=['position_cat', 'original_name'])

    st.subheader(f"U Cluj Roster ({len(roster_df)} Players Matched)")

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

    if not selected_rows:
        st.info("Select a player from the roster to start analysis.")
        return

    # ── Jucătorul selectat ──
    selected_idx = selected_rows[0]
    target = roster_df.iloc[selected_idx]

    st.divider()
    st.header(f"Replacement Analysis: {target['original_name']}")

    # Pool fără jucătorii U Cluj
    replacements_pool = df_master[
        ~df_master['player_id'].astype(str).isin(u_cluj_ids)
    ]

    u21_df = replacements_pool[
        (replacements_pool['position'] == target['position']) &
        (replacements_pool['age'].astype(float) <= 21)
    ].sort_values('minutes_played', ascending=False).head(1)

    liga_df = replacements_pool[
        replacements_pool['position'] == target['position']
    ].sort_values('market_value_in_eur', ascending=False).head(5)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("### 👤 Selected")
        st.metric("Name", target['original_name'])
        st.metric("Value", f"€{int(float(target['market_value_in_eur'])):,}")

    with c2:
        st.markdown("### ✨ Top U21 Prospect")
        if not u21_df.empty:
            p = u21_df.iloc[0]
            st.markdown(f"""
                <div class="scout-card">
                    <h3>{p['original_name']}</h3>
                    <p>Age: {int(float(p['age']))} | Mins: {int(float(p['minutes_played']))}</p>
                    <div class="value">€{int(float(p['market_value_in_eur'])):,}</div>
                </div>""", unsafe_allow_html=True)
        else:
            st.warning("No U21 prospects found for this position.")

    with c3:
        st.markdown("### 🏟️ Liga Replacement 1")
        if not liga_df.empty:
            r = liga_df.iloc[0]
            st.markdown(f"""
                <div class="scout-card">
                    <h3>{r['original_name']}</h3>
                    <p>Age: {int(float(r['age']))} | Mins: {int(float(r['minutes_played']))}</p>
                    <div class="value">€{int(float(r['market_value_in_eur'])):,}</div>
                </div>""", unsafe_allow_html=True)

    if len(liga_df) > 1:
        st.markdown("### 📋 Additional Options")
        grid = st.columns(4)
        for i, (_, row) in enumerate(liga_df.iloc[1:5].iterrows()):
            with grid[i]:
                st.markdown(f"""
                    <div class="scout-card">
                        <h3>{row['original_name']}</h3>
                        <p>Age: {int(float(row['age']))}</p>
                        <div class="value">€{int(float(row['market_value_in_eur'])):,}</div>
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

    # Găsim position_ml și shortName după Player ID
    target_id = str(target['player_id'])
    position_ml = get_player_position_by_id(target_id)
    name_a      = get_player_name_by_id(target_id)

    if not position_ml or not name_a:
        st.warning(
            "Jucătorul selectat nu are date suficiente în sistemul ML. "
            "Verifică că train.py a fost rulat."
        )
        return

    # Jucători disponibili — exclus U Cluj
    available_players = get_players_for_position_excluding(position_ml, u_cluj_ids)

    if not available_players:
        st.warning(f"Nu există jucători disponibili în ligă pentru poziția {position_ml}.")
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

        # ── Scor global ──
        score = result["similarity_score"]
        color = "🟢" if score >= 75 else "🟡" if score >= 50 else "🔴"
        st.markdown(f"## {color} Similarity Score: **{score}/100**")

        # ── Scoruri detaliate ──
        scores = result["scores"]
        c1, c2, c3 = st.columns(3)
        c1.metric("🏋️ Fizic", f"{scores['fizic']}/100" if scores['fizic'] else "N/A")
        c2.metric("🎨 Stil", f"{scores['stil']}/100")
        c3.metric("🎯 Calitate", f"{scores['calitate']}/100")

        # ── Raport text ──
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

        # ── Spider Chart ──
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

        # ── Breakdown tabel ──
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