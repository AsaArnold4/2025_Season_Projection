import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NFL Win Prediction · 2025",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Colour palette ─────────────────────────────────────────────────────────────
BEARS        = "#0B162A"
BEARS_ORANGE = "#E64100"
WIN_BLUE     = "#1D4ED8"   # clear, dark blue for wins
LOSS_RED     = "#B91C1C"   # clear, dark red for losses
BLUE         = "#2563EB"
TEAL         = "#0D9488"
AMBER        = "#D97706"
RED          = "#DC2626"
SLATE        = "#64748B"   # darkened from #94A3B8 so it reads on white
GREEN        = "#15803D"
PURPLE       = "#7C3AED"
BORDER       = "#CBD5E1"
TEXT_DARK    = "#1E293B"

TEAM_COLORS = {
    "ARI":"#97233F","ATL":"#A71930","BAL":"#241773","BUF":"#00338D",
    "CAR":"#0085CA","CHI":BEARS_ORANGE,"CIN":"#FB4F14","CLE":"#FF3C00",
    "DAL":"#003594","DEN":"#FB4F14","DET":"#0076B6","GB":"#203731",
    "HOU":"#03202F","IND":"#002C5F","JAX":"#006778","KC":"#E31837",
    "LAC":"#0073CF","LAR":"#003594","LV":"#4B5563","MIA":"#008E97",
    "MIN":"#4F2683","NE":"#002244","NO":"#9A7B4B","NYG":"#0B2265",
    "NYJ":"#125740","PHI":"#004C54","PIT":"#C89B0A","SEA":"#002244",
    "SF":"#AA0000","TB":"#D50A0A","TEN":"#4B92DB","WAS":"#5A1414",
}

DIVISIONS = {
    "AFC East":  ["BUF","MIA","NE","NYJ"],
    "AFC North": ["BAL","CIN","CLE","PIT"],
    "AFC South": ["HOU","IND","JAX","TEN"],
    "AFC West":  ["DEN","KC","LV","LAC"],
    "NFC East":  ["DAL","NYG","PHI","WAS"],
    "NFC North": ["CHI","DET","GB","MIN"],
    "NFC South": ["ATL","CAR","NO","TB"],
    "NFC West":  ["ARI","LAR","SEA","SF"],
}
TEAM_DIV  = {t: d for d, ts in DIVISIONS.items() for t in ts}
TEAM_CONF = {t: ("AFC" if d.startswith("AFC") else "NFC") for t, d in TEAM_DIV.items()}

DATA = os.path.join(os.path.dirname(__file__), "data")

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    proj_noveg = pd.read_json(f"{DATA}/proj_noveg.json")
    proj_veg   = pd.read_json(f"{DATA}/proj_veg.json")
    sched      = pd.read_json(f"{DATA}/sched_2025.json")
    team_roll  = pd.read_json(f"{DATA}/team_rolling.json")
    elo_2025   = pd.read_json(f"{DATA}/elo_2025.json")
    final_wins = pd.read_json(f"{DATA}/final_wins.json")
    model_res  = json.load(open(f"{DATA}/model_results.json"))
    feat_imp   = json.load(open(f"{DATA}/feature_importance.json"))
    otc_sp     = pd.read_csv(f"{DATA}/otc_positional_spending.csv")
    otc_ct     = pd.read_csv(f"{DATA}/otc_player_contracts.csv")
    hfa        = pd.read_json(f"{DATA}/hfa.json")
    return (proj_noveg, proj_veg, sched, team_roll, elo_2025,
            final_wins, model_res, feat_imp, otc_sp, otc_ct, hfa)

(proj_noveg, proj_veg, sched, team_roll, elo_2025,
 final_wins, model_res, feat_imp, otc_sp, otc_ct, hfa) = load_data()

ALL_TEAMS = sorted(proj_noveg["team"].unique())

# Midseason = projection made before Week 10 (halfway through season)
PRESEASON_WK  = 1
MIDSEASON_WK  = 10

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        f"<div style='background:{BEARS};padding:16px 12px;border-radius:8px;margin-bottom:16px'>"
        f"<span style='color:white;font-size:20px;font-weight:700'>NFL Win Predictor</span><br>"
        f"<span style='color:#94A3B8;font-size:12px'>2025 Season · Bears Analytics</span>"
        f"</div>", unsafe_allow_html=True)

    page = st.radio("Navigation", [
        "🏈 Team Deep Dive",
        "📊 Model Comparison",
        "🗺️ League Projections",
        "💰 Cap Spending",
        "🔬 EDA & Methodology",
    ], label_visibility="collapsed")

    st.divider()
    team = st.selectbox("Select Team", ALL_TEAMS,
                        index=ALL_TEAMS.index("CHI"),
                        key="global_team")

    if team != "CHI":
        if st.button("Reset to Bears", use_container_width=True):
            st.session_state["global_team"] = "CHI"
            st.rerun()

    st.divider()
    model_choice = st.selectbox("Primary Model", [
        "XGBoost (No Vegas)", "Random Forest (No Vegas)",
        "Neural Network (No Vegas)", "XGBoost (Vegas)", "Random Forest (Vegas)",
    ], index=0)

    proj_df   = proj_noveg if "No Vegas" in model_choice else proj_veg
    vegas_acc = model_res["vegas_acc"]

    st.caption(f"Vegas baseline: **{vegas_acc:.1%}**")
    st.caption("Train: 2018–2024 W1-17\nTest: 2025 W1-17")

# ── Plotly helper ─────────────────────────────────────────────────────────────
def clean(fig, h=400, margins=None):
    m = margins or dict(l=48, r=24, t=36, b=48)
    fig.update_layout(
        height=h, margin=m,
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family="Inter, system-ui, sans-serif", size=13, color=TEXT_DARK),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=BORDER,
                     tickcolor=BORDER, tickfont=dict(color=SLATE))
    fig.update_yaxes(showgrid=True, gridcolor=BORDER, zeroline=False,
                     linecolor=BORDER, tickfont=dict(color=SLATE))
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — TEAM DEEP DIVE
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏈 Team Deep Dive":
    tc = TEAM_COLORS.get(team, BLUE)

    final    = final_wins[final_wins["team"] == team]["actual_wins_2025"].values
    actual_w = int(final[0]) if len(final) else None
    actual_l = 17 - actual_w if actual_w is not None else None

    pre_proj      = proj_df[(proj_df["team"] == team) & (proj_df["proj_week"] == PRESEASON_WK)]
    pre_wins      = float(pre_proj["proj_total_wins"].values[0]) if len(pre_proj) else None
    mid_proj      = proj_df[(proj_df["team"] == team) & (proj_df["proj_week"] == MIDSEASON_WK)]
    mid_wins      = float(mid_proj["proj_total_wins"].values[0]) if len(mid_proj) else None
    mid_ci_lo     = float(mid_proj["ci_lo_80"].values[0]) if len(mid_proj) else None
    mid_ci_hi     = float(mid_proj["ci_hi_80"].values[0]) if len(mid_proj) else None

    div  = TEAM_DIV.get(team, "")
    conf = TEAM_CONF.get(team, "")
    record_str = f"{actual_w}–{actual_l}" if actual_w is not None else "—"

    st.markdown(
        f"<div style='background:{tc};padding:20px 24px;border-radius:10px;margin-bottom:20px'>"
        f"<div style='display:flex;justify-content:space-between;align-items:center'>"
        f"<div>"
        f"<span style='color:white;font-size:28px;font-weight:800'>{team}</span>"
        f"<span style='color:rgba(255,255,255,0.75);font-size:14px;margin-left:12px'>{div} · {conf}</span>"
        f"</div>"
        f"<div style='text-align:right'>"
        f"<span style='color:white;font-size:36px;font-weight:800'>{record_str}</span>"
        f"<span style='color:rgba(255,255,255,0.75);font-size:14px;display:block'>2025 Final Record</span>"
        f"</div></div></div>", unsafe_allow_html=True)

    # KPI row — 3 metrics (removed Projection Error)
    k1, k2, k3 = st.columns(3)
    with k1:
        delta_pre = f"{actual_w - pre_wins:+.1f} vs actual" if (pre_wins and actual_w is not None) else None
        st.metric("Preseason Projection", f"{pre_wins:.1f} W" if pre_wins else "—", delta=delta_pre)
    with k2:
        delta_mid = f"{actual_w - mid_wins:+.1f} vs actual" if (mid_wins and actual_w is not None) else None
        st.metric("Midseason Projection (Wk 10)", f"{mid_wins:.1f} W" if mid_wins else "—", delta=delta_mid)
    with k3:
        st.metric("Midseason 80% CI", f"{mid_ci_lo:.0f}–{mid_ci_hi:.0f} W" if mid_ci_lo is not None else "—")

    st.divider()

    # ── Projection evolution + game log ──────────────────────────────────────
    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.subheader("Win Projection Evolution")
        # Only show weeks 1–18 (before week 19 is after season ends — redundant)
        td = proj_df[(proj_df["team"] == team) & (proj_df["proj_week"] <= 18)].sort_values("proj_week")
        r, g_c, b = int(tc[1:3], 16), int(tc[3:5], 16), int(tc[5:7], 16)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pd.concat([td["proj_week"], td["proj_week"][::-1]]),
            y=pd.concat([td["ci_hi_80"], td["ci_lo_80"][::-1]]),
            fill="toself", fillcolor=f"rgba({r},{g_c},{b},0.12)",
            line=dict(width=0), name="80% CI", hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=td["proj_week"], y=td["proj_total_wins"], mode="lines+markers",
            name="Projected wins", line=dict(color=tc, width=3), marker=dict(size=6, color=tc),
            hovertemplate="<b>Before Wk %{x}</b><br>Projected: %{y:.1f} W<extra></extra>"))
        if actual_w is not None:
            fig.add_hline(y=actual_w, line_dash="dot", line_color=SLATE, line_width=2,
                          annotation_text=f"Actual: {actual_w} W",
                          annotation_font_color=SLATE, annotation_position="top right")
        # Mark preseason and midseason snapshots
        fig.add_vline(x=PRESEASON_WK, line_dash="dot", line_color=GREEN, line_width=1.2,
                      annotation_text="Preseason", annotation_font_color=GREEN,
                      annotation_position="top left")
        fig.add_vline(x=MIDSEASON_WK, line_dash="dot", line_color=PURPLE, line_width=1.2,
                      annotation_text="Midseason", annotation_font_color=PURPLE,
                      annotation_position="top left")
        fig.update_layout(
            xaxis=dict(title="Projection made before week...", dtick=2, range=[1, 18]),
            yaxis=dict(title="Projected wins", range=[0, 18]),
            showlegend=True, legend=dict(x=0.01, y=0.99))
        st.plotly_chart(clean(fig, h=360), use_container_width=True)

    with col_b:
        st.subheader("2025 Game Log")
        tg = sched[sched["team"] == team].sort_values("week").copy()
        tg["Result"] = tg.apply(
            lambda r: f"{'W' if r['win']==1 else 'L'} {int(r['team_score'])}-{int(r['opp_score'])}"
            if pd.notna(r.get("win")) and pd.notna(r.get("team_score")) else "—", axis=1)
        tg["Win Prob"] = (tg["win_prob_noveg"] * 100).round(1).astype(str) + "%"
        tg["Location"] = tg["is_home"].map({1: "Home", 0: "Away"})
        disp = tg[["week", "opponent", "Location", "Result", "Win Prob"]].rename(
            columns={"week": "Wk", "opponent": "Opp"})
        st.dataframe(disp, hide_index=True, height=340,
                     column_config={
                         "Wk":       st.column_config.NumberColumn(width="small"),
                         "Location": st.column_config.TextColumn(width="small"),
                         "Win Prob": st.column_config.TextColumn(width="medium"),
                     })

    st.divider()

    # ── ELO + rolling metrics ─────────────────────────────────────────────────
    col_c, col_d = st.columns(2)
    with col_c:
        st.subheader("ELO Ratings Over Season")
        elo_t = elo_2025[elo_2025["team"] == team].sort_values("week")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=elo_t["week"], y=elo_t["team_off_elo"], mode="lines+markers",
            name="Offensive ELO", line=dict(color=tc, width=2.5), marker=dict(size=5)))
        fig2.add_trace(go.Scatter(x=elo_t["week"], y=elo_t["team_def_elo"], mode="lines+markers",
            name="Defensive ELO", line=dict(color=AMBER, width=2.5, dash="dash"), marker=dict(size=5)))
        fig2.add_hline(y=1500, line_dash="dot", line_color=SLATE, line_width=1,
                       annotation_text="League avg (1500)", annotation_font_color=SLATE)
        fig2.update_layout(xaxis=dict(title="Week", dtick=2),
                           yaxis=dict(title="ELO Rating"),
                           showlegend=True, legend=dict(x=0.01, y=0.01,
                               font=dict(color=TEXT_DARK)))
        st.plotly_chart(clean(fig2, h=300), use_container_width=True)

    with col_d:
        st.subheader("Key Efficiency Metrics")
        tr = team_roll[team_roll["team"] == team].sort_values("week")
        metric_sel = st.selectbox("Metric", [
            "net_ypa", "def_yds_per_tgt", "pressure_rate", "yds_per_carry"], key="metric_sel",
            format_func=lambda x: {
                "net_ypa":          "Net Yards/Attempt (Offense)",
                "def_yds_per_tgt":  "Yards/Target Allowed (Defense)",
                "pressure_rate":    "Pressure Rate (Offense)",
                "yds_per_carry":    "Yards/Carry (Rush)",
            }[x])
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=tr["week"], y=tr[metric_sel], mode="lines+markers",
            name=metric_sel, line=dict(color=tc, width=2.5), marker=dict(size=6),
            hovertemplate="Week %{x}<br>%{y:.3f}<extra></extra>"))
        fig3.update_layout(xaxis=dict(title="Week", dtick=2),
                           yaxis=dict(title="Value"), showlegend=False)
        st.plotly_chart(clean(fig3, h=260), use_container_width=True)

    # ── Schedule win probability ──────────────────────────────────────────────
    st.subheader("Schedule Win Probability by Week")
    tg2 = sched[sched["team"] == team].sort_values("week").copy()
    bar_colors = []
    for _, r in tg2.iterrows():
        if r["win"] == 1:
            bar_colors.append(WIN_BLUE)
        elif r["win"] == 0:
            bar_colors.append(LOSS_RED)
        else:
            bar_colors.append(SLATE)
    fig4 = go.Figure()
    fig4.add_trace(go.Bar(
        x=tg2["week"], y=tg2["win_prob_noveg"],
        marker_color=bar_colors, marker_line_width=0,
        text=tg2["opponent"], textposition="outside", textfont=dict(size=9, color=TEXT_DARK),
        hovertemplate="<b>Wk %{x} vs %{text}</b><br>Win Prob: %{y:.1%}<extra></extra>"))
    fig4.add_hline(y=0.5, line_dash="dot", line_color=SLATE, line_width=1.5)
    fig4.update_layout(
        xaxis=dict(title="Week", dtick=1),
        yaxis=dict(title="Model Win Probability", tickformat=".0%", range=[0, 1.15]),
        showlegend=False)
    st.plotly_chart(clean(fig4, h=320), use_container_width=True)
    st.caption(f"Bar color: Win (blue) · Loss (red)  |  Label = Opponent")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Comparison":
    st.title("Model Performance Comparison")
    st.markdown("All models trained on **2018–2024 Weeks 1–17** and evaluated on **2025 Weeks 1–17**.")

    tab1, tab2, tab3 = st.tabs(["Classifier Accuracy", "Regressor Accuracy", "Feature Importance"])

    with tab1:
        col_nv, col_v = st.columns(2)

        def build_acc_bar(results_dict, title, color):
            names = list(results_dict.keys())
            accs  = [v["test_acc"] for v in results_dict.values()]
            train = [v["train_acc"] for v in results_dict.values()]
            fig = go.Figure()
            fig.add_trace(go.Bar(name="Train Acc", x=names, y=train,
                marker_color=SLATE, opacity=0.5, marker_line_width=0,
                text=[f"{v:.3f}" for v in train], textposition="outside",
                textfont=dict(size=9, color=SLATE)))
            fig.add_trace(go.Bar(name="Test Acc", x=names, y=accs,
                marker_color=color, marker_line_width=0,
                text=[f"{v:.3f}" for v in accs], textposition="outside",
                textfont=dict(size=10, color=TEXT_DARK)))
            fig.add_hline(y=vegas_acc, line_dash="dash", line_color=RED, line_width=2,
                          annotation_text=f"Vegas {vegas_acc:.3f}",
                          annotation_font_color=RED, annotation_position="top right")
            fig.update_layout(barmode="group", title=dict(text=title, font=dict(color=TEXT_DARK)),
                yaxis=dict(title="Accuracy", tickformat=".0%",
                    range=[min(accs+train+[vegas_acc])-0.05, max(accs+train+[vegas_acc])+0.09]),
                showlegend=True, legend=dict(x=0.01, y=0.99, font=dict(color=TEXT_DARK)),
                xaxis=dict(tickangle=-20))
            return clean(fig, h=420)

        with col_nv:
            st.plotly_chart(build_acc_bar(model_res["cls_noveg"], "Without Vegas Lines", BLUE),
                            use_container_width=True)
        with col_v:
            st.plotly_chart(build_acc_bar(model_res["cls_vegas"], "With Vegas Lines", TEAL),
                            use_container_width=True)

        st.subheader("Full Metrics Table — Classifiers")
        rows = []
        for cat, label in [("cls_noveg", "No Vegas"), ("cls_vegas", "Vegas")]:
            for name, res in model_res[cat].items():
                rows.append({
                    "Model": name, "Features": label,
                    "Test Acc": f"{res['test_acc']:.1%}",
                    "Train Acc": f"{res['train_acc']:.1%}",
                    "Gap": f"{res['train_acc']-res['test_acc']:+.3f}",
                    "AUC": f"{res['auc']:.4f}" if res.get("auc") else "—",
                    "Brier": f"{res['brier']:.4f}" if res.get("brier") else "—",
                    "vs Vegas": f"{res['vs_vegas']:+.1%}",
                })
        rows.append({"Model": "Vegas Spread Baseline", "Features": "N/A",
                     "Test Acc": f"{vegas_acc:.1%}", "Train Acc": "—", "Gap": "—",
                     "AUC": "—", "Brier": "—", "vs Vegas": "—"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    with tab2:
        col_nv2, col_v2 = st.columns(2)

        def build_reg_bar(results_dict, title, color):
            names = list(results_dict.keys())
            accs  = [v["test_acc"] for v in results_dict.values()]
            maes  = [v["mae"] for v in results_dict.values()]
            fig = make_subplots(rows=1, cols=2,
                subplot_titles=["Win/Loss Accuracy (from margin)", "MAE (points)"])
            fig.add_trace(go.Bar(x=names, y=accs, marker_color=color, marker_line_width=0,
                text=[f"{v:.3f}" for v in accs], textposition="outside",
                textfont=dict(size=9, color=TEXT_DARK), showlegend=False), row=1, col=1)
            fig.add_hline(y=vegas_acc, line_dash="dash", line_color=RED, line_width=1.5, row=1, col=1)
            fig.add_trace(go.Bar(x=names, y=maes, marker_color=AMBER, marker_line_width=0,
                text=[f"{v:.2f}" for v in maes], textposition="outside",
                textfont=dict(size=9, color=TEXT_DARK), showlegend=False), row=1, col=2)
            fig.update_xaxes(tickangle=-20, tickfont=dict(size=9, color=SLATE))
            fig.update_yaxes(tickformat=".0%", tickfont=dict(color=SLATE),
                range=[min(accs+[vegas_acc])-0.04, max(accs+[vegas_acc])+0.08], row=1, col=1)
            fig.update_yaxes(tickfont=dict(color=SLATE), row=1, col=2)
            fig.update_layout(title=dict(text=title, font=dict(color=TEXT_DARK)),
                height=400, margin=dict(l=48, r=24, t=64, b=80),
                paper_bgcolor="white", plot_bgcolor="white",
                font=dict(family="Inter, system-ui, sans-serif", size=12, color=TEXT_DARK))
            return fig

        with col_nv2:
            st.plotly_chart(build_reg_bar(model_res["reg_noveg"], "Without Vegas Lines", BLUE),
                            use_container_width=True)
        with col_v2:
            st.plotly_chart(build_reg_bar(model_res["reg_vegas"], "With Vegas Lines", TEAL),
                            use_container_width=True)

        st.subheader("Full Metrics Table — Regressors")
        rows2 = []
        for cat, label in [("reg_noveg", "No Vegas"), ("reg_vegas", "Vegas")]:
            for name, res in model_res[cat].items():
                rows2.append({
                    "Model": name, "Features": label,
                    "Test Acc": f"{res['test_acc']:.1%}",
                    "Train Acc": f"{res['train_acc']:.1%}",
                    "MAE": f"{res['mae']:.2f} pts",
                    "RMSE": f"{res['rmse']:.2f} pts",
                    "vs Vegas": f"{res['vs_vegas']:+.1%}",
                })
        st.dataframe(pd.DataFrame(rows2), hide_index=True, use_container_width=True)

    with tab3:
        st.subheader("Feature Importance — XGBoost (No Vegas)")
        fi_sorted = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)
        fi_df = pd.DataFrame(fi_sorted, columns=["Feature", "Importance"])
        threshold = fi_df["Importance"].quantile(0.75)
        fi_df["color"] = fi_df["Importance"].apply(lambda v: BEARS if v >= threshold else SLATE)
        fig_fi = go.Figure(go.Bar(
            x=fi_df["Importance"], y=fi_df["Feature"], orientation="h",
            marker_color=fi_df["color"], marker_line_width=0,
            text=fi_df["Importance"].round(4).astype(str), textposition="outside",
            textfont=dict(size=9, color=TEXT_DARK)))
        fig_fi.update_layout(
            xaxis=dict(title="Feature Importance (gain)", tickfont=dict(color=SLATE)),
            yaxis=dict(autorange="reversed", tickfont=dict(color=TEXT_DARK)))
        st.plotly_chart(clean(fig_fi, h=600, margins=dict(l=220, r=80, t=32, b=48)),
                        use_container_width=True)

        st.subheader("Linear vs. Non-Linear Model Gap")
        linear_avg    = np.mean([model_res["cls_noveg"]["Logistic Regression"]["test_acc"],
                                  model_res["cls_noveg"]["Ridge Classifier"]["test_acc"]])
        nonlinear_avg = np.mean([model_res["cls_noveg"]["XGBoost"]["test_acc"],
                                  model_res["cls_noveg"]["Random Forest"]["test_acc"]])
        nn_acc        = model_res["cls_noveg"]["Neural Network"]["test_acc"]
        fig_gap = go.Figure(go.Bar(
            x=["Linear (Logistic / Ridge)", "Non-linear Trees (XGBoost / RF)", "Neural Network (MLP)"],
            y=[linear_avg, nonlinear_avg, nn_acc],
            marker_color=[AMBER, BLUE, GREEN], marker_line_width=0,
            text=[f"{v:.3f}" for v in [linear_avg, nonlinear_avg, nn_acc]],
            textposition="outside", textfont=dict(size=13, color=TEXT_DARK)))
        fig_gap.add_hline(y=vegas_acc, line_dash="dot", line_color=RED, line_width=2,
                          annotation_text=f"Vegas {vegas_acc:.3f}", annotation_font_color=RED)
        fig_gap.update_layout(
            yaxis=dict(title="Test accuracy", tickformat=".0%",
                range=[min(linear_avg, nonlinear_avg, nn_acc, vegas_acc)-0.04,
                       max(linear_avg, nonlinear_avg, nn_acc, vegas_acc)+0.08]),
            showlegend=False)
        st.plotly_chart(clean(fig_gap, h=380), use_container_width=True)
        st.info("**Interpretation:** The gap between linear and tree-based models reflects how much "
                "interaction structure exists in the data. A larger gap means non-linear patterns are "
                "real and the added complexity of tree models is justified.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — LEAGUE PROJECTIONS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🗺️ League Projections":
    st.title("League-Wide 2025 Projections")

    tab_a, tab_b, tab_c = st.tabs(["All 32 Teams", "By Division", "Week-by-Week Table"])

    with tab_a:
        snap_week = MIDSEASON_WK

        cur = (proj_df[proj_df["proj_week"] == snap_week]
               .sort_values("proj_total_wins", ascending=True).reset_index(drop=True))
        cur["actual_wins"] = cur["team"].map(final_wins.set_index("team")["actual_wins_2025"])
        colors_bar = [TEAM_COLORS.get(t, BLUE) for t in cur["team"]]

        fig_all = go.Figure()
        for _, row in cur.iterrows():
            fig_all.add_trace(go.Scatter(
                x=[row["proj_total_wins"], row["proj_total_wins"]],
                y=[row["ci_lo_80"], row["ci_hi_80"]],
                mode="lines", line=dict(color=SLATE, width=7), opacity=0.3,
                showlegend=False, hoverinfo="skip"))
        fig_all.add_trace(go.Scatter(
            x=cur["proj_total_wins"], y=cur["team"],
            mode="markers+text",
            text=[f"{v:.1f}" for v in cur["proj_total_wins"]],
            textposition="middle right", textfont=dict(size=9, color=TEXT_DARK),
            marker=dict(size=11, color=colors_bar, line=dict(width=1.5, color="white")),
            hovertemplate="<b>%{y}</b><br>Projected: %{x:.1f} W<br>"
                          "80% CI: %{customdata[0]:.0f}–%{customdata[1]:.0f}<extra></extra>",
            customdata=cur[["ci_lo_80", "ci_hi_80"]].values,
            showlegend=False))
        fig_all.add_trace(go.Scatter(
            x=cur["actual_wins"], y=cur["team"], mode="markers",
            marker=dict(size=9, symbol="diamond", color=TEXT_DARK, opacity=0.55),
            name="Actual wins",
            hovertemplate="<b>%{y}</b><br>Actual: %{x:.0f} W<extra></extra>"))
        fig_all.update_layout(
            xaxis=dict(title="Wins", range=[0, 20], tickfont=dict(color=SLATE)),
            yaxis=dict(tickfont=dict(size=10, color=TEXT_DARK)),
            showlegend=True, legend=dict(x=0.75, y=0.02, font=dict(color=TEXT_DARK)),
            title=dict(text="Midseason (before Week 10) — grey bar = 80% CI  ·  diamond = actual final wins",
                       font=dict(color=TEXT_DARK)))
        st.plotly_chart(clean(fig_all, h=max(520, len(cur)*16),
                               margins=dict(l=56, r=100, t=48, b=48)),
                        use_container_width=True)

        # Projection vs actual scatter
        st.subheader("Projection vs Actual Wins")
        snap_proj = proj_df[proj_df["proj_week"] == snap_week].copy()
        snap_proj["actual"] = snap_proj["team"].map(final_wins.set_index("team")["actual_wins_2025"])
        mae_val = (snap_proj["proj_total_wins"] - snap_proj["actual"]).abs().mean()
        fig_scat = go.Figure()
        fig_scat.add_trace(go.Scatter(x=[2, 17], y=[2, 17], mode="lines", hoverinfo="skip",
            line=dict(color=SLATE, dash="dot", width=1.5), showlegend=False))
        for _, row in snap_proj.dropna(subset=["actual"]).iterrows():
            fig_scat.add_trace(go.Scatter(
                x=[row["proj_total_wins"]], y=[row["actual"]],
                mode="markers+text", text=[row["team"]],
                textposition="top center", textfont=dict(size=9, color=SLATE),
                marker=dict(size=10, color=TEAM_COLORS.get(row["team"], BLUE),
                            line=dict(width=1.5, color="white")),
                showlegend=False,
                hovertemplate=f"<b>{row['team']}</b><br>Proj: {row['proj_total_wins']:.1f}"
                              f"<br>Actual: {int(row['actual'])}<extra></extra>"))
        fig_scat.update_layout(
            title=dict(text=f"Midseason MAE = {mae_val:.2f} wins", font=dict(color=TEXT_DARK)),
            xaxis=dict(title="Projected wins", range=[3, 16], tickfont=dict(color=SLATE)),
            yaxis=dict(title="Actual 2025 wins", range=[1, 18], tickfont=dict(color=SLATE)),
            showlegend=False)
        st.plotly_chart(clean(fig_scat, h=440), use_container_width=True)

    with tab_b:
        st.subheader("Division Breakdown")
        # Use toggle selection here too
        snap_label_b = st.radio("Snapshot", ["Preseason", "Midseason"], horizontal=True, key="div_snap")
        snap_week_b  = PRESEASON_WK if snap_label_b == "Preseason" else MIDSEASON_WK
        snap_all = proj_df[proj_df["proj_week"] == snap_week_b].copy()
        snap_all["actual"] = snap_all["team"].map(final_wins.set_index("team")["actual_wins_2025"])

        conf_sel = st.radio("Conference", ["AFC", "NFC", "Both"], horizontal=True)
        divs_to_show = {d: ts for d, ts in DIVISIONS.items()
                        if conf_sel == "Both" or d.startswith(conf_sel)}
        div_list = list(divs_to_show.items())
        nrows = (len(div_list) + 1) // 2
        for row_i in range(nrows):
            cols = st.columns(2)
            for col_i in range(2):
                idx = row_i * 2 + col_i
                if idx >= len(div_list):
                    break
                div_name, div_teams = div_list[idx]
                sub = snap_all[snap_all["team"].isin(div_teams)].sort_values("proj_total_wins", ascending=False)
                with cols[col_i]:
                    fig_div = go.Figure()
                    fig_div.add_trace(go.Bar(
                        x=sub["team"], y=sub["proj_total_wins"],
                        name="Projected",
                        marker_color=[TEAM_COLORS.get(t, BLUE) for t in sub["team"]],
                        marker_line_width=0,
                        text=[f"{v:.1f}" for v in sub["proj_total_wins"]],
                        textposition="outside", textfont=dict(size=11, color=TEXT_DARK)))
                    fig_div.add_trace(go.Scatter(
                        x=sub["team"], y=sub["actual"], mode="markers",
                        marker=dict(size=12, symbol="diamond", color=TEXT_DARK, opacity=0.6),
                        name="Actual", showlegend=False,
                        hovertemplate="%{x}: %{y:.0f} actual<extra></extra>"))
                    fig_div.update_layout(
                        title=dict(text=div_name, font=dict(color=TEXT_DARK)),
                        showlegend=False,
                        yaxis=dict(title="Wins", range=[0, 18], tickfont=dict(color=SLATE)),
                        xaxis=dict(title=None, tickfont=dict(color=TEXT_DARK)))
                    st.plotly_chart(clean(fig_div, h=280), use_container_width=True)

    with tab_c:
        st.subheader("Week-by-Week Projection Table")
        # Only show weeks 1-18 (week 19 = after full season, same as actual = not useful)
        piv = (proj_df[proj_df["proj_week"] <= 18]
               .pivot(index="team", columns="proj_week", values="proj_total_wins")
               .round(1))
        piv.columns = ["Pre"] + [f"Wk{w}" for w in range(2, len(piv.columns) + 1)]
        piv["Actual"] = piv.index.map(final_wins.set_index("team")["actual_wins_2025"])
        piv = piv.sort_values("Pre", ascending=False)
        st.dataframe(piv, use_container_width=True,
                     column_config={"Actual": st.column_config.NumberColumn(format="%.0f")})


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — CAP SPENDING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💰 Cap Spending":
    st.title("Cap Spending Analysis")
    st.markdown(
        "> **Scope note:** Cap data from Over The Cap (OTC) is used as a *post-model* descriptive "
        "analysis only — it has no influence on any prediction. Question explored: what do "
        "top-projected teams invest in differently?")

    EXCL_POS = {"K", "P", "LS", "FB"}
    POS_GROUPS = {
        "QB":    ["QB"], "RB":    ["RB", "FB"], "OL":    ["OT", "OG", "C"],
        "WR/TE": ["WR", "TE"],  "DL":    ["EDGE", "IDL"], "LB":    ["LB"],
        "CB":    ["CB"],        "S":     ["S"],
    }
    SPIDER_ORDER = ["QB", "WR/TE", "OL", "RB", "S", "CB", "LB", "DL"]

    # Assign quadrants from preseason projections
    pre_all = proj_df[proj_df["proj_week"] == PRESEASON_WK].copy()
    pre_all = pre_all.sort_values("proj_total_wins", ascending=False).reset_index(drop=True)
    pre_all["rank"]     = pre_all.index + 1
    pre_all["quadrant"] = pd.cut(pre_all["rank"], bins=[0, 8, 16, 24, 32],
        labels=["Q1 (Top 8)", "Q2 (Upper-mid)", "Q3 (Lower-mid)", "Q4 (Bottom 8)"])
    pre_all["actual"]   = pre_all["team"].map(final_wins.set_index("team")["actual_wins_2025"])

    QUAD_LABELS = ["Q1 (Top 8)", "Q2 (Upper-mid)", "Q3 (Lower-mid)", "Q4 (Bottom 8)"]
    Q_COLORS    = {"Q1 (Top 8)": BEARS, "Q2 (Upper-mid)": BLUE,
                   "Q3 (Lower-mid)": AMBER, "Q4 (Bottom 8)": RED}

    tab_sp1, tab_sp2, tab_sp3, tab_sp4 = st.tabs([
        "Team Spending", "Quadrant Patterns", "Radar Charts", "Player Contracts"])

    with tab_sp1:
        team_sp  = st.selectbox("Select team", ALL_TEAMS, index=ALL_TEAMS.index("CHI"), key="cap_team")
        tq       = pre_all[pre_all["team"] == team_sp]["quadrant"].values
        team_quad  = str(tq[0]) if len(tq) else "—"
        team_rank  = pre_all[pre_all["team"] == team_sp]["rank"].values
        team_proj  = pre_all[pre_all["team"] == team_sp]["proj_total_wins"].values

        c1, c2, c3 = st.columns(3)
        c1.metric("Win Quadrant (Preseason)", team_quad)
        c2.metric("League Rank", f"#{int(team_rank[0])}" if len(team_rank) else "—")
        c3.metric("Preseason Projection", f"{team_proj[0]:.1f} W" if len(team_proj) else "—")

        t_spend  = otc_sp[otc_sp["team"] == team_sp].copy()
        lg_avg   = otc_sp.groupby("position")["cap_pct"].mean()
        t_spend["delta"] = t_spend.apply(lambda r: r["cap_pct"] - lg_avg.get(r["position"], 0), axis=1)
        t_spend  = t_spend[~t_spend["position"].isin(EXCL_POS)].sort_values("cap_pct", ascending=False)
        tc_sp    = TEAM_COLORS.get(team_sp, BLUE)

        col_aa, col_bb = st.columns(2)
        with col_aa:
            st.subheader("Positional Cap %")
            fig_sp = go.Figure(go.Bar(
                x=t_spend["position"], y=t_spend["cap_pct"],
                marker_color=tc_sp, marker_line_width=0,
                text=t_spend["cap_pct"].round(1).astype(str) + "%",
                textposition="outside", textfont=dict(size=10, color=TEXT_DARK)))
            fig_sp.update_layout(
                xaxis=dict(title=None, tickfont=dict(color=TEXT_DARK)),
                yaxis=dict(title="Cap %", ticksuffix="%", tickfont=dict(color=SLATE)))
            st.plotly_chart(clean(fig_sp, h=320), use_container_width=True)

        with col_bb:
            st.subheader("Delta vs League Average")
            colors_d = ["#0F6E56" if d >= 0 else RED for d in t_spend["delta"]]
            fig_delt = go.Figure(go.Bar(
                x=t_spend["position"], y=t_spend["delta"],
                marker_color=colors_d, marker_line_width=0,
                text=[f"{d:+.1f}%" for d in t_spend["delta"]],
                textposition="outside", textfont=dict(size=10, color=TEXT_DARK)))
            fig_delt.add_hline(y=0, line_color=SLATE, line_width=1)
            fig_delt.update_layout(
                xaxis=dict(title=None, tickfont=dict(color=TEXT_DARK)),
                yaxis=dict(title="Delta vs League Avg", ticksuffix="%", tickfont=dict(color=SLATE)))
            st.plotly_chart(clean(fig_delt, h=320), use_container_width=True)

        st.subheader("Cap Spending Details")
        disp_sp = t_spend[["position", "total_cap_spend", "n_players", "avg_cap_per_player",
                             "cap_pct", "delta"]].copy()
        disp_sp.columns = ["Position", "Total Spend", "# Players", "Avg/Player", "Cap %", "vs League Avg"]
        disp_sp["Total Spend"]   = disp_sp["Total Spend"].apply(lambda x: f"${x/1e6:.1f}M")
        disp_sp["Avg/Player"]    = disp_sp["Avg/Player"].apply(lambda x: f"${x/1e6:.1f}M")
        disp_sp["Cap %"]         = disp_sp["Cap %"].apply(lambda x: f"{x:.1f}%")
        disp_sp["vs League Avg"] = disp_sp["vs League Avg"].apply(lambda x: f"{x:+.2f}%")
        st.dataframe(disp_sp, hide_index=True, use_container_width=True)

    with tab_sp2:
        st.subheader("Positional Spending by Win Quadrant")
        lg_avg_pos = otc_sp.groupby("position")["cap_pct"].mean()
        all_pos = [p for p in sorted(otc_sp["position"].unique()) if p not in EXCL_POS]
        view = st.radio("View", ["Raw Cap %", "Delta vs League Average"], horizontal=True)

        fig_quad = make_subplots(rows=2, cols=2, subplot_titles=QUAD_LABELS,
                                  vertical_spacing=0.14, horizontal_spacing=0.10)
        for qi, q in enumerate(QUAD_LABELS):
            row, col = (qi // 2) + 1, (qi % 2) + 1
            teams_q  = pre_all[pre_all["quadrant"] == q]["team"].tolist()
            q_spend  = otc_sp[otc_sp["team"].isin(teams_q) & ~otc_sp["position"].isin(EXCL_POS)]
            pos_avg  = q_spend.groupby("position")["cap_pct"].mean().reindex(all_pos).fillna(0)
            if view == "Raw Cap %":
                vals = pos_avg.values
                bar_clr = [Q_COLORS[q]] * len(vals)
                fmt = ".1f"
            else:
                vals = (pos_avg - lg_avg_pos.reindex(all_pos).fillna(0)).values
                bar_clr = ["#0F6E56" if v >= 0 else RED for v in vals]
                fmt = "+.1f"
            fig_quad.add_trace(go.Bar(
                x=all_pos, y=vals, marker_color=bar_clr, marker_line_width=0,
                text=[f"{v:{fmt}}" for v in vals], textposition="outside",
                textfont=dict(size=8, color=TEXT_DARK), showlegend=False), row=row, col=col)
            if view == "Delta vs League Average":
                fig_quad.add_hline(y=0, line_color="rgba(0,0,0,0.2)", line_width=1, row=row, col=col)
        fig_quad.update_xaxes(tickangle=-40, tickfont=dict(size=9, color=TEXT_DARK))
        fig_quad.update_yaxes(ticksuffix="%", tickfont=dict(color=SLATE))
        fig_quad.update_layout(height=580, margin=dict(l=48, r=24, t=72, b=48),
            paper_bgcolor="white", plot_bgcolor="white",
            font=dict(family="Inter, system-ui, sans-serif", size=12, color=TEXT_DARK))
        st.plotly_chart(fig_quad, use_container_width=True)

        st.subheader(f"{team} vs Its Quadrant Peers")
        team_q_label = str(pre_all[pre_all["team"] == team]["quadrant"].values[0]) \
                       if len(pre_all[pre_all["team"] == team]) else None
        if team_q_label:
            peers     = pre_all[pre_all["quadrant"] == team_q_label]["team"].tolist()
            peer_sp   = otc_sp[otc_sp["team"].isin(peers) & ~otc_sp["position"].isin(EXCL_POS)]
            peer_avg  = peer_sp.groupby("position")["cap_pct"].mean()
            team_s    = otc_sp[(otc_sp["team"] == team) & ~otc_sp["position"].isin(EXCL_POS)]
            positions = sorted(set(peer_avg.index) | set(team_s["position"].values))
            fig_peer  = go.Figure()
            fig_peer.add_trace(go.Bar(name=team, x=positions,
                y=[team_s[team_s["position"] == p]["cap_pct"].sum() for p in positions],
                marker_color=TEAM_COLORS.get(team, BLUE), marker_line_width=0, opacity=0.9))
            fig_peer.add_trace(go.Bar(name=f"{team_q_label} avg", x=positions,
                y=[peer_avg.get(p, 0) for p in positions],
                marker_color=SLATE, marker_line_width=0, opacity=0.55))
            fig_peer.update_layout(barmode="group",
                xaxis=dict(title=None, tickfont=dict(color=TEXT_DARK)),
                yaxis=dict(title="Cap %", ticksuffix="%", tickfont=dict(color=SLATE)),
                showlegend=True, legend=dict(font=dict(color=TEXT_DARK)))
            st.plotly_chart(clean(fig_peer, h=340), use_container_width=True)

    with tab_sp3:
        st.subheader("Spending Profile Radars by Quadrant")
        lg_grp_avg = {}
        for grp, positions in POS_GROUPS.items():
            sub = otc_sp[otc_sp["position"].isin(positions)]
            per_team = sub.groupby("team")["cap_pct"].sum()
            lg_grp_avg[grp] = per_team.mean()

        Q_FILL = {
            "Q1 (Top 8)":     "rgba(11,22,42,0.15)",
            "Q2 (Upper-mid)": "rgba(37,99,235,0.15)",
            "Q3 (Lower-mid)": "rgba(217,119,6,0.15)",
            "Q4 (Bottom 8)":  "rgba(220,38,38,0.15)",
        }
        fig_spider = make_subplots(rows=2, cols=2,
            specs=[[{"type": "polar"}, {"type": "polar"}],
                   [{"type": "polar"}, {"type": "polar"}]],
            subplot_titles=QUAD_LABELS, vertical_spacing=0.12, horizontal_spacing=0.08)
        for qi, q in enumerate(QUAD_LABELS):
            row, col = (qi // 2) + 1, (qi % 2) + 1
            teams_q  = pre_all[pre_all["quadrant"] == q]["team"].tolist()
            q_spend  = otc_sp[otc_sp["team"].isin(teams_q)]
            grp_vals = {}
            for grp, positions in POS_GROUPS.items():
                sub      = q_spend[q_spend["position"].isin(positions)]
                per_team = sub.groupby("team")["cap_pct"].sum()
                grp_vals[grp] = per_team.mean() if len(per_team) else 0
            r_vals   = [grp_vals.get(g, 0) for g in SPIDER_ORDER]
            r_closed = r_vals + [r_vals[0]]
            t_closed = SPIDER_ORDER + [SPIDER_ORDER[0]]
            fig_spider.add_trace(go.Scatterpolar(
                r=r_closed, theta=t_closed, fill="toself",
                fillcolor=Q_FILL[q], line=dict(color=Q_COLORS[q], width=2.2),
                name=q, showlegend=False), row=row, col=col)
            lg_r = [lg_grp_avg.get(g, 0) for g in SPIDER_ORDER]
            fig_spider.add_trace(go.Scatterpolar(
                r=lg_r + [lg_r[0]], theta=t_closed, fill="none",
                line=dict(color="rgba(80,80,80,0.45)", width=1.2, dash="dot"),
                showlegend=False), row=row, col=col)
            pk = "polar" if qi == 0 else f"polar{qi + 1}"
            fig_spider.update_layout(**{pk: dict(
                radialaxis=dict(visible=True, range=[0, 30],
                                tickfont=dict(size=8, color=SLATE),
                                gridcolor="rgba(100,100,100,0.2)"),
                angularaxis=dict(tickfont=dict(size=11, color=TEXT_DARK),
                                 gridcolor="rgba(100,100,100,0.2)"),
                bgcolor="white")})
        fig_spider.update_layout(
            height=640, margin=dict(l=40, r=40, t=80, b=40),
            paper_bgcolor="white",
            font=dict(family="Inter, system-ui, sans-serif", size=12, color=TEXT_DARK))
        st.plotly_chart(fig_spider, use_container_width=True)
        st.caption("Dotted circle = league average. Each axis = positional group cap %.")

    with tab_sp4:
        st.subheader("Player Contract Details")
        team_ct = st.selectbox("Team", ALL_TEAMS, index=ALL_TEAMS.index("CHI"), key="ct_team")
        ct_data = otc_ct[otc_ct["team"] == team_ct].copy()
        if len(ct_data):
            ct_data = ct_data.sort_values("cap_number", ascending=False)
            ct_disp = ct_data[["player", "position", "cap_number", "base_salary",
                                 "signing_bonus", "guaranteed"]].copy()
            for col in ["cap_number", "base_salary", "signing_bonus", "guaranteed"]:
                ct_disp[col] = ct_disp[col].apply(
                    lambda x: f"${x/1e6:.2f}M" if pd.notna(x) and x > 0 else "—")
            ct_disp.columns = ["Player", "Pos", "Cap #", "Base", "Signing Bonus", "Guaranteed"]
            st.dataframe(ct_disp, hide_index=True, use_container_width=True)
        else:
            st.info("No contract data available for this team.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — EDA & METHODOLOGY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔬 EDA & Methodology":
    st.title("EDA & Methodology")

    tab_e1, tab_e2, tab_e3 = st.tabs(["Key EDA Charts", "Leakage Audit", "Methodology Summary"])

    with tab_e1:
        st.subheader("Home Field Advantage by Season")
        fig_hfa = go.Figure()
        fig_hfa.add_trace(go.Scatter(
            x=hfa["season"], y=hfa["home_win_pct"], mode="lines+markers",
            line=dict(color=BLUE, width=2.5), marker=dict(size=7, color=BLUE),
            fill="tozeroy", fillcolor="rgba(37,99,235,0.08)"))
        fig_hfa.add_hline(y=hfa["home_win_pct"].mean(), line_dash="dot",
            line_color=SLATE,
            annotation_text=f"Avg {hfa['home_win_pct'].mean():.1%}",
            annotation_font_color=SLATE, annotation_position="top right")
        hfa_2020 = hfa[hfa["season"] == 2020]["home_win_pct"].values
        if len(hfa_2020):
            fig_hfa.add_annotation(x=2020, y=float(hfa_2020[0]),
                text="No fans (COVID)", showarrow=True, arrowhead=2,
                arrowcolor=RED, font=dict(color=RED, size=11), ax=40, ay=-32)
        fig_hfa.update_layout(
            xaxis=dict(title="Season", dtick=1, tickangle=-45, tickfont=dict(color=SLATE)),
            yaxis=dict(title="Home win rate", tickformat=".0%", tickfont=dict(color=SLATE)))
        st.plotly_chart(clean(fig_hfa, h=340), use_container_width=True)

        st.subheader("ELO System — Yards Per Play Signal")
        st.markdown("""
The ELO system separates **offensive and defensive ratings** per team.

**Key design choices:**
- **Signal:** Net yards per play (passing + rushing, net of sack yards) — more informative than raw wins
- **Threshold:** Season-to-date rolling median YPP to classify each game as above/below average
- **Regression:** Ratings regress 1/3 back to 1500 between seasons
- **Week 1 seed:** Historical fallback of 5.5 net YPP before any season data exists

An offense finishing above the median earns an ELO "win" for their offensive rating; the opponent's defensive rating is updated symmetrically.
        """)

        st.subheader("Feature Engineering Overview")
        feat_groups = {
            "ELO (5 features)": "Off ELO, Def ELO, total ELO diff + matchup differentials",
            "Rolling Efficiency (13)": "Net YPA, 3rd-down rate, pressure rate, def YPP allowed, etc.",
            "Matchup Differentials (8)": "Team metric minus opponent metric (e.g., YPA diff, TO diff)",
            "Recency EWM (6)": "4-game exponential weighted moving avg for key metrics",
            "Context (8)": "Is home, rest days, week, division game, playoff stakes flags",
            "SOS-Adjusted (2)": "Net YPA & def yards adjusted for opponent quality faced",
            "Vegas (2 — augmented only)": "Spread line, total line",
        }
        rows_fg = [{"Group": k, "Description": v} for k, v in feat_groups.items()]
        st.dataframe(pd.DataFrame(rows_fg), hide_index=True, use_container_width=True)

    with tab_e2:
        st.subheader("Data Leakage Audit")
        st.markdown("Every potential leakage vector was identified and resolved before reporting accuracy.")
        audit = [
            ("Rolling features", "Using game N stats to predict game N",
             "Fixed", ".shift(1) on every expanding mean; Week 1 seeded from prior season"),
            ("SOS adjustment", "Normalising game N by opponent's full-season average",
             "Fixed", "Pre-game rolling opponent quality; Week 1 seeded from prior season"),
            ("ELO update", "Recording post-game ELO as pre-game feature",
             "Fixed", "ELO snapshot taken before update in every game loop"),
            ("Playoff stakes flags", "Using final clinch/elimination status",
             "Fixed", "True pre-game snapshot built by processing games in chronological order"),
            ("Train/test contamination", "Imputing test NAs with test median",
             "Fixed", "fillna uses train-split median only"),
            ("Target in features", "point_diff, team_score, win accessible in feature table",
             "Fixed", "None appear in FEATURES or FEATURES_VEGAS lists"),
            ("Vegas lines in base model", "spread_line_team leaking into base model",
             "Fixed", "Excluded from FEATURES; only in FEATURES_VEGAS"),
            ("XGBoost early stopping", "eval_set using 2025 test set",
             "Fixed", "OOT slice (2023-2024 W1-17) used for early stopping; 2025 never seen"),
        ]
        df_audit = pd.DataFrame(audit, columns=["Risk", "How it would leak", "Status", "Resolution"])
        st.dataframe(df_audit, hide_index=True, use_container_width=True,
                     column_config={
                         "Status": st.column_config.TextColumn(width="small"),
                         "Resolution": st.column_config.TextColumn(width="large"),
                     })
        st.success("All leakage vectors resolved. Results represent genuine out-of-time performance on the full 2025 season.")

    with tab_e3:
        st.subheader("Methodology Summary")
        methodology = {
            "Data Source": "Pro Football Reference via nflverse (2018–2025)",
            "Training Window": "2018–2024, Weeks 1–17 only",
            "Test Set": "2025 Weeks 1–17 (272 games, full completed season)",
            "Week 18 Treatment": "Evaluated separately (motivational confound)",
            "Models — No Vegas": "XGBoost, Random Forest, Logistic Regression, Ridge Classifier, Neural Network (MLP)",
            "Models — Vegas": "Same 5 models with spread_line_team + total_line added",
            "Regressors": "Same 5 architectures predicting point differential, converted to win/loss",
            "Hyperparameter Tuning": "Walk-forward CV (4 folds: 2019–2022 hold-one-out)",
            "XGBoost Early Stopping": "OOT slice (2023-2024); 2025 test never touched during training",
            "ELO Signal": "Net yards per play (replaces first downs for efficiency purity)",
            "Uncertainty": "5,000 Monte Carlo simulations per team, 80% confidence interval on final wins",
            "Cap Analysis": "Post-model descriptive only (OTC 2025); no influence on predictions",
        }
        df_meth = pd.DataFrame(methodology.items(), columns=["Component", "Detail"])
        st.dataframe(df_meth, hide_index=True, use_container_width=True,
                     column_config={"Detail": st.column_config.TextColumn(width="large")})

        st.subheader("Known Limitations")
        for lim in [
            "Red zone efficiency stats unavailable at game level in PFR",
            "EPA per play (gold-standard efficiency metric) not available in PFR",
            "Clinch/elimination flags use a wins>=10/losses>=10 proxy; exact playoff math is more complex",
            "Walk-forward CV covers 4 folds (2019–2022); extending back requires pre-2018 advanced stats",
            "A Bayesian or Kalman filter ELO would produce uncertainty bands on ratings themselves",
            "Vegas spread is a high bar — it aggregates sharp bettor information this model cannot replicate without lines",
        ]:
            st.markdown(f"- {lim}")

        st.subheader("References")
        for ref in [
            "Pro Football Reference (pro-football-reference.com) — all model input data",
            "nflverse project (nflverse.com) — data pipeline and hosting",
            "FiveThirtyEight NFL ELO methodology — season-carryover and regression approach",
            "Over The Cap (overthecap.com) — 2025 positional cap spending (supplementary only)",
        ]:
            st.markdown(f"- {ref}")
