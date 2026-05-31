import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
import plotly.express as px
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
BEARS  = "#0B162A"
BEARS_ORANGE = "#E64100"
BLUE   = "#2563EB"
TEAL   = "#0D9488"
AMBER  = "#D97706"
RED    = "#DC2626"
SLATE  = "#94A3B8"
GREEN  = "#16A34A"
PURPLE = "#7C3AED"
BORDER = "#E2E8F0"

TEAM_COLORS = {
    "ARI":"#97233F","ATL":"#A71930","BAL":"#241773","BUF":"#00338D",
    "CAR":"#0085CA","CHI":BEARS_ORANGE,"CIN":"#FB4F14","CLE":"#FF3C00",
    "DAL":"#003594","DEN":"#FB4F14","DET":"#0076B6","GB":"#203731",
    "HOU":"#03202F","IND":"#002C5F","JAX":"#006778","KC":"#E31837",
    "LAC":"#0073CF","LAR":"#003594","LV":"#000000","MIA":"#008E97",
    "MIN":"#4F2683","NE":"#002244","NO":"#D3BC8D","NYG":"#0B2265",
    "NYJ":"#125740","PHI":"#004C54","PIT":"#FFB612","SEA":"#002244",
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
TEAM_DIV = {t: d for d, ts in DIVISIONS.items() for t in ts}
TEAM_CONF = {t: ("AFC" if d.startswith("AFC") else "NFC") for t, d in TEAM_DIV.items()}

DATA = os.path.join(os.path.dirname(__file__), "data")

# ── Data loading (cached) ─────────────────────────────────────────────────────
@st.cache_data
def load_data():
    proj_noveg  = pd.read_json(f"{DATA}/proj_noveg.json")
    proj_veg    = pd.read_json(f"{DATA}/proj_veg.json")
    sched       = pd.read_json(f"{DATA}/sched_2025.json")
    team_roll   = pd.read_json(f"{DATA}/team_rolling.json")
    elo_2025    = pd.read_json(f"{DATA}/elo_2025.json")
    final_wins  = pd.read_json(f"{DATA}/final_wins.json")
    model_res   = json.load(open(f"{DATA}/model_results.json"))
    feat_imp    = json.load(open(f"{DATA}/feature_importance.json"))
    otc_sp      = pd.read_csv(f"{DATA}/otc_positional_spending.csv")
    otc_ct      = pd.read_csv(f"{DATA}/otc_player_contracts.csv")
    hfa         = pd.read_json(f"{DATA}/hfa.json")
    return (proj_noveg, proj_veg, sched, team_roll, elo_2025,
            final_wins, model_res, feat_imp, otc_sp, otc_ct, hfa)

(proj_noveg, proj_veg, sched, team_roll, elo_2025,
 final_wins, model_res, feat_imp, otc_sp, otc_ct, hfa) = load_data()

ALL_TEAMS = sorted(proj_noveg["team"].unique())

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        f"<div style='background:{BEARS};padding:16px 12px;border-radius:8px;margin-bottom:16px'>"
        f"<span style='color:white;font-size:20px;font-weight:700'>🐻 NFL Win Predictor</span><br>"
        f"<span style='color:{SLATE};font-size:12px'>2025 Season · Bears Analytics</span>"
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
        if st.button("↩ Reset to Bears", use_container_width=True):
            st.session_state["global_team"] = "CHI"
            st.rerun()

    st.divider()
    model_choice = st.selectbox("Primary Model", [
        "XGBoost (No Vegas)", "Random Forest (No Vegas)",
        "Neural Network (No Vegas)", "XGBoost (Vegas)", "Random Forest (Vegas)",
    ], index=0)

    proj_df = proj_noveg if "No Vegas" in model_choice else proj_veg
    vegas_acc = model_res["vegas_acc"]

    st.caption(f"Vegas baseline: **{vegas_acc:.1%}**")
    st.caption("Train: 2018–2024 W1-17\nTest: 2025 W1-17")

# ── Helper: clean plotly fig ──────────────────────────────────────────────────
def clean(fig, h=400, margins=None):
    m = margins or dict(l=48, r=24, t=36, b=48)
    fig.update_layout(
        height=h, margin=m,
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family="Inter, system-ui, sans-serif", size=13, color="#1E293B"),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=BORDER,
                     tickcolor=BORDER, tickfont_color=SLATE)
    fig.update_yaxes(showgrid=True, gridcolor=BORDER, zeroline=False,
                     linecolor=BORDER, tickfont_color=SLATE)
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — TEAM DEEP DIVE
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏈 Team Deep Dive":
    tc = TEAM_COLORS.get(team, BLUE)

    # Header
    final = final_wins[final_wins["team"] == team]["actual_wins_2025"].values
    actual_w = int(final[0]) if len(final) else "—"
    actual_l = 17 - actual_w if isinstance(actual_w, int) else "—"

    pre_proj = proj_df[(proj_df["team"] == team) & (proj_df["proj_week"] == 1)]
    pre_wins = float(pre_proj["proj_total_wins"].values[0]) if len(pre_proj) else None
    final_proj = proj_df[proj_df["team"] == team].sort_values("proj_week").iloc[-1]
    final_proj_wins = round(float(final_proj["proj_total_wins"]), 1)
    ci_lo = float(final_proj["ci_lo_80"]); ci_hi = float(final_proj["ci_hi_80"])

    div = TEAM_DIV.get(team, ""); conf = TEAM_CONF.get(team, "")
    st.markdown(
        f"<div style='background:{tc};padding:20px 24px;border-radius:10px;margin-bottom:20px'>"
        f"<div style='display:flex;justify-content:space-between;align-items:center'>"
        f"<div><span style='color:white;font-size:28px;font-weight:800'>{team}</span>"
        f"<span style='color:rgba(255,255,255,0.65);font-size:14px;margin-left:12px'>{div} · {conf}</span></div>"
        f"<div style='text-align:right'>"
        f"<span style='color:white;font-size:36px;font-weight:800'>{actual_w}–{actual_l}</span>"
        f"<span style='color:rgba(255,255,255,0.65);font-size:14px;display:block'>2025 Final Record</span>"
        f"</div></div></div>", unsafe_allow_html=True)

    # KPI row
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Preseason Projection", f"{pre_wins:.1f} W" if pre_wins else "—",
                  delta=f"{actual_w - pre_wins:+.1f} vs actual" if pre_wins else None)
    with k2:
        st.metric("Final Projection", f"{final_proj_wins} W")
    with k3:
        st.metric("80% CI Range", f"{ci_lo:.0f}–{ci_hi:.0f} W")
    with k4:
        acc_delta = final_proj_wins - actual_w if isinstance(actual_w, int) else None
        st.metric("Projection Error", f"{abs(acc_delta):.1f} W off" if acc_delta is not None else "—",
                  delta=f"{acc_delta:+.1f}" if acc_delta is not None else None,
                  delta_color="inverse")

    st.divider()

    # ── Projection evolution ──────────────────────────────────────────────────
    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.subheader("📈 Win Projection Evolution")
        td = proj_df[proj_df["team"] == team].sort_values("proj_week")
        fig = go.Figure()
        # CI band
        fig.add_trace(go.Scatter(
            x=pd.concat([td["proj_week"], td["proj_week"][::-1]]),
            y=pd.concat([td["ci_hi_80"], td["ci_lo_80"][::-1]]),
            fill="toself", fillcolor=f"rgba({int(tc[1:3],16)},{int(tc[3:5],16)},{int(tc[5:7],16)},0.12)",
            line=dict(width=0), name="80% CI", hoverinfo="skip"))
        # Projection line
        fig.add_trace(go.Scatter(
            x=td["proj_week"], y=td["proj_total_wins"], mode="lines+markers",
            name="Projected wins", line=dict(color=tc, width=3),
            marker=dict(size=6, color=tc),
            hovertemplate="<b>Before Wk %{x}</b><br>Proj: %{y:.1f} W<extra></extra>"))
        # Actual line
        if isinstance(actual_w, int):
            fig.add_hline(y=actual_w, line_dash="dot", line_color=SLATE, line_width=2,
                          annotation_text=f"Actual: {actual_w}W",
                          annotation_font_color=SLATE, annotation_position="top right")
        fig.update_layout(xaxis=dict(title="Projection made before week...", dtick=2),
                          yaxis=dict(title="Projected wins", range=[0, 18]),
                          showlegend=True, legend=dict(x=0.01, y=0.99))
        st.plotly_chart(clean(fig, h=360), use_container_width=True)

    with col_b:
        st.subheader("📅 2025 Game Log")
        tg_sched = sched[sched["team"] == team].sort_values("week").copy()
        tg_sched["Result"] = tg_sched.apply(
            lambda r: f"{'W' if r['win']==1 else 'L'} {int(r['team_score'])}-{int(r['opp_score'])}"
            if pd.notna(r.get("win")) and pd.notna(r.get("team_score")) else "—", axis=1)
        tg_sched["Win Prob"] = (tg_sched["win_prob_noveg"] * 100).round(1).astype(str) + "%"
        tg_sched["H/A"] = tg_sched["is_home"].map({1: "🏠 H", 0: "✈️ A"})
        disp = tg_sched[["week","opponent","H/A","Result","Win Prob"]].rename(
            columns={"week":"Wk","opponent":"Opp"})
        st.dataframe(disp, hide_index=True, height=340,
                     column_config={"Wk": st.column_config.NumberColumn(width="small"),
                                    "Win Prob": st.column_config.TextColumn(width="medium")})

    st.divider()

    # ── ELO + Rolling metrics ─────────────────────────────────────────────────
    col_c, col_d = st.columns(2)
    with col_c:
        st.subheader("⚡ ELO Ratings Over Season")
        elo_t = elo_2025[elo_2025["team"] == team].sort_values("week")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=elo_t["week"], y=elo_t["team_off_elo"], mode="lines+markers",
            name="Offensive ELO", line=dict(color=tc, width=2.5), marker=dict(size=5)))
        fig2.add_trace(go.Scatter(x=elo_t["week"], y=elo_t["team_def_elo"], mode="lines+markers",
            name="Defensive ELO", line=dict(color=AMBER, width=2.5, dash="dash"), marker=dict(size=5)))
        fig2.add_hline(y=1500, line_dash="dot", line_color=SLATE, line_width=1,
                       annotation_text="Avg (1500)", annotation_font_color=SLATE)
        fig2.update_layout(xaxis=dict(title="Week", dtick=2),
                           yaxis=dict(title="ELO Rating"),
                           showlegend=True, legend=dict(x=0.01, y=0.01))
        st.plotly_chart(clean(fig2, h=300), use_container_width=True)

    with col_d:
        st.subheader("📊 Key Efficiency Metrics")
        tr = team_roll[team_roll["team"] == team].sort_values("week")
        metric_sel = st.selectbox("Metric", [
            "net_ypa", "def_yds_per_tgt", "pressure_rate", "yds_per_carry"], key="metric_sel",
            format_func=lambda x: {
                "net_ypa": "Net Yards/Attempt (Off)",
                "def_yds_per_tgt": "Yards/Target Allowed (Def)",
                "pressure_rate": "Pressure Rate (Off)",
                "yds_per_carry": "Yards/Carry (Rush)"
            }[x])
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=tr["week"], y=tr[metric_sel], mode="lines+markers",
            name=metric_sel, line=dict(color=tc, width=2.5), marker=dict(size=6),
            hovertemplate="Week %{x}<br>%{y:.3f}<extra></extra>"))
        fig3.update_layout(xaxis=dict(title="Week", dtick=2),
                           yaxis=dict(title="Game value"),
                           showlegend=False)
        st.plotly_chart(clean(fig3, h=260), use_container_width=True)

    # ── Schedule difficulty heatmap ───────────────────────────────────────────
    st.subheader("🗓️ Schedule Win Probability by Week")
    tg_sched2 = sched[sched["team"] == team].sort_values("week").copy()
    tg_sched2["wp"] = tg_sched2["win_prob_noveg"]
    fig4 = go.Figure()
    colors_wp = [tc if r["win"] == 1 else RED if r["win"] == 0 else SLATE
                 for _, r in tg_sched2.iterrows()]
    fig4.add_trace(go.Bar(
        x=tg_sched2["week"], y=tg_sched2["wp"],
        marker_color=colors_wp, marker_line_width=0,
        text=tg_sched2["opponent"], textposition="outside", textfont=dict(size=9),
        hovertemplate="<b>Wk %{x} vs %{text}</b><br>Win Prob: %{y:.1%}<extra></extra>"))
    fig4.add_hline(y=0.5, line_dash="dot", line_color=SLATE, line_width=1.5)
    fig4.update_layout(xaxis=dict(title="Week", dtick=1),
                       yaxis=dict(title="Model Win Probability", tickformat=".0%", range=[0, 1.15]),
                       showlegend=False)
    st.plotly_chart(clean(fig4, h=320), use_container_width=True)
    st.caption(f"Bar color: {'🟦' if tc else ''} Win · 🔴 Loss · ⬜ N/A  |  Label = Opponent")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Comparison":
    st.title("📊 Model Performance Comparison")
    st.markdown("All models trained on **2018–2024 Weeks 1–17** and evaluated on **2025 Weeks 1–17**.")

    tab1, tab2, tab3 = st.tabs(["🎯 Classifier Accuracy", "📐 Regressor Accuracy", "🔍 Feature Importance"])

    with tab1:
        col_nv, col_v = st.columns(2)

        def build_acc_bar(results_dict, title, color):
            names = list(results_dict.keys())
            accs  = [v["test_acc"] for v in results_dict.values()]
            train = [v["train_acc"] for v in results_dict.values()]
            fig = go.Figure()
            fig.add_trace(go.Bar(name="Train Acc", x=names, y=train,
                marker_color=SLATE, opacity=0.45, marker_line_width=0,
                text=[f"{v:.3f}" for v in train], textposition="outside", textfont=dict(size=9)))
            fig.add_trace(go.Bar(name="Test Acc", x=names, y=accs,
                marker_color=color, marker_line_width=0,
                text=[f"{v:.3f}" for v in accs], textposition="outside", textfont=dict(size=10)))
            fig.add_hline(y=vegas_acc, line_dash="dash", line_color=RED, line_width=2,
                          annotation_text=f"Vegas {vegas_acc:.3f}",
                          annotation_font_color=RED, annotation_position="top right")
            fig.update_layout(barmode="group", title=title,
                yaxis=dict(title="Accuracy", tickformat=".0%",
                    range=[min(accs+train+[vegas_acc])-0.05, max(accs+train+[vegas_acc])+0.09]),
                showlegend=True, legend=dict(x=0.01, y=0.99),
                xaxis=dict(tickangle=-20))
            return clean(fig, h=420)

        with col_nv:
            st.plotly_chart(build_acc_bar(model_res["cls_noveg"],
                "Without Vegas Lines", BLUE), use_container_width=True)
        with col_v:
            st.plotly_chart(build_acc_bar(model_res["cls_vegas"],
                "With Vegas Lines", TEAL), use_container_width=True)

        # Summary table
        st.subheader("Full Metrics Table — Classifiers")
        rows = []
        for cat, label in [("cls_noveg","No Vegas"),("cls_vegas","Vegas")]:
            for name, res in model_res[cat].items():
                rows.append({"Model": name, "Features": label,
                    "Test Acc": f"{res['test_acc']:.1%}",
                    "Train Acc": f"{res['train_acc']:.1%}",
                    "Gap": f"{res['train_acc']-res['test_acc']:+.3f}",
                    "AUC": f"{res['auc']:.4f}" if res.get("auc") else "—",
                    "Brier": f"{res['brier']:.4f}" if res.get("brier") else "—",
                    "vs Vegas": f"{res['vs_vegas']:+.1%}"})
        rows.append({"Model":"Vegas Spread Baseline","Features":"N/A",
            "Test Acc":f"{vegas_acc:.1%}","Train Acc":"—","Gap":"—","AUC":"—","Brier":"—","vs Vegas":"—"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    with tab2:
        col_nv2, col_v2 = st.columns(2)

        def build_reg_bar(results_dict, title, color):
            names = list(results_dict.keys())
            accs  = [v["test_acc"] for v in results_dict.values()]
            maes  = [v["mae"] for v in results_dict.values()]
            fig = make_subplots(rows=1, cols=2,
                subplot_titles=["Win/Loss Acc (from margin)", "MAE (points)"])
            fig.add_trace(go.Bar(x=names, y=accs, marker_color=color, marker_line_width=0,
                text=[f"{v:.3f}" for v in accs], textposition="outside", textfont=dict(size=9),
                showlegend=False), row=1, col=1)
            fig.add_hline(y=vegas_acc, line_dash="dash", line_color=RED, line_width=1.5,
                          row=1, col=1)
            fig.add_trace(go.Bar(x=names, y=maes, marker_color=AMBER, marker_line_width=0,
                text=[f"{v:.2f}" for v in maes], textposition="outside", textfont=dict(size=9),
                showlegend=False), row=1, col=2)
            fig.update_xaxes(tickangle=-20, tickfont=dict(size=9))
            fig.update_yaxes(tickformat=".0%",
                range=[min(accs+[vegas_acc])-0.04, max(accs+[vegas_acc])+0.08], row=1, col=1)
            fig.update_layout(title=title, height=400,
                margin=dict(l=48,r=24,t=64,b=80),
                paper_bgcolor="white", plot_bgcolor="white",
                font=dict(family="Inter, system-ui, sans-serif", size=12))
            return fig

        with col_nv2:
            st.plotly_chart(build_reg_bar(model_res["reg_noveg"],"Without Vegas Lines",BLUE),
                            use_container_width=True)
        with col_v2:
            st.plotly_chart(build_reg_bar(model_res["reg_vegas"],"With Vegas Lines",TEAL),
                            use_container_width=True)

        st.subheader("Full Metrics Table — Regressors")
        rows2 = []
        for cat, label in [("reg_noveg","No Vegas"),("reg_vegas","Vegas")]:
            for name, res in model_res[cat].items():
                rows2.append({"Model":name,"Features":label,
                    "Test Acc":f"{res['test_acc']:.1%}","Train Acc":f"{res['train_acc']:.1%}",
                    "MAE":f"{res['mae']:.2f} pts","RMSE":f"{res['rmse']:.2f} pts",
                    "vs Vegas":f"{res['vs_vegas']:+.1%}"})
        st.dataframe(pd.DataFrame(rows2), hide_index=True, use_container_width=True)

    with tab3:
        st.subheader("🔍 Feature Importance — XGBoost (No Vegas)")
        fi_sorted = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)
        fi_df = pd.DataFrame(fi_sorted, columns=["Feature","Importance"])
        threshold = fi_df["Importance"].quantile(0.75)
        fi_df["color"] = fi_df["Importance"].apply(
            lambda v: BEARS if v>=threshold else SLATE)
        fig_fi = go.Figure(go.Bar(
            x=fi_df["Importance"], y=fi_df["Feature"],
            orientation="h", marker_color=fi_df["color"], marker_line_width=0,
            text=fi_df["Importance"].round(4).astype(str),
            textposition="outside", textfont=dict(size=9)))
        fig_fi.update_layout(xaxis_title="Feature Importance (gain)",
                             yaxis=dict(autorange="reversed"))
        st.plotly_chart(clean(fig_fi, h=600, margins=dict(l=220,r=80,t=32,b=48)),
                        use_container_width=True)

        # Linear vs Non-linear story
        st.subheader("Linear vs. Non-Linear Model Gap")
        linear_avg = np.mean([
            model_res["cls_noveg"]["Logistic Regression"]["test_acc"],
            model_res["cls_noveg"]["Ridge Classifier"]["test_acc"]])
        nonlinear_avg = np.mean([
            model_res["cls_noveg"]["XGBoost"]["test_acc"],
            model_res["cls_noveg"]["Random Forest"]["test_acc"]])
        nn_acc = model_res["cls_noveg"]["Neural Network"]["test_acc"]
        fig_gap = go.Figure(go.Bar(
            x=["Linear\n(Logistic / Ridge)", "Non-linear Trees\n(XGBoost / RF)", "Neural Network\n(MLP)"],
            y=[linear_avg, nonlinear_avg, nn_acc],
            marker_color=[AMBER, BLUE, GREEN], marker_line_width=0,
            text=[f"{v:.3f}" for v in [linear_avg, nonlinear_avg, nn_acc]],
            textposition="outside", textfont=dict(size=13)))
        fig_gap.add_hline(y=vegas_acc, line_dash="dot", line_color=RED, line_width=2,
                          annotation_text=f"Vegas {vegas_acc:.3f}",
                          annotation_font_color=RED)
        fig_gap.update_layout(
            yaxis=dict(title="Test accuracy", tickformat=".0%",
                range=[min(linear_avg,nonlinear_avg,nn_acc,vegas_acc)-0.04,
                       max(linear_avg,nonlinear_avg,nn_acc,vegas_acc)+0.08]),
            showlegend=False)
        st.plotly_chart(clean(fig_gap, h=380), use_container_width=True)
        st.info("**Interpretation:** The gap between linear and tree-based models reflects how much interaction "
                "structure exists in the data. A larger gap means the non-linear patterns are real and complex "
                "models are justified — a key finding for the presentation.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — LEAGUE PROJECTIONS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🗺️ League Projections":
    st.title("🗺️ League-Wide 2025 Projections")

    tab_a, tab_b, tab_c = st.tabs(["🏆 All 32 Teams", "🏈 By Division", "📅 Week-by-Week"])

    with tab_a:
        week_sel = st.slider("Projection snapshot (before week...)", 1,
                             int(proj_df["proj_week"].max()), int(proj_df["proj_week"].max()))
        cur = (proj_df[proj_df["proj_week"] == week_sel]
               .sort_values("proj_total_wins", ascending=True).reset_index(drop=True))
        cur["actual_wins"] = cur["team"].map(
            final_wins.set_index("team")["actual_wins_2025"])
        colors_bar = [TEAM_COLORS.get(t, BLUE) for t in cur["team"]]
        fig_all = go.Figure()
        # CI error bars
        for _, row in cur.iterrows():
            fig_all.add_trace(go.Scatter(
                x=[row["proj_total_wins"], row["proj_total_wins"]],
                y=[row["ci_lo_80"], row["ci_hi_80"]],
                mode="lines", line=dict(color=SLATE, width=7), opacity=0.3,
                showlegend=False, hoverinfo="skip"))
        fig_all.add_trace(go.Scatter(
            x=cur["proj_total_wins"], y=cur["team"],
            mode="markers+text", text=[f"{v:.1f}" for v in cur["proj_total_wins"]],
            textposition="middle right", textfont=dict(size=9),
            marker=dict(size=11, color=colors_bar, line=dict(width=1.5, color="white")),
            hovertemplate="<b>%{y}</b><br>Proj: %{x:.1f} W<br>"
                          "CI: %{customdata[0]:.0f}–%{customdata[1]:.0f}<extra></extra>",
            customdata=cur[["ci_lo_80","ci_hi_80"]].values,
            showlegend=False))
        # Actual wins overlay
        fig_all.add_trace(go.Scatter(
            x=cur["actual_wins"], y=cur["team"], mode="markers",
            marker=dict(size=8, symbol="diamond", color="rgba(0,0,0,0.5)"),
            name="Actual wins",
            hovertemplate="<b>%{y}</b><br>Actual: %{x:.0f} W<extra></extra>"))
        fig_all.update_layout(
            xaxis=dict(title="Wins", range=[0, 20]),
            yaxis=dict(tickfont=dict(size=10)),
            showlegend=True, legend=dict(x=0.75, y=0.02),
            title=f"Week {week_sel} Projections — grey bar = 80% CI · ◆ = actual final wins")
        st.plotly_chart(clean(fig_all, h=max(520, len(cur)*16),
                              margins=dict(l=56,r=90,t=48,b=48)), use_container_width=True)

        # Preseason vs actual scatter
        st.subheader("🎯 Preseason Projection vs Actual Wins")
        pre = proj_df[proj_df["proj_week"] == 1].copy()
        pre["actual"] = pre["team"].map(final_wins.set_index("team")["actual_wins_2025"])
        pre["color"] = [TEAM_COLORS.get(t, BLUE) for t in pre["team"]]
        mae_val = (pre["proj_total_wins"] - pre["actual"]).abs().mean()
        fig_scat = go.Figure()
        fig_scat.add_trace(go.Scatter(x=[2,17], y=[2,17], mode="lines", hoverinfo="skip",
            line=dict(color=SLATE, dash="dot", width=1.5), showlegend=False))
        for _, row in pre.dropna(subset=["actual"]).iterrows():
            fig_scat.add_trace(go.Scatter(
                x=[row["proj_total_wins"]], y=[row["actual"]],
                mode="markers+text", text=[row["team"]],
                textposition="top center", textfont=dict(size=9, color=SLATE),
                marker=dict(size=10, color=TEAM_COLORS.get(row["team"], BLUE),
                            line=dict(width=1.5, color="white")),
                showlegend=False,
                hovertemplate=f"<b>{row['team']}</b><br>Proj: {row['proj_total_wins']:.1f}<br>Actual: {int(row['actual'])}<extra></extra>"))
        fig_scat.update_layout(
            title=f"Preseason MAE = {mae_val:.2f} wins",
            xaxis=dict(title="Preseason projected wins", range=[3,16]),
            yaxis=dict(title="Actual 2025 wins", range=[1,18]),
            showlegend=False)
        st.plotly_chart(clean(fig_scat, h=440), use_container_width=True)

    with tab_b:
        st.subheader("Division Breakdown")
        final_proj_all = (proj_df.groupby("team")["proj_total_wins"]
                          .last().reset_index().rename(columns={"proj_total_wins":"proj_wins"}))
        final_proj_all["actual"] = final_proj_all["team"].map(
            final_wins.set_index("team")["actual_wins_2025"])
        final_proj_all["division"] = final_proj_all["team"].map(TEAM_DIV)
        final_proj_all["conference"] = final_proj_all["team"].map(TEAM_CONF)

        conf_sel = st.radio("Conference", ["AFC", "NFC", "Both"], horizontal=True)
        divs_to_show = {d: ts for d, ts in DIVISIONS.items()
                        if conf_sel == "Both" or d.startswith(conf_sel)}

        n_divs = len(divs_to_show)
        ncols = 2
        nrows = (n_divs + 1) // 2
        div_list = list(divs_to_show.items())
        for row_i in range(nrows):
            cols = st.columns(2)
            for col_i in range(2):
                idx = row_i * 2 + col_i
                if idx >= len(div_list): break
                div_name, div_teams = div_list[idx]
                sub = final_proj_all[final_proj_all["team"].isin(div_teams)].sort_values("proj_wins", ascending=False)
                with cols[col_i]:
                    fig_div = go.Figure()
                    fig_div.add_trace(go.Bar(
                        x=sub["team"], y=sub["proj_wins"],
                        name="Projected", marker_color=[TEAM_COLORS.get(t,BLUE) for t in sub["team"]],
                        marker_line_width=0,
                        text=[f"{v:.1f}" for v in sub["proj_wins"]],
                        textposition="outside", textfont=dict(size=11)))
                    fig_div.add_trace(go.Scatter(
                        x=sub["team"], y=sub["actual"], mode="markers",
                        marker=dict(size=12, symbol="diamond", color="black", opacity=0.6),
                        name="Actual", showlegend=False,
                        hovertemplate="%{x}: %{y:.0f} actual<extra></extra>"))
                    fig_div.update_layout(
                        title=div_name, showlegend=False,
                        yaxis=dict(title="Wins", range=[0,18]),
                        xaxis_title=None)
                    st.plotly_chart(clean(fig_div, h=280), use_container_width=True)

    with tab_c:
        st.subheader("Week-by-Week Projection Table")
        piv = (proj_df.pivot(index="team", columns="proj_week", values="proj_total_wins")
               .round(1))
        piv.columns = ["Pre"] + [f"Wk{w}" for w in range(1, len(piv.columns))]
        piv["Actual"] = piv.index.map(final_wins.set_index("team")["actual_wins_2025"])
        piv["Error"] = (piv[piv.columns[-2]] - piv["Actual"]).abs().round(1)
        piv = piv.sort_values("Pre", ascending=False)
        st.dataframe(piv, use_container_width=True,
                     column_config={"Actual": st.column_config.NumberColumn(format="%.0f"),
                                    "Error":  st.column_config.NumberColumn(format="%.1f")})


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — CAP SPENDING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💰 Cap Spending":
    st.title("💰 Cap Spending Analysis")
    st.markdown(
        "> **Scope note:** Cap data from Over The Cap (OTC) is used here as a *post-model* "
        "descriptive analysis only. It has **no influence** on any prediction. "
        "Questions explored: what do top-projected teams invest in differently?")

    EXCL_POS = {"K","P","LS","FB"}
    POS_GROUPS = {
        "QB":    ["QB"], "RB": ["RB","FB"], "OL": ["OT","OG","C"],
        "WR/TE": ["WR","TE"], "DL": ["EDGE","IDL"], "LB": ["LB"],
        "CB":    ["CB"], "S":  ["S"],
    }

    # Assign quadrants from final projections
    final_all = proj_df.groupby("team")["proj_total_wins"].last().reset_index()
    final_all = final_all.sort_values("proj_total_wins", ascending=False).reset_index(drop=True)
    final_all["rank"] = final_all.index + 1
    final_all["quadrant"] = pd.cut(final_all["rank"], bins=[0,8,16,24,32],
        labels=["Q1 (Top 8)","Q2 (Upper-mid)","Q3 (Lower-mid)","Q4 (Bottom 8)"])
    final_all["actual"] = final_all["team"].map(
        final_wins.set_index("team")["actual_wins_2025"])

    QUAD_LABELS = ["Q1 (Top 8)","Q2 (Upper-mid)","Q3 (Lower-mid)","Q4 (Bottom 8)"]
    Q_COLORS = {"Q1 (Top 8)":BEARS,"Q2 (Upper-mid)":BLUE,"Q3 (Lower-mid)":AMBER,"Q4 (Bottom 8)":RED}

    tab_sp1, tab_sp2, tab_sp3, tab_sp4 = st.tabs([
        "🏈 Team Spending", "📊 Quadrant Patterns", "🕸️ Radar Charts", "💼 Player Contracts"])

    with tab_sp1:
        team_sp = st.selectbox("Select team", ALL_TEAMS,
                               index=ALL_TEAMS.index("CHI"), key="cap_team")
        tq = final_all[final_all["team"]==team_sp]["quadrant"].values
        team_quad = tq[0] if len(tq) else "—"
        team_rank = final_all[final_all["team"]==team_sp]["rank"].values
        team_rank_str = f"#{int(team_rank[0])}" if len(team_rank) else "—"
        team_proj  = final_all[final_all["team"]==team_sp]["proj_total_wins"].values
        team_proj_str = f"{team_proj[0]:.1f} W" if len(team_proj) else "—"

        c1,c2,c3 = st.columns(3)
        c1.metric("Win Quadrant", str(team_quad))
        c2.metric("League Rank", team_rank_str)
        c3.metric("Projected Wins", team_proj_str)

        t_spend = otc_sp[otc_sp["team"]==team_sp].copy()
        lg_avg = otc_sp.groupby("position")["cap_pct"].mean()
        t_spend["delta"] = t_spend.apply(lambda r: r["cap_pct"]-lg_avg.get(r["position"],0), axis=1)
        t_spend = t_spend[~t_spend["position"].isin(EXCL_POS)].sort_values("cap_pct", ascending=False)

        col_aa, col_bb = st.columns(2)
        with col_aa:
            st.subheader("Positional Cap %")
            tc2 = TEAM_COLORS.get(team_sp, BLUE)
            fig_sp = go.Figure(go.Bar(
                x=t_spend["position"], y=t_spend["cap_pct"],
                marker_color=tc2, marker_line_width=0,
                text=t_spend["cap_pct"].round(1).astype(str)+"%",
                textposition="outside"))
            fig_sp.update_layout(xaxis_title=None, yaxis=dict(title="Cap %", ticksuffix="%"))
            st.plotly_chart(clean(fig_sp, h=320), use_container_width=True)
        with col_bb:
            st.subheader("Delta vs League Average")
            colors_delta = ["#0F6E56" if d>=0 else RED for d in t_spend["delta"]]
            fig_delt = go.Figure(go.Bar(
                x=t_spend["position"], y=t_spend["delta"],
                marker_color=colors_delta, marker_line_width=0,
                text=[f"{d:+.1f}%" for d in t_spend["delta"]], textposition="outside"))
            fig_delt.add_hline(y=0, line_color=SLATE, line_width=1)
            fig_delt.update_layout(xaxis_title=None,
                yaxis=dict(title="Delta vs League Avg", ticksuffix="%"))
            st.plotly_chart(clean(fig_delt, h=320), use_container_width=True)

        st.subheader("Cap Spending Details")
        disp_sp = t_spend[["position","total_cap_spend","n_players","avg_cap_per_player","cap_pct","delta"]].copy()
        disp_sp.columns = ["Position","Total Spend","# Players","Avg/Player","Cap %","vs League Avg"]
        disp_sp["Total Spend"] = disp_sp["Total Spend"].apply(lambda x: f"${x/1e6:.1f}M")
        disp_sp["Avg/Player"] = disp_sp["Avg/Player"].apply(lambda x: f"${x/1e6:.1f}M")
        disp_sp["Cap %"] = disp_sp["Cap %"].apply(lambda x: f"{x:.1f}%")
        disp_sp["vs League Avg"] = disp_sp["vs League Avg"].apply(lambda x: f"{x:+.2f}%")
        st.dataframe(disp_sp, hide_index=True, use_container_width=True)

    with tab_sp2:
        st.subheader("Positional Spending by Win Quadrant")
        lg_avg_pos = otc_sp.groupby("position")["cap_pct"].mean()
        all_pos = [p for p in sorted(otc_sp["position"].unique()) if p not in EXCL_POS]

        view = st.radio("View", ["Raw Cap %", "Delta vs League Average"], horizontal=True)

        fig_quad = make_subplots(rows=2, cols=2,
            subplot_titles=QUAD_LABELS, vertical_spacing=0.14, horizontal_spacing=0.10)
        for qi, q in enumerate(QUAD_LABELS):
            row, col = (qi//2)+1, (qi%2)+1
            teams_q = final_all[final_all["quadrant"]==q]["team"].tolist()
            q_spend = otc_sp[otc_sp["team"].isin(teams_q) & ~otc_sp["position"].isin(EXCL_POS)]
            pos_avg = q_spend.groupby("position")["cap_pct"].mean().reindex(all_pos).fillna(0)
            if view == "Raw Cap %":
                vals = pos_avg.values
                bar_colors = [Q_COLORS[q]]*len(vals)
                yaxis_fmt = ".1f"
            else:
                vals = (pos_avg - lg_avg_pos.reindex(all_pos).fillna(0)).values
                bar_colors = ["#0F6E56" if v>=0 else RED for v in vals]
                yaxis_fmt = "+.1f"
            fig_quad.add_trace(go.Bar(
                x=all_pos, y=vals, marker_color=bar_colors, marker_line_width=0,
                text=[f"{v:{yaxis_fmt}}" for v in vals],
                textposition="outside", textfont=dict(size=8), showlegend=False), row=row, col=col)
            if view == "Delta vs League Average":
                fig_quad.add_hline(y=0, line_color="rgba(0,0,0,0.2)", line_width=1, row=row, col=col)
        fig_quad.update_xaxes(tickangle=-40, tickfont=dict(size=9))
        fig_quad.update_yaxes(ticksuffix="%")
        fig_quad.update_layout(height=580, margin=dict(l=48,r=24,t=72,b=48),
            paper_bgcolor="white", plot_bgcolor="white",
            font=dict(family="Inter, system-ui, sans-serif", size=12))
        st.plotly_chart(fig_quad, use_container_width=True)

        # Compare team vs its quadrant peers
        st.subheader(f"Selected Team ({team}) vs Its Quadrant Peers")
        team_q_label = str(final_all[final_all["team"]==team]["quadrant"].values[0]) if len(final_all[final_all["team"]==team]) else None
        if team_q_label:
            peers = final_all[final_all["quadrant"]==team_q_label]["team"].tolist()
            peer_spend = otc_sp[otc_sp["team"].isin(peers) & ~otc_sp["position"].isin(EXCL_POS)]
            peer_avg = peer_spend.groupby("position")["cap_pct"].mean()
            team_s = otc_sp[otc_sp["team"]==team & ~otc_sp["position"].isin(EXCL_POS)
                            if False else otc_sp["team"]==team].copy()
            team_s = team_s[~team_s["position"].isin(EXCL_POS)]
            positions = sorted(set(peer_avg.index) | set(team_s["position"].values))
            fig_peer = go.Figure()
            tc3 = TEAM_COLORS.get(team, BLUE)
            fig_peer.add_trace(go.Bar(name=f"{team}", x=positions,
                y=[team_s[team_s["position"]==p]["cap_pct"].sum() for p in positions],
                marker_color=tc3, marker_line_width=0, opacity=0.9))
            fig_peer.add_trace(go.Bar(name=f"{team_q_label} avg", x=positions,
                y=[peer_avg.get(p,0) for p in positions],
                marker_color=SLATE, marker_line_width=0, opacity=0.55))
            fig_peer.update_layout(barmode="group", xaxis_title=None,
                yaxis=dict(title="Cap %", ticksuffix="%"), showlegend=True)
            st.plotly_chart(clean(fig_peer, h=340), use_container_width=True)

    with tab_sp3:
        st.subheader("🕸️ Spending Profile Radars by Quadrant")
        SPIDER_ORDER = ["QB","WR/TE","OL","RB","S","CB","LB","DL"]
        lg_grp_avg = {}
        for grp, positions in POS_GROUPS.items():
            sub = otc_sp[otc_sp["position"].isin(positions)]
            per_team = sub.groupby("team")["cap_pct"].sum()
            lg_grp_avg[grp] = per_team.mean()

        Q_FILL = {
            "Q1 (Top 8)": f"rgba(11,22,42,0.15)",
            "Q2 (Upper-mid)": f"rgba(37,99,235,0.15)",
            "Q3 (Lower-mid)": f"rgba(217,119,6,0.15)",
            "Q4 (Bottom 8)": f"rgba(220,38,38,0.15)",
        }
        fig_spider = make_subplots(rows=2, cols=2,
            specs=[[{"type":"polar"},{"type":"polar"}],[{"type":"polar"},{"type":"polar"}]],
            subplot_titles=QUAD_LABELS, vertical_spacing=0.12, horizontal_spacing=0.08)
        for qi, q in enumerate(QUAD_LABELS):
            row, col = (qi//2)+1, (qi%2)+1
            teams_q = final_all[final_all["quadrant"]==q]["team"].tolist()
            q_spend = otc_sp[otc_sp["team"].isin(teams_q)]
            grp_vals = {}
            for grp, positions in POS_GROUPS.items():
                sub = q_spend[q_spend["position"].isin(positions)]
                per_team = sub.groupby("team")["cap_pct"].sum()
                grp_vals[grp] = per_team.mean() if len(per_team) else 0
            r_vals = [grp_vals.get(g,0) for g in SPIDER_ORDER]
            r_closed = r_vals + [r_vals[0]]
            t_closed = SPIDER_ORDER + [SPIDER_ORDER[0]]
            fig_spider.add_trace(go.Scatterpolar(
                r=r_closed, theta=t_closed, fill="toself",
                fillcolor=Q_FILL[q], line=dict(color=Q_COLORS[q], width=2.2),
                name=q, showlegend=False), row=row, col=col)
            lg_r = [lg_grp_avg.get(g,0) for g in SPIDER_ORDER]
            fig_spider.add_trace(go.Scatterpolar(
                r=lg_r+[lg_r[0]], theta=t_closed, fill="none",
                line=dict(color="rgba(128,128,128,0.5)", width=1, dash="dot"),
                showlegend=False), row=row, col=col)
            pk = "polar" if qi==0 else f"polar{qi+1}"
            fig_spider.update_layout(**{pk: dict(
                radialaxis=dict(visible=True, range=[0,30], tickfont=dict(size=8),
                                gridcolor="rgba(128,128,128,0.2)"),
                angularaxis=dict(tickfont=dict(size=10), gridcolor="rgba(128,128,128,0.2)"),
                bgcolor="white")})
        fig_spider.update_layout(height=640, margin=dict(l=40,r=40,t=80,b=40),
            paper_bgcolor="white",
            font=dict(family="Inter, system-ui, sans-serif", size=12))
        st.plotly_chart(fig_spider, use_container_width=True)
        st.caption("Dotted circle = league average. Each axis = positional group cap %.")

    with tab_sp4:
        st.subheader("💼 Player Contract Details")
        team_ct = st.selectbox("Team", ALL_TEAMS, index=ALL_TEAMS.index("CHI"), key="ct_team")
        ct_data = otc_ct[otc_ct["team"]==team_ct].copy()
        if len(ct_data):
            ct_data = ct_data.sort_values("cap_number", ascending=False)
            ct_disp = ct_data[["player","position","cap_number","base_salary",
                                "signing_bonus","guaranteed","apy"]].copy()
            for col in ["cap_number","base_salary","signing_bonus","guaranteed","apy"]:
                ct_disp[col] = ct_disp[col].apply(
                    lambda x: f"${x/1e6:.2f}M" if pd.notna(x) and x>0 else "—")
            ct_disp.columns = ["Player","Pos","Cap #","Base","Signing Bonus","Guaranteed","APY"]
            st.dataframe(ct_disp, hide_index=True, use_container_width=True)
        else:
            st.info("No contract data available for this team.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — EDA & METHODOLOGY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔬 EDA & Methodology":
    st.title("🔬 EDA & Methodology")

    tab_e1, tab_e2, tab_e3 = st.tabs(["📈 Key EDA Charts", "🛡️ Leakage Audit", "📋 Methodology Summary"])

    with tab_e1:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Home Field Advantage by Season")
            fig_hfa = go.Figure()
            fig_hfa.add_trace(go.Scatter(
                x=hfa["season"], y=hfa["home_win_pct"], mode="lines+markers",
                line=dict(color=BLUE, width=2.5), marker=dict(size=7, color=BLUE),
                fill="tozeroy", fillcolor="rgba(37,99,235,0.08)"))
            fig_hfa.add_hline(y=hfa["home_win_pct"].mean(), line_dash="dot",
                line_color=SLATE, annotation_text=f"Avg {hfa['home_win_pct'].mean():.1%}",
                annotation_font_color=SLATE, annotation_position="top right")
            hfa_2020 = hfa[hfa["season"]==2020]["home_win_pct"].values
            if len(hfa_2020):
                fig_hfa.add_annotation(x=2020, y=float(hfa_2020[0]),
                    text="No fans<br>(COVID)", showarrow=True, arrowhead=2,
                    arrowcolor=RED, font=dict(color=RED, size=11), ax=40, ay=-32)
            fig_hfa.update_layout(
                xaxis=dict(title="Season", dtick=1, tickangle=-45),
                yaxis=dict(title="Home win rate", tickformat=".0%"))
            st.plotly_chart(clean(fig_hfa, h=340), use_container_width=True)

        with col2:
            st.subheader("Vegas Spread Accuracy by Season")
            # Compute from schedules
            sched_all = pd.read_json(f"{DATA}/sched_2025.json")
            # Show static note
            st.info(f"Vegas spread accuracy (2025 test): **{vegas_acc:.1%}**\n\n"
                    "The betting market correctly picks the winner ~65–72% of the time — "
                    "the baseline any model must aim to beat.")
            # Show simple bar comparing model vs vegas
            models_to_show = {
                "XGBoost": model_res["cls_noveg"]["XGBoost"]["test_acc"],
                "Rand Forest": model_res["cls_noveg"]["Random Forest"]["test_acc"],
                "Neural Net": model_res["cls_noveg"]["Neural Network"]["test_acc"],
                "XGB (Vegas)": model_res["cls_vegas"]["XGBoost (Vegas)"]["test_acc"],
                "Vegas Spread": vegas_acc,
            }
            fig_vb = go.Figure(go.Bar(
                x=list(models_to_show.keys()), y=list(models_to_show.values()),
                marker_color=[BLUE,TEAL,GREEN,PURPLE,AMBER], marker_line_width=0,
                text=[f"{v:.1%}" for v in models_to_show.values()],
                textposition="outside", textfont=dict(size=11)))
            fig_vb.update_layout(yaxis=dict(title="Test Accuracy",
                tickformat=".0%", range=[0.55, 0.73]))
            st.plotly_chart(clean(fig_vb, h=260), use_container_width=True)

        st.subheader("ELO System — Yards Per Play Signal")
        st.markdown("""
The ELO system separates **offensive and defensive ratings** per team.

**Key design choices:**
- **Signal:** Net yards per play (passing + rushing, net of sack yards) — more informative than raw wins
- **Threshold:** Season-to-date rolling median YPP to classify each game as above/below average
- **Regression:** Ratings regress 1/3 back to 1500 between seasons
- **Week 1 seed:** Historical fallback of 5.5 net YPP for the first game before any season data exists

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
        st.subheader("🛡️ Data Leakage Audit")
        st.markdown("Every potential source of leakage was explicitly checked before reporting accuracy.")
        audit = [
            ("Rolling features", "Using game N stats to predict game N",
             "✅ Fixed", ".shift(1) on every expanding mean; Week 1 seeded from prior season"),
            ("SOS adjustment", "Normalising game N by opponent's full-season average",
             "✅ Fixed", "Pre-game rolling opponent quality; Week 1 seeded from prior season"),
            ("ELO update", "Recording post-game ELO as pre-game feature",
             "✅ Fixed", "ELO snapshot taken before update in every game loop"),
            ("Playoff stakes flags", "Using final clinch/elimination status",
             "✅ Fixed", "True pre-game snapshot built by processing games in chronological order"),
            ("Train/test contamination", "Imputing test NAs with test median",
             "✅ Fixed", "fillna uses train-split median only"),
            ("Target in features", "point_diff, team_score, win accessible in feature table",
             "✅ Fixed", "None appear in FEATURES or FEATURES_VEGAS lists"),
            ("Vegas lines in base model", "spread_line_team leaking into base model",
             "✅ Fixed", "Excluded from FEATURES; only in FEATURES_VEGAS"),
            ("XGBoost early stopping", "eval_set using 2025 test set",
             "✅ Fixed", "OOT slice (2023-2024 W1-17) used for early stopping; 2025 never seen"),
        ]
        df_audit = pd.DataFrame(audit, columns=["Risk","How it would leak","Status","Resolution"])
        st.dataframe(df_audit, hide_index=True, use_container_width=True,
                     column_config={
                         "Status": st.column_config.TextColumn(width="small"),
                         "Resolution": st.column_config.TextColumn(width="large"),
                     })
        st.success("All leakage vectors identified and resolved. Results represent genuine out-of-time performance on the full 2025 season.")

    with tab_e3:
        st.subheader("📋 Methodology Summary")
        methodology = {
            "Data Source": "Pro Football Reference via nflverse (2018–2025)",
            "Training Window": "2018–2024, Weeks 1–17 only",
            "Test Set": "2025 Weeks 1–17 (272 games, full completed season)",
            "Week 18 Treatment": "Evaluated separately (motivational confound)",
            "Models — No Vegas": "XGBoost, Random Forest, Logistic Regression, Ridge Classifier, Neural Network (MLP)",
            "Models — Vegas": "Same 5 models with spread_line_team + total_line added",
            "Regressors": "Same 5 architectures predicting point differential → converted to win/loss",
            "Hyperparameter Tuning": "Walk-forward CV (4 folds: 2019–2022 hold-one-out)",
            "XGBoost Early Stopping": "OOT slice (2023-2024); 2025 test never touched during training",
            "ELO Signal": "Net yards per play (v10: replaces first downs for efficiency purity)",
            "Uncertainty": "5,000 Monte Carlo simulations per team → 80% confidence interval on final wins",
            "Cap Analysis": "Post-model descriptive only (OTC 2025); no influence on predictions",
        }
        df_meth = pd.DataFrame(methodology.items(), columns=["Component","Detail"])
        st.dataframe(df_meth, hide_index=True, use_container_width=True,
                     column_config={"Detail": st.column_config.TextColumn(width="large")})

        st.subheader("Known Limitations")
        limitations = [
            "Red zone efficiency stats unavailable at game level in PFR",
            "EPA per play (gold-standard efficiency metric) not in PFR",
            "Clinch/elimination flags use a wins≥10/losses≥10 proxy; exact playoff math is more complex",
            "Walk-forward CV covers 4 folds (2019–2022); extending back requires pre-2018 advanced stats unavailable in PFR",
            "A Bayesian or Kalman filter ELO would produce honest uncertainty bands on ELO ratings themselves",
            "Vegas spread accuracy is a high bar — the market aggregates sharp bettor information this model cannot replicate without lines",
        ]
        for lim in limitations:
            st.markdown(f"• {lim}")

        st.subheader("References")
        refs = [
            "Pro Football Reference (pro-football-reference.com) — all model input data",
            "nflverse project (nflverse.com) — data pipeline and hosting",
            "FiveThirtyEight NFL ELO methodology — season-carryover and regression approach",
            "Over The Cap (overthecap.com) — 2025 positional cap spending (supplementary only)",
        ]
        for r in refs:
            st.markdown(f"• {r}")
