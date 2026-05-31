# NFL Win Prediction — 2025 Season
### Chicago Bears Football Analytics Interview Project

An interactive Streamlit app visualizing machine learning models that predict NFL team win totals throughout the 2025 season, with rolling updates after each game.

---

## Features

| Tab | What it shows |
|-----|--------------|
| 🏈 **Team Deep Dive** | Per-team projection evolution, game log, ELO ratings, rolling efficiency metrics, week-by-week win probability. Bears default. |
| 📊 **Model Comparison** | 20-model comparison (5 classifiers × no-vegas/vegas + 5 regressors × no-vegas/vegas) vs. Vegas baseline. Feature importance. Linear vs. non-linear gap. |
| 🗺️ **League Projections** | All 32 teams ranked with 80% Monte Carlo CI bands, division breakdowns, week-by-week projection table. |
| 💰 **Cap Spending** | OTC positional spending by team and win quadrant. Delta vs. league average. Radar charts. Player contracts. |
| 🔬 **EDA & Methodology** | Home field advantage, leakage audit table, full methodology summary. |

---

## Methodology Summary

- **Data:** Pro Football Reference via nflverse (2018–2025 regular seasons)
- **Train:** 2018–2024, Weeks 1–17 | **Test:** Full 2025 season, Weeks 1–17
- **Models:** XGBoost, Random Forest, Logistic Regression, Ridge Classifier, Neural Network (MLP) — each in no-vegas and vegas-augmented variants
- **ELO signal:** Net yards per play (leakage-free, pre-game threshold)
- **Features:** 42 base features across ELO, rolling efficiency, matchup differentials, SOS-adjusted metrics, and context flags
- **Uncertainty:** 5,000 Monte Carlo simulations → 80% CI on projected final win totals
- **Hyperparameter tuning:** Walk-forward CV (4 temporal folds)
- **Cap analysis:** Post-model descriptive only (Over The Cap data) — no influence on predictions

---

## Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Data Sources

- **Pro Football Reference** (pro-football-reference.com) — all model input data
- **nflverse** (nflverse.com) — data pipeline
- **Over The Cap** (overthecap.com) — 2025 cap spending (supplementary)
- FiveThirtyEight NFL ELO methodology — season regression approach
