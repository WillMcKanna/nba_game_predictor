import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import os

# --- Load artifacts ---
# Paths work both locally (same folder) and in the repo (models/ folder)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_artifact(filename):
    # Try same folder first (local), then models/ folder (repo structure)
    local_path = os.path.join(BASE_DIR, filename)
    repo_path = os.path.join(BASE_DIR, "..", "models", filename)
    if os.path.exists(local_path):
        return local_path
    elif os.path.exists(repo_path):
        return repo_path
    else:
        raise FileNotFoundError(f"Could not find {filename}")

model = pickle.load(open(load_artifact("nba_model.pkl"), "rb"))
scaler = pickle.load(open(load_artifact("scaler.pkl"), "rb"))
latest_stats = pd.read_csv(load_artifact("latest_stats.csv"))

# --- Config ---
stat_cols = ["PTS", "FG_PCT", "FG3_PCT", "FT_PCT", "AST", "REB"]
stat_labels = {
    "PTS": "Points",
    "FG_PCT": "Field Goal %",
    "FG3_PCT": "3-Point %",
    "FT_PCT": "Free Throw %",
    "AST": "Assists",
    "REB": "Rebounds"
}

# --- UI ---
st.title("🏀 NBA Game Predictor")
st.markdown("Select a home and away team to predict the winner based on recent performance.")

teams = sorted(latest_stats["TEAM_NAME"].dropna().unique())

col1, col2 = st.columns(2)
with col1:
    home_team = st.selectbox("🏠 Home Team", teams)
with col2:
    away_team = st.selectbox("✈️ Away Team", teams, index=1)

if home_team == away_team:
    st.warning("Please select two different teams.")
    st.stop()

if st.button("Predict", use_container_width=True):

    # --- Get each team's latest rolling averages ---
    home = latest_stats[latest_stats["TEAM_NAME"] == home_team].iloc[-1]
    away = latest_stats[latest_stats["TEAM_NAME"] == away_team].iloc[-1]

    # --- Compute differentials ---
    diffs = np.array([[home[col] - away[col] for col in stat_cols]])
    diffs_scaled = scaler.transform(diffs)

    # --- Predict ---
    prob = model.predict_proba(diffs_scaled)[0]
    home_prob = prob[1]
    away_prob = prob[0]

    # --- Display result ---
    st.divider()
    winner = home_team if home_prob >= 0.5 else away_team
    win_prob = max(home_prob, away_prob)

    st.subheader(f"Predicted Winner: {winner}")
    st.metric("Win Probability", f"{win_prob * 100:.1f}%")

    col1, col2 = st.columns(2)
    with col1:
        st.metric(f"{home_team} (Home)", f"{home_prob * 100:.1f}%")
    with col2:
        st.metric(f"{away_team} (Away)", f"{away_prob * 100:.1f}%")

    # --- SHAP ---
    st.divider()
    st.subheader("Why this prediction?")

    # Build SHAP explainer using training data distribution
    background = scaler.transform(
        latest_stats[stat_cols].dropna().values
    )
    explainer = shap.LinearExplainer(model, background)
    shap_values = explainer.shap_values(diffs_scaled)

    # Waterfall plot
    explanation = shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        feature_names=[stat_labels[c] for c in stat_cols]
    )
    fig, ax = plt.subplots()
    shap.plots.waterfall(explanation, show=False)
    st.pyplot(fig)
    plt.close()

    # Written sentence
    shap_pairs = list(zip(shap_values[0], stat_cols))
    shap_pairs.sort(key=lambda x: abs(x[0]), reverse=True)
    top = shap_pairs[0]
    second = shap_pairs[1]

    top_label = stat_labels[top[1]]
    second_label = stat_labels[second[1]]
    top_dir = "disadvantage" if top[0] > 0 else "advantage"
    second_dir = "disadvantage" if second[0] > 0 else "advantage"

    st.markdown(
        f"**{winner}** is predicted to win primarily because of a "
        f"**{top_label}** {top_dir} for the home team, "
        f"followed by a **{second_label}** {second_dir}."
    )