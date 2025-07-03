import streamlit as st
import joblib
import pandas as pd
import numpy as np

model_classifier = joblib.load('nba_win_classifier_3.0.pkl')
model_regressor = joblib.load('nba_margin_regressor_2.0.pkl')
team_stats_df = pd.read_csv('TeamStatistics.csv')
player_stats_df = pd.read_csv('PlayerStatistics.csv')

def get_latest_team_stats(team_name):
    team_data = team_stats_df[team_stats_df['teamName'] == team_name].sort_values('gameDate')
    if len(team_data) == 0:
        st.error(f"No stats found for {team_name}")
        return None
    return team_data.iloc[-1]

def get_avg_player_stats(team_name):
    try:
        latest_game = player_stats_df[player_stats_df['playerteamName'] == team_name].sort_values('gameDate').iloc[-1]['gameId']
        players = player_stats_df[(player_stats_df['playerteamName'] == team_name) & (player_stats_df['gameId'] == latest_game)]
        return {
            'avgPlayerPoints': players['points'].mean(),
            'avgPlayerMinutes': players['numMinutes'].mean()
        }
    except:
        return {
            'avgPlayerPoints': 0,
            'avgPlayerMinutes': 0
        }

def build_feature(home_team, away_team, X_columns):
    home_stats = get_latest_team_stats(home_team)
    away_stats = get_latest_team_stats(away_team)

    if home_stats is None or away_stats is None:
        return None

    home_stats = home_stats.add_prefix('home_')
    away_stats = away_stats.add_prefix('away_')
    combined = pd.concat([home_stats, away_stats])

    home_player_stats = get_avg_player_stats(home_team)
    away_player_stats = get_avg_player_stats(away_team)

    for k, v in home_player_stats.items():
        combined[f'home_{k}'] = v
    for k, v in away_player_stats.items():
        combined[f'away_{k}'] = v

    feature_df = pd.DataFrame([combined])
    feature_df = feature_df.select_dtypes(include=[np.number])

    missing_cols = set(X_columns) - set(feature_df.columns)
    for col in missing_cols:
        feature_df[col] = 0

    feature_df = feature_df[X_columns]

    return feature_df

st.title("🏀 NBA Game Predictor")
st.write("Select two teams below to predict the winner and margin!")

team_list = sorted(team_stats_df['teamName'].unique())

home_team = st.selectbox("Home Team", team_list)
away_team = st.selectbox("Away Team", team_list)

if home_team and away_team and home_team != away_team:
    if st.button("Predict Winner and Margin"):
        X_columns = model_classifier.feature_names_in_
        features = build_feature(home_team, away_team, X_columns)

        if features is not None:
            win_prediction = model_classifier.predict(features)[0]
            win_proba = model_classifier.predict_proba(features)[0]
            margin_prediction = model_regressor.predict(features)[0]

            st.subheader("Prediction Result")
            if win_prediction == 1:
                st.success(f"🏠 {home_team} is predicted to WIN!")
                st.write(f"Estimated margin of victory: **{margin_prediction:.2f} points**")
            else:
                st.success(f"🛫 {away_team} is predicted to WIN!")
                st.write(f"Estimated margin of victory: **{abs(margin_prediction):.2f} points**")

            st.write(f"Confidence (Home Team Win): **{win_proba[1]:.2%}**")
else:
    st.warning("Please select two different teams!")
