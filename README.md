**NBA Game Predictor**
This is a Streamlit app that predicts the winner of an NBA game, along with the expected margin of victory. It uses recent team and player statistics and two pre-trained machine learning models.

You pick a home team and an away team, and the app tells you:
- Which team is more likely to win
- How many points they’ll win by
- How confident the model is in its prediction

***How it works***
The app uses two models:
- A classification model to predict the winner (home or away)
- A regression model to estimate the point margin
It pulls the latest stats for each team from CSV files and builds a feature set that combines:
- Recent team game stats
- Average player performance from the latest game
That feature set is passed to both models, and the results are shown in the app.

***How to run it***
Make sure these files are in the same folder:
1. app.py
2. nba_win_classifier_3.0.pkl
3. nba_margin_regressor_2.0.pkl
4. TeamStatistics.csv
5. PlayerStatistics.csv

Install the required Python packages:
- pip install streamlit pandas numpy scikit-learn
- Run the app with: streamlit run app.py

Then open the URL it gives you (usually http://localhost:8501) in your browser.

***What you’ll see***
Once you select the two teams and click the Predict button, the app will:
- Load the latest team and player data
- Predict who will win and by how much
- Show a confidence score for the prediction
