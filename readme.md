# <span style="color: #1E90FF;">Premier League Football Predictor</span>

A machine learning project aimed at predicting the outcomes of Premier League matches. The primary goal is to forecast match results (win, lose, or draw) rather than exact scores, utilizing various statistical features extracted from historical match data.

## <span style="color: #32CD32;">Overview</span>

This project scrapes data from the [Premier League Stats website](https://www.premierleague.com/) and processes it for match prediction. Past results for the most recent year are collected via API requests. The dataset includes all statistics for every game played so far in the season.

## <span style="color: #FFD700;">Data Preprocessing and Feature Engineering</span>

- **Feature Extraction:**
  - Key statistics such as <span style="color: #FF4500;">conversion rate</span>, <span style="color: #FF4500;">pass accuracy</span>, and <span style="color: #FF4500;">momentum</span> were extracted.
- **Feature Selection:**
  - Focused on the most relevant data to avoid overfitting due to an excessive number of features.
  - Selected the <span style="color: #4682B4;">top 15 out of 28 features</span> that were the most predictive.
- **Recent Form Metrics:**
  - Captured recent form as a metric to reflect teams’ current performance levels.
- **Head-to-Head (H2H) Statistics:**
  - Included past H2H results between teams to capture trends not reflected in recent form alone.
  - H2H trends proved to be highly predictive features.

## <span style="color: #FF69B4;">Modeling Approach</span>

- **Initial Model:**
  - Started with Linear Regression for simplicity and ease of debugging.
- **Custom Loss Function:**
  - Created a custom loss function that evaluates predictions based on the correctness of the match result rather than the exact score difference.
  - This approach aligns the model’s focus with the primary goal of predicting match outcomes.
- **Overfitting Mitigation:**
  - Observed high variance in the model, indicating overfitting and poor generalization to test data.
  - Applied L2 regularization (<span style="color: #8A2BE2;">Ridge Regression</span>) to reduce overfitting.
- **Feature Engineering:**
  - Enhanced the predictive power of features through feature engineering techniques.

## <span style="color: #DC143C;">Evaluation Metrics</span>

- **Initial Metrics:**
  - **Mean Squared Error (MSE):** Measured how far off the predicted goals were from the actual goals scored.
  - **R-squared (R²):** Indicated the proportion of the total variance in the actual outcomes accounted for by the model (the closer to 1, the better).
- **Custom Metric:**
  - Shifted to an evaluation metric that assesses whether the match result prediction (win, lose, draw) is correct.
  - This change better reflects the project’s primary objective.

## <span style="color: #FFA500;">Results</span>

- **Accuracy Achieved:**
  - Reached a <span style="color: #008000;">51% average accuracy</span> for match result predictions using the current model.
- **Observations:**
  - Including data from past seasons reduced accuracy, suggesting that recent data is more predictive.
  - Head-to-head statistics were among the top predictive features. This will be revisited as it seems intuitive that past data would be useful.

## <span style="color: #4B0082;">Future Work</span>

To improve the model’s accuracy, the following enhancements are planned:

- **Player Performance Metrics:**
  - Key player form and fitness levels.
  - Identification of key players based on goals assisted/scored; potentially using fantasy points.
  - Impact of key players being injured or suspended.
- **Team Dynamics:**
  - Historical performance against similar formations.
  - Stability of the starting lineup.
  - Impact of recent transfers in or out of the club.
- **Financial Factors:**
  - Club’s financial health.
- **Fatigue and Recovery:**
  - Minutes played per player in the last five matches.
  - Recovery time since the last match.
- **External Factors:**
  - Weather conditions.
  - Average yellow cards per match.
  - Importance of the match (e.g., critical matches, derbies).
  - Substitution impact.
- **Sentiment Analysis:**
  - Manager sentiment analysis.
  - Key player sentiment analysis.
  - Social media sentiment analysis (e.g., Twitter).
- **Synthetic Data:**
  - Add synthetic data taken from the same distribution.

## <span style="color: #00CED1;">Acknowledgments</span>

- Applied principles from Andrew Ng on evaluating models based on bias and variance to improve model performance.

## <span style="color: #ADFF2F;">Installation and Usage</span>

#TODO