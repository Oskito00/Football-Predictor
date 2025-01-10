import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib  # for saving the LabelEncoder

# Load and preprocess data
df = pd.read_csv('sportradar/data/processed_data/training_data_advanced.csv')

# Drop non-feature columns
df_features = df.drop(["fixture_id"], axis=1)

# Label encode competition_id
label_encoder = LabelEncoder()
df_features["competition_id"] = label_encoder.fit_transform(df_features["competition_id"])

# Save the label encoder for future use
joblib.dump(label_encoder, 'sportradar/AI/competition_id_encoder.joblib')

# Create derived difference features
df_features["goals_scored_difference"] = df_features["average_home_goals_scored"] - df_features["average_away_goals_scored"]
df_features["goals_conceded_difference"] = df_features["average_home_goals_conceded"] - df_features["average_away_goals_conceded"]
df_features["win_rate_difference"] = df_features["average_home_win_rate"] - df_features["average_away_win_rate"]
df_features["squad_strength_difference"] = df_features["home_squad_strength"] - df_features["away_squad_strength"]
df_features["fatigue_difference"] = df_features["home_fatigue"] - df_features["away_fatigue"]
df_features["h2h_points_difference"] = df_features["home_h2h_avg_points"] - df_features["away_h2h_avg_points"]
df_features["pass_effectiveness_difference"] = df_features["home_pass_effectiveness"] - df_features["away_pass_effectiveness"]
df_features["shot_accuracy_difference"] = df_features["home_shot_accuracy"] - df_features["away_shot_accuracy"]
df_features["conversion_rate_difference"] = df_features["home_conversion_rate"] - df_features["away_conversion_rate"]
df_features["defensive_success_difference"] = df_features["home_defensive_success"] - df_features["away_defensive_success"]
df_features["clean_sheets_difference"] = df_features["average_home_clean_sheets"] - df_features["average_away_clean_sheets"]
df_features["h2h_goals_difference"] = df_features["home_h2h_avg_goals"] - df_features["away_h2h_avg_goals"]
df_features["h2h_clean_sheets_difference"] = df_features["home_h2h_avg_clean_sheets"] - df_features["away_h2h_avg_clean_sheets"]

# Drop redundant original features
columns_to_drop = [
    "average_home_goals_scored", "average_away_goals_scored",
    "average_home_goals_conceded", "average_away_goals_conceded",
    "average_home_win_rate", "average_away_win_rate",
    "home_squad_strength", "away_squad_strength",
    "home_fatigue", "away_fatigue",
    "home_h2h_avg_points", "away_h2h_avg_points",
    "home_pass_effectiveness", "away_pass_effectiveness",
    "home_shot_accuracy", "away_shot_accuracy",
    "home_conversion_rate", "away_conversion_rate",
    "home_defensive_success", "away_defensive_success",
    "average_home_clean_sheets", "average_away_clean_sheets",
    "home_h2h_avg_goals", "away_h2h_avg_goals",
    "home_h2h_avg_clean_sheets", "away_h2h_avg_clean_sheets"
]
df_features = df_features.drop(columns=columns_to_drop)

# Automatically move 'home_goals' and 'away_goals' to the end
target_columns = ["home_goals", "away_goals"]
feature_columns = [col for col in df_features.columns if col not in target_columns]
df_features = df_features[feature_columns + target_columns]

# Save preprocessed features
df_features.to_csv('sportradar/AI/preprocessed_features.csv', index=False)

print("Preprocessed features saved!")