import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import json
from datetime import datetime
import os

# Load 2024 data
df = pd.read_csv('LinearRegression/data/2024/training_data.csv')
print(f"\nLoaded 2024 data with {len(df)} samples")

# Separate features and targets
features = df.iloc[:, 3:-2]  # Exclude identifiers and labels
targets = df[["home_goals", "away_goals"]]

# Analyze feature importance using correlation analysis
def analyze_feature_importance(features, targets):
    corr_home = features.apply(lambda x: x.corr(targets['home_goals']))
    corr_away = features.apply(lambda x: x.corr(targets['away_goals']))
    
    correlations = pd.DataFrame({
        'home_goals_corr': corr_home,
        'away_goals_corr': corr_away
    })
    
    correlations['mean_abs_corr'] = correlations.abs().mean(axis=1)
    correlations = correlations.sort_values('mean_abs_corr', ascending=False)
    
    # Plot correlation heatmap
    plt.figure(figsize=(12, 8))
    plt.imshow(correlations[['home_goals_corr', 'away_goals_corr']].head(15).values.T, 
               cmap='coolwarm', aspect='auto')
    plt.colorbar()
    plt.yticks([0, 1], ['home_goals_corr', 'away_goals_corr'])
    plt.xticks(range(15), correlations.index[:15], rotation=45, ha='right')
    plt.title('Top 15 Feature Correlations with Target Variables')
    plt.tight_layout()
    plt.savefig('feature_correlations.png')
    plt.close()
    
    return correlations

# Analyze and select features
correlations = analyze_feature_importance(features, targets)
print("\nTop 10 most important features based on correlation:")
print(correlations.head(10))

# Select features with absolute correlation > threshold
correlation_threshold = 0.1
important_features = correlations[correlations['mean_abs_corr'] > correlation_threshold].index
X = features[important_features]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, targets, test_size=0.2, random_state=42
)

print(f"\nTraining set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame to keep column names
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# Train Ridge regression model
alpha = 1.0  # Regularization strength
model = Ridge(alpha=alpha)
model.fit(X_train_scaled, y_train)

# Make predictions
train_predictions = model.predict(X_train_scaled)
test_predictions = model.predict(X_test_scaled)

# Save predictions to JSON
predictions = []
for i, idx in enumerate(X_test.index):
    prediction = {
        'fixture': {
            'home_team': df.iloc[idx]['home_team'],
            'away_team': df.iloc[idx]['away_team'],
            'fixture_id': df.iloc[idx]['fixture_id']
        },
        'predictions': {
            'home_goals': test_predictions[i][0],
            'away_goals': test_predictions[i][1]
        }
    }
    predictions.append(prediction)

# Create predictions directory if it doesn't exist
os.makedirs('predictions', exist_ok=True)

# Generate timestamp
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Save predictions to JSON file
filename = f'predictions/predictions_{timestamp}.json'
with open(filename, 'w') as f:
    json.dump({'predictions': predictions}, f, indent=4)

print(f"\nPredictions have been saved to {filename}")

# Calculate metrics
print("\nModel Performance:")
print("Training Set:")
print(f"Home Goals RMSE: {np.sqrt(mean_squared_error(y_train['home_goals'], train_predictions[:, 0])):.3f}")
print(f"Away Goals RMSE: {np.sqrt(mean_squared_error(y_train['away_goals'], train_predictions[:, 1])):.3f}")
print(f"Home Goals R2: {r2_score(y_train['home_goals'], train_predictions[:, 0]):.3f}")
print(f"Away Goals R2: {r2_score(y_train['away_goals'], train_predictions[:, 1]):.3f}")

print("\nTest Set:")
print(f"Home Goals RMSE: {np.sqrt(mean_squared_error(y_test['home_goals'], test_predictions[:, 0])):.3f}")
print(f"Away Goals RMSE: {np.sqrt(mean_squared_error(y_test['away_goals'], test_predictions[:, 1])):.3f}")
print(f"Home Goals R2: {r2_score(y_test['home_goals'], test_predictions[:, 0]):.3f}")
print(f"Away Goals R2: {r2_score(y_test['away_goals'], test_predictions[:, 1]):.3f}")

# Feature importance based on model coefficients
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance_home_goals': model.coef_[:, 0],
    'Importance_away_goals': model.coef_[:, 1]
})
feature_importance['Absolute_Mean_Importance'] = feature_importance[['Importance_home_goals', 'Importance_away_goals']].abs().mean(axis=1)
feature_importance = feature_importance.sort_values('Absolute_Mean_Importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

# Plot feature importance
plt.figure(figsize=(12, 8))
importance_data = feature_importance.head(15)
plt.barh(range(len(importance_data)), importance_data['Absolute_Mean_Importance'])
plt.yticks(range(len(importance_data)), importance_data['Feature'])
plt.title('Top 15 Feature Importance based on Ridge Regression Coefficients')
plt.xlabel('Absolute Mean Importance')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

