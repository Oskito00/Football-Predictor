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

# Load your data
file_path = './data/training_data.csv'
df = pd.read_csv(file_path)

# Load test data
test_path = './data/test_data.csv'
test_df = pd.read_csv(test_path)

# Separate features and targets for training data
features = df.iloc[:, 3:-2]  # Exclude identifiers and labels
targets = df[["home_goals", "away_goals"]]

# Feature importance using correlation analysis
def analyze_feature_importance(features, targets):
    # Calculate correlation with both targets
    corr_home = features.apply(lambda x: x.corr(targets['home_goals']))
    corr_away = features.apply(lambda x: x.corr(targets['away_goals']))
    
    # Combine correlations
    correlations = pd.DataFrame({
        'home_goals_corr': corr_home,
        'away_goals_corr': corr_away
    })
    
    # Calculate absolute mean correlation for each feature
    correlations['mean_abs_corr'] = correlations.abs().mean(axis=1)
    
    # Sort by absolute mean correlation
    correlations = correlations.sort_values('mean_abs_corr', ascending=False)
    
    # Plot correlation heatmap using matplotlib
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

# Prepare test features (using same columns as training)
X_test = test_df[important_features]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame to keep column names
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# Train Ridge regression model on full training data
alpha = 1.0  # Regularization strength
model = Ridge(alpha=alpha)
model.fit(X_scaled, targets)

# Make predictions on test data
test_predictions = model.predict(X_test_scaled)

# Format predictions
def format_predictions(test_df, y_pred):
    """Format predictions into a JSON structure"""
    predictions = []
    
    for i, (_, row) in enumerate(test_df.iterrows()):
        prediction = {
            'fixture': {
                'home_team': row['home_team'],
                'away_team': row['away_team'],
                'fixture_id': row['fixture_id']
            },
            'predictions': {
                'home_goals': (y_pred[i][0]),
                'away_goals': (y_pred[i][1])
            }
        }
        predictions.append(prediction)
    
    return predictions

# Format and save predictions
predictions = format_predictions(test_df, test_predictions)

# Create predictions directory if it doesn't exist
os.makedirs('predictions', exist_ok=True)

# Generate timestamp
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Save predictions to JSON file
filename = f'predictions/predictions_{timestamp}.json'
with open(filename, 'w') as f:
    json.dump({'predictions': predictions}, f, indent=4)

print(f"\nPredictions have been saved to {filename}")

# Print first few predictions as example
print("\nExample predictions:")
for pred in predictions[:3]:
    print(f"\n{pred['fixture']['home_team']} vs {pred['fixture']['away_team']}:")
    print(f"Predicted score: {pred['predictions']['home_goals']} - {pred['predictions']['away_goals']}")

# Feature importance based on model coefficients
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance_home_goals': model.coef_[:, 0],
    'Importance_away_goals': model.coef_[:, 1]
})
feature_importance['Absolute_Mean_Importance'] = feature_importance[['Importance_home_goals', 'Importance_away_goals']].abs().mean(axis=1)
feature_importance = feature_importance.sort_values('Absolute_Mean_Importance', ascending=False)

print("\nFeature Importance based on Ridge Regression coefficients:")
print(feature_importance)

# Plot feature importance using matplotlib
plt.figure(figsize=(12, 8))
importance_data = feature_importance.head(15)
plt.barh(range(len(importance_data)), importance_data['Absolute_Mean_Importance'])
plt.yticks(range(len(importance_data)), importance_data['Feature'])
plt.title('Top 15 Feature Importance based on Ridge Regression Coefficients')
plt.xlabel('Absolute Mean Importance')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()