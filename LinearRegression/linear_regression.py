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
from sklearn.impute import SimpleImputer

def load_and_prepare_data(start_year, end_year):
    """Load and combine data from multiple years"""
    all_data = []
    
    for year in range(start_year, end_year + 1):
        try:
            df = pd.read_csv(f'LinearRegression/data/{year}/training_data.csv')
            df['season'] = year
            all_data.append(df)
            print(f"Loaded data for {year}")
        except FileNotFoundError:
            print(f"No data found for {year}")
            continue
    
    if not all_data:
        raise ValueError("No data found")
    
    return pd.concat(all_data, ignore_index=True)

def add_historical_features(df):
    """Add historical performance features"""
    # Calculate rolling averages (last 5 games)
    window = 5
    df['rolling_home_goals'] = df.groupby('home_team')['home_goals'].transform(
        lambda x: x.rolling(window, min_periods=1).mean()
    )
    df['rolling_away_goals'] = df.groupby('away_team')['away_goals'].transform(
        lambda x: x.rolling(window, min_periods=1).mean()
    )
    
    # Add form features (based on chronological order)
    df['home_team_form'] = df.groupby('home_team')['home_goals'].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )
    df['away_team_form'] = df.groupby('away_team')['away_goals'].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )
    
    # Add win/loss streaks
    df['home_team_wins'] = df.groupby('home_team').apply(
        lambda x: (x['home_goals'] > x['away_goals']).rolling(window, min_periods=1).sum()
    ).reset_index(level=0, drop=True)
    
    df['away_team_wins'] = df.groupby('away_team').apply(
        lambda x: (x['away_goals'] > x['home_goals']).rolling(window, min_periods=1).sum()
    ).reset_index(level=0, drop=True)
    
    return df

def analyze_feature_importance(features, targets):
    """Analyze and visualize feature importance"""
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

def clean_data(df):
    """Clean data by handling NaN values"""
    # Create imputer for numerical values
    imputer = SimpleImputer(strategy='mean')
    
    # Get numeric columns
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    
    # Impute missing values
    df[numeric_columns] = imputer.fit_transform(df[numeric_columns])
    
    return df

def main():
    # Load historical and current data
    print("Loading historical data...")
    historical_data = load_and_prepare_data(2014, 2023)
    print("Loading current season data...")
    current_data = pd.read_csv('LinearRegression/data/2024/training_data.csv')
    
    # Add historical features to both datasets
    print("Adding historical features...")
    historical_data = add_historical_features(historical_data)
    current_data = add_historical_features(current_data)
    
    # Clean data
    print("Cleaning data...")
    current_data = clean_data(current_data)
    
    # Prepare features (including existing h2h features)
    feature_cols = [col for col in current_data.columns 
                   if col not in ['fixture_id', 'home_team', 'away_team', 
                                'home_goals', 'away_goals', 'season']]
    
    # Analyze feature importance
    correlations = analyze_feature_importance(current_data[feature_cols], 
                                           current_data[['home_goals', 'away_goals']])
    print("\nTop 10 most important features based on correlation:")
    print(correlations.head(10))
    
    # Select important features
    correlation_threshold = 0.1
    important_features = correlations[correlations['mean_abs_corr'] > correlation_threshold].index
    
    # Split current data into train/test sets
    X = current_data[important_features]
    y = current_data[['home_goals', 'away_goals']]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True
    )
    
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = Ridge(alpha=1.0)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    train_predictions = model.predict(X_train_scaled)
    test_predictions = model.predict(X_test_scaled)
    
    # Calculate and print metrics
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
    
    # Save predictions
    predictions = []
    test_df_indices = X_test.index
    
    for i, idx in enumerate(test_df_indices):
        prediction = {
            'fixture': {
                'home_team': current_data.iloc[idx]['home_team'],
                'away_team': current_data.iloc[idx]['away_team'],
                'fixture_id': current_data.iloc[idx]['fixture_id']
            },
            'predictions': {
                'home_goals': float(test_predictions[i][0]),
                'away_goals': float(test_predictions[i][1])
            }
        }
        predictions.append(prediction)
    
    # Create predictions directory if it doesn't exist
    os.makedirs('predictions', exist_ok=True)
    
    # Save predictions to JSON file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'predictions/predictions_{timestamp}.json'
    with open(filename, 'w') as f:
        json.dump({'predictions': predictions}, f, indent=4)
    
    print(f"\nPredictions have been saved to {filename}")
    
    # Print example predictions
    print("\nExample Predictions vs Actual:")
    for i in range(min(5, len(predictions))):
        pred = predictions[i]
        actual_idx = test_df_indices[i]
        print(f"\nMatch: {pred['fixture']['home_team']} vs {pred['fixture']['away_team']}")
        print(f"Predicted: {pred['predictions']['home_goals']:.2f} - {pred['predictions']['away_goals']:.2f}")
        print(f"Actual: {current_data.iloc[actual_idx]['home_goals']} - {current_data.iloc[actual_idx]['away_goals']}")

if __name__ == "__main__":
    main()