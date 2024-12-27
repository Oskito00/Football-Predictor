import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import json
from datetime import datetime
import os
import matplotlib.pyplot as plt

def analyze_feature_importance(features, targets):
    """Analyze feature importance based on correlations."""
    corr_home = features.apply(lambda x: x.corr(targets['home_goals']))
    corr_away = features.apply(lambda x: x.corr(targets['away_goals']))
    
    correlations = pd.DataFrame({
        'home_goals_corr': corr_home,
        'away_goals_corr': corr_away
    })
    
    correlations['mean_abs_corr'] = correlations.abs().mean(axis=1)
    correlations = correlations.sort_values('mean_abs_corr', ascending=False)
    
    # Plot correlations
    plt.figure(figsize=(15, 10))
    
    # Create heatmap-style plot
    plt.subplot(1, 2, 1)
    plt.imshow(correlations[['home_goals_corr', 'away_goals_corr']].head(15).values.T, 
               cmap='coolwarm', aspect='auto')
    plt.colorbar(label='Correlation')
    plt.yticks([0, 1], ['Home Goals', 'Away Goals'])
    plt.xticks(range(15), correlations.index[:15], rotation=45, ha='right')
    plt.title('Top 15 Feature Correlations')
    
    # Create bar plot
    plt.subplot(1, 2, 2)
    correlations['mean_abs_corr'].head(15).plot(kind='bar')
    plt.xticks(range(15), correlations.index[:15], rotation=45, ha='right')
    plt.title('Top 15 Features by Mean Absolute Correlation')
    plt.ylabel('Mean Absolute Correlation')
    
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    
    # Print correlation details
    print("\nFeature Correlations:")
    print("--------------------")
    for idx, row in correlations.iterrows():
        print(f"{idx:30} | Home: {row['home_goals_corr']:6.3f} | Away: {row['away_goals_corr']:6.3f} | Mean: {row['mean_abs_corr']:6.3f}")
    
    return correlations

def save_predictions(predictions, actual_values, fixture_data, is_test=True):
    """Save predictions to JSON file."""
    predictions_list = []
    
    for i in range(len(predictions)):
        prediction = {
            'fixture': {
                'home_team': fixture_data.iloc[i]['home_team'],
                'away_team': fixture_data.iloc[i]['away_team'],
                'fixture_id': fixture_data.iloc[i]['fixture_id']
            },
            'predictions': {
                'home_goals': float(predictions[i][0]),
                'away_goals': float(predictions[i][1])
            }
        }
        
        if is_test:
            prediction['actual'] = {
                'home_goals': float(actual_values.iloc[i]['home_goals']),
                'away_goals': float(actual_values.iloc[i]['away_goals'])
            }
        
        predictions_list.append(prediction)
    
    # Save predictions
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'predictions/{"test" if is_test else "future"}_predictions_{timestamp}.json'
    os.makedirs('predictions', exist_ok=True)
    
    with open(filename, 'w') as f:
        json.dump({'predictions': predictions_list}, f, indent=4)
    
    print(f"\nPredictions saved to {filename}")

def plot_learning_curve(X, y, model, scaler, train_sizes_rel=np.linspace(0.1, 1.0, 10)):
    """Plot learning curves showing model performance vs training set size."""
    n_samples = len(X)
    
    # Use a fixed test set (20% of data)
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Fit scaler on full training data and transform test data
    X_train_full_scaled = scaler.fit_transform(X_train_full)
    X_test_scaled = scaler.transform(X_test)
    
    # Calculate train sizes based on training data size, not full dataset
    train_sizes = [int(train_size * len(X_train_full)) for train_size in train_sizes_rel]
    
    train_rmse_home, train_rmse_away = [], []
    test_rmse_home, test_rmse_away = [], []
    train_r2_home, train_r2_away = [], []
    test_r2_home, test_r2_away = [], []
    
    for train_size in train_sizes:
        # Randomly sample training data of current size
        indices = np.random.choice(len(X_train_full), size=train_size, replace=False)
        X_train_subset = X_train_full_scaled[indices]
        y_train_subset = y_train_full.iloc[indices]
        
        # Train model
        model.fit(X_train_subset, y_train_subset)
        
        # Make predictions
        train_pred = model.predict(X_train_subset)
        test_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        train_rmse_home.append(np.sqrt(mean_squared_error(y_train_subset['home_goals'], train_pred[:, 0])))
        train_rmse_away.append(np.sqrt(mean_squared_error(y_train_subset['away_goals'], train_pred[:, 1])))
        test_rmse_home.append(np.sqrt(mean_squared_error(y_test['home_goals'], test_pred[:, 0])))
        test_rmse_away.append(np.sqrt(mean_squared_error(y_test['away_goals'], test_pred[:, 1])))
        
        train_r2_home.append(r2_score(y_train_subset['home_goals'], train_pred[:, 0]))
        train_r2_away.append(r2_score(y_train_subset['away_goals'], train_pred[:, 1]))
        test_r2_home.append(r2_score(y_test['home_goals'], test_pred[:, 0]))
        test_r2_away.append(r2_score(y_test['away_goals'], test_pred[:, 1]))
    
    # Plot learning curves
    plt.figure(figsize=(15, 10))
    
    # Plot RMSE
    plt.subplot(2, 1, 1)
    plt.plot(train_sizes, train_rmse_home, 'b-', label='Train Home RMSE')
    plt.plot(train_sizes, test_rmse_home, 'b--', label='Test Home RMSE')
    plt.plot(train_sizes, train_rmse_away, 'r-', label='Train Away RMSE')
    plt.plot(train_sizes, test_rmse_away, 'r--', label='Test Away RMSE')
    plt.xlabel('Training Set Size')
    plt.ylabel('RMSE')
    plt.title('Learning Curves - RMSE vs Training Size')
    plt.legend()
    plt.grid(True)
    
    # Plot R²
    plt.subplot(2, 1, 2)
    plt.plot(train_sizes, train_r2_home, 'b-', label='Train Home R²')
    plt.plot(train_sizes, test_r2_home, 'b--', label='Test Home R²')
    plt.plot(train_sizes, train_r2_away, 'r-', label='Train Away R²')
    plt.plot(train_sizes, test_r2_away, 'r--', label='Test Away R²')
    plt.xlabel('Training Set Size')
    plt.ylabel('R²')
    plt.title('Learning Curves - R² vs Training Size')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('learning_curves.png')
    plt.close()

def main():
    # Load 2024 data
    print("Loading 2024 data...")
    data = pd.read_csv('LinearRegression/data/2024/training_data.csv')
    
    # Shuffle the data
    data = data.sample(frac=1, random_state=np.random.randint(1000))
    print(f"Random seed for this run: {data.index[0]}")
    
    # Get feature columns (exclude non-feature columns and labels)
    feature_cols = [col for col in data.columns 
                   if col not in ['fixture_id', 'home_team', 'away_team', 
                                'home_goals', 'away_goals']]
    print("\nAll available features:")
    print("----------------------")
    for col in sorted(feature_cols):
        print(col)
    
    # Prepare features and targets
    X = data[feature_cols]
    y = data[['home_goals', 'away_goals']]
    
    # Initialize K-Fold
    k_folds = 5
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=np.random.randint(1000))
    
    # Initialize metrics storage
    fold_metrics = {
        'train_home_rmse': [], 'train_away_rmse': [],
        'train_home_r2': [], 'train_away_r2': [],
        'test_home_rmse': [], 'test_away_rmse': [],
        'test_home_r2': [], 'test_away_r2': []
    }
    
    print(f"\nPerforming {k_folds}-fold cross-validation...")
    
    # Perform k-fold cross-validation
    for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
        print(f"\nFold {fold}/{k_folds}")
        
        # Split data
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        data_test = data.iloc[test_idx]
        
        # Use all features
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = Ridge(alpha=1.0)
        model.fit(X_train_scaled, y_train)
        
        # Make predictions on both train and test sets
        train_predictions = model.predict(X_train_scaled)
        

        test_predictions = model.predict(X_test_scaled)
        save_predictions(test_predictions, y, data, is_test=True)
        
        # Calculate metrics for training set
        train_home_rmse = np.sqrt(mean_squared_error(y_train['home_goals'], train_predictions[:, 0]))
        train_away_rmse = np.sqrt(mean_squared_error(y_train['away_goals'], train_predictions[:, 1]))
        train_home_r2 = r2_score(y_train['home_goals'], train_predictions[:, 0])
        train_away_r2 = r2_score(y_train['away_goals'], train_predictions[:, 1])
        
        # Calculate metrics for test set
        test_home_rmse = np.sqrt(mean_squared_error(y_test['home_goals'], test_predictions[:, 0]))
        test_away_rmse = np.sqrt(mean_squared_error(y_test['away_goals'], test_predictions[:, 1]))
        test_home_r2 = r2_score(y_test['home_goals'], test_predictions[:, 0])
        test_away_r2 = r2_score(y_test['away_goals'], test_predictions[:, 1])
        
        # Store metrics
        fold_metrics['train_home_rmse'].append(train_home_rmse)
        fold_metrics['train_away_rmse'].append(train_away_rmse)
        fold_metrics['train_home_r2'].append(train_home_r2)
        fold_metrics['train_away_r2'].append(train_away_r2)
        fold_metrics['test_home_rmse'].append(test_home_rmse)
        fold_metrics['test_away_rmse'].append(test_away_rmse)
        fold_metrics['test_home_r2'].append(test_home_r2)
        fold_metrics['test_away_r2'].append(test_away_r2)
        
        print(f"\nFold {fold} Metrics:")
        print("Training Set:")
        print(f"Home Goals RMSE: {train_home_rmse:.3f}")
        print(f"Away Goals RMSE: {train_away_rmse:.3f}")
        print(f"Home Goals R²: {train_home_r2:.3f}")
        print(f"Away Goals R²: {train_away_r2:.3f}")
        
        print("\nTest Set:")
        print(f"Home Goals RMSE: {test_home_rmse:.3f}")
        print(f"Away Goals RMSE: {test_away_rmse:.3f}")
        print(f"Home Goals R²: {test_home_r2:.3f}")
        print(f"Away Goals R²: {test_away_r2:.3f}")
    
    # Plot training vs test metrics
    plt.figure(figsize=(15, 10))
    
    # Plot RMSE
    plt.subplot(2, 1, 1)
    folds = range(1, k_folds + 1)
    plt.plot(folds, fold_metrics['train_home_rmse'], 'b-', label='Train Home RMSE')
    plt.plot(folds, fold_metrics['test_home_rmse'], 'b--', label='Test Home RMSE')
    plt.plot(folds, fold_metrics['train_away_rmse'], 'r-', label='Train Away RMSE')
    plt.plot(folds, fold_metrics['test_away_rmse'], 'r--', label='Test Away RMSE')
    plt.xlabel('Fold')
    plt.ylabel('RMSE')
    plt.title('Training vs Test RMSE Across Folds')
    plt.legend()
    plt.grid(True)
    
    # Plot R²
    plt.subplot(2, 1, 2)
    plt.plot(folds, fold_metrics['train_home_r2'], 'b-', label='Train Home R²')
    plt.plot(folds, fold_metrics['test_home_r2'], 'b--', label='Test Home R²')
    plt.plot(folds, fold_metrics['train_away_r2'], 'r-', label='Train Away R²')
    plt.plot(folds, fold_metrics['test_away_r2'], 'r--', label='Test Away R²')
    plt.xlabel('Fold')
    plt.ylabel('R²')
    plt.title('Training vs Test R² Across Folds')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_vs_test_metrics.png')
    plt.close()

    # Print average metrics
    print("\nAverage Performance Across All Folds:")
    print("\nTraining Set:")
    print(f"Home RMSE: {np.mean(fold_metrics['train_home_rmse']):.3f} (±{np.std(fold_metrics['train_home_rmse']):.3f})")
    print(f"Away RMSE: {np.mean(fold_metrics['train_away_rmse']):.3f} (±{np.std(fold_metrics['train_away_rmse']):.3f})")
    print(f"Home R²: {np.mean(fold_metrics['train_home_r2']):.3f} (±{np.std(fold_metrics['train_home_r2']):.3f})")
    print(f"Away R²: {np.mean(fold_metrics['train_away_r2']):.3f} (±{np.std(fold_metrics['train_away_r2']):.3f})")
    
    print("\nTest Set:")
    print(f"Home RMSE: {np.mean(fold_metrics['test_home_rmse']):.3f} (±{np.std(fold_metrics['test_home_rmse']):.3f})")
    print(f"Away RMSE: {np.mean(fold_metrics['test_away_rmse']):.3f} (±{np.std(fold_metrics['test_away_rmse']):.3f})")
    print(f"Home R²: {np.mean(fold_metrics['test_home_r2']):.3f} (±{np.std(fold_metrics['test_home_r2']):.3f})")
    print(f"Away R²: {np.mean(fold_metrics['test_away_r2']):.3f} (±{np.std(fold_metrics['test_away_r2']):.3f})")
    
    # After k-fold cross-validation and before future predictions
    print("\nGenerating learning curves...")
    plot_learning_curve(X, y, Ridge(alpha=1.0), StandardScaler())
    
    # Train final model on all data for future predictions
    print("\nTraining final model on all data...")
    
    # Use all features for final model
    X_scaled = scaler.fit_transform(X)
    model = Ridge(alpha=1.0)
    model.fit(X_scaled, y)
    
    # Predict future fixtures
    print("\nPredicting future fixtures...")
    future_data = pd.read_csv('LinearRegression/data/2024/future_fixtures_prepared.csv')
    X_future = future_data[feature_cols]
    X_future_scaled = scaler.transform(X_future)
    future_predictions = model.predict(X_future_scaled)
    
    # Save future predictions
    save_predictions(future_predictions, None, future_data, is_test=False)

if __name__ == "__main__":
    main()