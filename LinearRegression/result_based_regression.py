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

def custom_result_loss(y_true, y_pred):
    """
    Custom loss function that focuses solely on match results (win/draw/loss).
    Rounds predicted scores to integers before comparing results.
    
    Returns the number of correctly predicted results.
    """
    correct_results = 0
    
    for i in range(len(y_true)):
        true_home = y_true.iloc[i]['home_goals']
        true_away = y_true.iloc[i]['away_goals']
        pred_home = round(y_pred[i][0])  # Round to nearest integer
        pred_away = round(y_pred[i][1])  # Round to nearest integer
        
        # Determine actual and predicted results
        true_result = 'D' if true_home == true_away else ('H' if true_home > true_away else 'A')
        pred_result = 'D' if pred_home == pred_away else ('H' if pred_home > pred_away else 'A')
        
        # Count correct results
        if true_result == pred_result:
            correct_results += 1
    
    return correct_results / len(y_true)

class ResultBasedRidge(Ridge):
    """Custom Ridge regression that optimizes for match results."""
    
    def fit(self, X, y):
        # Initial fit using standard Ridge regression with L2 regularization
        super().fit(X, y)
        
        # Fine-tune using custom loss function
        learning_rate = 0.01
        n_iterations = 100
        
        for _ in range(n_iterations):
            y_pred = self.predict(X)
            
            # Calculate gradients for both home and away goals separately
            grad_home = np.zeros_like(self.coef_[0])
            grad_away = np.zeros_like(self.coef_[1])
            
            for i in range(len(X)):
                true_home = y.iloc[i]['home_goals']
                true_away = y.iloc[i]['away_goals']
                pred_home = y_pred[i][0]
                pred_away = y_pred[i][1]
                
                true_result = 'D' if true_home == true_away else ('H' if true_home > true_away else 'A')
                pred_result = 'D' if round(pred_home) == round(pred_away) else ('H' if round(pred_home) > round(pred_away) else 'A')
                
                # Adjust gradient based on result prediction accuracy
                multiplier = 2 if true_result != pred_result else 1
                
                # Update gradients separately for home and away predictions
                grad_home += multiplier * 2 * (pred_home - true_home) * X[i]
                grad_away += multiplier * 2 * (pred_away - true_away) * X[i]
            
            # Update coefficients with L2 regularization
            self.coef_[0] -= learning_rate * (grad_home / len(X) + self.alpha * self.coef_[0])
            self.coef_[1] -= learning_rate * (grad_away / len(X) + self.alpha * self.coef_[1])
        
        return self
    
    def feature_importance(self):
        """Calculate feature importance based on absolute coefficient values."""
        return np.abs(self.coef_)

def save_fold_predictions(predictions, actual_values, fixture_data, fold_number, is_test=True):
    """Save predictions for each fold to JSON file."""
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
    filename = f'predictions/fold_{fold_number}_{"test" if is_test else "train"}_predictions_{timestamp}.json'
    os.makedirs('predictions', exist_ok=True)
    
    with open(filename, 'w') as f:
        json.dump({'predictions': predictions_list}, f, indent=4)
    
    print(f"\nFold {fold_number} {'Test' if is_test else 'Train'} predictions saved to {filename}")

def select_important_features(X, feature_importance, threshold_percentile):
    """
    Select features based on their importance scores.
    Always include H2H features.
    Returns the selected feature indices and their names.
    """
    # If percentile is 0, use all features
    if threshold_percentile == 0:
        return np.arange(X.shape[1]), X.columns.tolist()
    
    # Get the average importance across both home and away predictions
    avg_importance = np.mean(np.abs(feature_importance), axis=0)
    
    # Calculate the importance threshold based on percentile
    threshold = np.percentile(avg_importance, threshold_percentile)
    
    # Get indices of important features
    important_indices = np.where(avg_importance >= threshold)[0]
    
    # Always include H2H features
    h2h_features = [i for i, col in enumerate(X.columns) if 'h2h_' in col]
    important_indices = np.unique(np.concatenate((important_indices, h2h_features)))
    
    # Get names of important features
    feature_names = X.columns[important_indices].tolist()
    
    return important_indices, feature_names

def main():
    # Load 2024 data
    print("Loading 2024 data...")
    data = pd.read_csv('LinearRegression/data/2024/training_data.csv')
    
    # Prepare features (X) and targets (y)
    feature_cols = [col for col in data.columns 
                   if col not in ['fixture_id', 'home_team', 'away_team', 
                                'home_goals', 'away_goals']]
    X = data[feature_cols]
    y = data[['home_goals', 'away_goals']]
    
    # Initialize k-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Use the best alpha value and percentile
    best_alpha = 2.0
    percentile = 50
    
    print("\nTraining with selected features...")
    all_train_accuracies = []
    all_test_accuracies = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        data_test = data.iloc[test_idx]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Get feature importance from training data only
        model = ResultBasedRidge(alpha=best_alpha)
        model.fit(X_train_scaled, y_train)
        feature_importance = model.feature_importance()
        
        # Select important features using only training data
        important_indices, important_features = select_important_features(X_train, feature_importance, percentile)
        
        if fold == 1:
            print(f"\nNumber of features selected: {len(important_features)}")
            print("\nSelected Important Features (from first fold):")
            for idx, feature in enumerate(important_features):
                importance_score = np.mean(np.abs(feature_importance[:, important_indices[idx]]))
                print(f"{feature}: {importance_score:.4f}")
        
        # Use only important features
        X_train_important = X_train_scaled[:, important_indices]
        X_test_important = X_test_scaled[:, important_indices]
        
        # Train model with important features
        model = ResultBasedRidge(alpha=best_alpha)
        model.fit(X_train_important, y_train)
        
        # Make predictions
        train_predictions = model.predict(X_train_important)
        test_predictions = model.predict(X_test_important)
        
        # Save predictions
        save_fold_predictions(train_predictions, y_train, data.iloc[train_idx], fold, is_test=False)
        save_fold_predictions(test_predictions, y_test, data_test, fold, is_test=True)
        
        # Calculate metrics
        train_accuracy = custom_result_loss(y_train, train_predictions)
        test_accuracy = custom_result_loss(y_test, test_predictions)
        
        all_train_accuracies.append(train_accuracy)
        all_test_accuracies.append(test_accuracy)
        
        print(f"Fold {fold} - Train Accuracy: {train_accuracy:.3f}, Test Accuracy: {test_accuracy:.3f}")
    
    # Print final summary
    print("\nFinal Results:")
    print(f"Average Train Accuracy: {np.mean(all_train_accuracies):.3f} ± {np.std(all_train_accuracies):.3f}")
    print(f"Average Test Accuracy: {np.mean(all_test_accuracies):.3f} ± {np.std(all_test_accuracies):.3f}")
    print(f"\nFinal model trained with alpha = {best_alpha}")

def predict_future_fixtures():
    """Predict future fixtures using the trained model."""
    print("\nPredicting future fixtures...")
    
    # Load training data
    print("Loading training data...")
    data = pd.read_csv('LinearRegression/data/2024/training_data.csv')
    
    # Prepare features (X) and targets (y)
    feature_cols = [col for col in data.columns 
                   if col not in ['fixture_id', 'home_team', 'away_team', 
                                'home_goals', 'away_goals']]
    X = data[feature_cols]
    y = data[['home_goals', 'away_goals']]
    
    # Use the best alpha value and percentile
    best_alpha = 2.0
    percentile = 50
    
    # First get feature importance from initial model
    print("\nSelecting important features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    initial_model = ResultBasedRidge(alpha=best_alpha)
    initial_model.fit(X_scaled, y)
    feature_importance = initial_model.feature_importance()
    
    # Select important features
    important_indices, important_features = select_important_features(X, feature_importance, percentile)
    print(f"\nNumber of features selected: {len(important_features)}")
    print("\nSelected Important Features:")
    for idx, feature in enumerate(important_features):
        importance_score = np.mean(np.abs(feature_importance[:, important_indices[idx]]))
        print(f"{feature}: {importance_score:.4f}")
    
    # Train final model using only important features
    print("\nTraining final model with selected features...")
    X_important = X_scaled[:, important_indices]
    final_model = ResultBasedRidge(alpha=best_alpha)
    final_model.fit(X_important, y)
    
    # Load and prepare future fixtures
    print("\nLoading future fixtures...")
    future_data = pd.read_csv('LinearRegression/data/2024/future_fixtures_prepared.csv')
    X_future = future_data[feature_cols]
    
    # Scale future features and select important ones
    X_future_scaled = scaler.transform(X_future)
    X_future_important = X_future_scaled[:, important_indices]
    
    # Make predictions
    print("\nMaking predictions...")
    predictions = final_model.predict(X_future_important)
    
    # Save predictions
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'predictions/future_predictions_{timestamp}.json'
    
    predictions_list = []
    for i in range(len(predictions)):
        prediction = {
            'fixture': {
                'home_team': future_data.iloc[i]['home_team'],
                'away_team': future_data.iloc[i]['away_team'],
                'fixture_id': future_data.iloc[i]['fixture_id']
            },
            'predictions': {
                'home_goals': float(predictions[i][0]),
                'away_goals': float(predictions[i][1])
            }
        }
        predictions_list.append(prediction)
    
    os.makedirs('predictions', exist_ok=True)
    with open(filename, 'w') as f:
        json.dump({'predictions': predictions_list}, f, indent=4)
    
    print(f"\nFuture predictions saved to {filename}")
    print(f"Final model trained with alpha = {best_alpha} using {len(important_features)} important features")

def find_optimal_parameters():
    """Find optimal alpha and percentile values using grid search."""
    print("\nFinding optimal parameters...")
    
    # Load 2024 data
    print("Loading 2024 data...")
    data = pd.read_csv('LinearRegression/data/2024/training_data.csv')
    
    # Prepare features (X) and targets (y)
    feature_cols = [col for col in data.columns 
                   if col not in ['fixture_id', 'home_team', 'away_team', 
                                'home_goals', 'away_goals']]
    X = data[feature_cols]
    y = data[['home_goals', 'away_goals']]
    
    # Initialize k-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Define parameter grid
    alphas = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    percentiles = [0, 25, 50, 75]  # 0 means use all features
    
    best_params = {'alpha': None, 'percentile': None}
    best_test_accuracy = 0
    results = []
    
    # Grid search
    for alpha in alphas:
        for percentile in percentiles:
            print(f"\nTesting alpha={alpha}, percentile={percentile}")
            train_accuracies = []
            test_accuracies = []
            
            for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Get feature importance from training data only
                model = ResultBasedRidge(alpha=alpha)
                model.fit(X_train_scaled, y_train)
                feature_importance = model.feature_importance()
                
                # Select important features using only training data
                important_indices, _ = select_important_features(X_train, feature_importance, percentile)
                
                # Use only important features
                X_train_important = X_train_scaled[:, important_indices]
                X_test_important = X_test_scaled[:, important_indices]
                
                # Train model with important features
                model = ResultBasedRidge(alpha=alpha)
                model.fit(X_train_important, y_train)
                
                # Make predictions
                train_predictions = model.predict(X_train_important)
                test_predictions = model.predict(X_test_important)
                
                # Calculate accuracies
                train_accuracy = custom_result_loss(y_train, train_predictions)
                test_accuracy = custom_result_loss(y_test, test_predictions)
                train_accuracies.append(train_accuracy)
                test_accuracies.append(test_accuracy)
            
            # Calculate average accuracies across folds
            avg_train_accuracy = np.mean(train_accuracies)
            avg_test_accuracy = np.mean(test_accuracies)
            std_test_accuracy = np.std(test_accuracies)
            
            results.append({
                'alpha': alpha,
                'percentile': percentile,
                'train_accuracy': avg_train_accuracy,
                'test_accuracy': avg_test_accuracy,
                'std': std_test_accuracy
            })
            
            print(f"Average Train Accuracy: {avg_train_accuracy:.3f}, "
                  f"Average Test Accuracy: {avg_test_accuracy:.3f} ± {std_test_accuracy:.3f}")
            
            # Update best parameters if necessary
            if avg_test_accuracy > best_test_accuracy:
                best_test_accuracy = avg_test_accuracy
                best_params['alpha'] = alpha
                best_params['percentile'] = percentile
    
    # Print results summary
    print("\nResults Summary:")
    print("\nAll Results (sorted by test accuracy):")
    sorted_results = sorted(results, key=lambda x: x['test_accuracy'], reverse=True)
    for result in sorted_results:
        print(f"Alpha: {result['alpha']}, Percentile: {result['percentile']}, "
              f"Train Accuracy: {result['train_accuracy']:.3f}, "
              f"Test Accuracy: {result['test_accuracy']:.3f} ± {result['std']:.3f}")
    
    print(f"\nBest Parameters:")
    print(f"Alpha: {best_params['alpha']}")
    print(f"Percentile: {best_params['percentile']}")
    print(f"Best Test Accuracy: {best_test_accuracy:.3f}")
    
    return best_params

if __name__ == "__main__":
    # Find optimal parameters
    # best_params = find_optimal_parameters()
    
    # Run main training with best parameters
    main()
    
    # Make future predictions with best parameters
    predict_future_fixtures()
