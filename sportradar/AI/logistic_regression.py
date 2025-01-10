import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def run_experiment_3_class(n_runs=10):
    """
    Train a 3-class classifier (Home Win, Draw, Away Win) over multiple runs,
    record accuracies, and print details for each match in the test set.
    """
    accuracies = []
    home_win_accuracies = []  # Baseline: always predict Home Win

    for run_i in range(n_runs):
        # 1. Load your preprocessed data
        df = pd.read_csv("sportradar/AI/preprocessed_features.csv")
        # Columns (for reference):
        # start_time, home_team, away_team, competition_id, match_importance,
        # goals_scored_difference, goals_conceded_difference, win_rate_difference,
        # squad_strength_difference, fatigue_difference, h2h_points_difference,
        # pass_effectiveness_difference, shot_accuracy_difference, conversion_rate_difference,
        # defensive_success_difference, clean_sheets_difference, h2h_goals_difference,
        # h2h_clean_sheets_difference, home_goals, away_goals

        # 2. Create a 3-class outcome label: 2 = Home Win, 1 = Draw, 0 = Away Win
        def outcome_label(home_g, away_g):
            if home_g > away_g:
                return 2  # Home Win
            elif home_g < away_g:
                return 0  # Away Win
            else:
                return 1  # Draw

        df["outcome"] = [
            outcome_label(h, a) for h, a in zip(df["home_goals"], df["away_goals"])
        ]

        # We'll keep `start_time, home_team, away_team` for printing later
        metadata_cols = ["start_time", "home_team", "away_team", "home_goals", "away_goals", "outcome"]
        
        # 3. Separate features (X) and target (y)
        #    We drop the metadata columns + original goals from the features
        X = df.drop(columns=metadata_cols)
        y = df["outcome"]  # Our 3-class label

        # 4. Scale the features (optional, but generally helps tree-based methods less than linear)
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )

        # 5. Split into train/test
        random_state = np.random.randint(0, 1000)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=random_state
        )

        # Keep test set metadata for printing predictions
        df_test_metadata = df.loc[X_test.index, metadata_cols]

        # 6. Train a 3-class classifier
        #    Example: RandomForest with a small hyperparameter grid
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [None, 5, 10]
        }
        clf = GridSearchCV(
            RandomForestClassifier(random_state=42),
            param_grid,
            cv=3,
            scoring='accuracy'
        )
        clf.fit(X_train, y_train)

        # Best model after grid search
        best_model = clf.best_estimator_

        # 7. Predict on the test set
        y_pred = best_model.predict(X_test)

        # 8. Evaluate model accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        # Baseline: always predict Home Win (class = 2)
        y_baseline = np.full_like(y_test, fill_value=2)  # 2 => Home Win
        home_win_accuracy = accuracy_score(y_test, y_baseline)

        accuracies.append(accuracy)
        home_win_accuracies.append(home_win_accuracy)

        # Print run info
        print(f"Run {run_i+1}/{n_runs}")
        print(f"  Best Params: {clf.best_params_}")
        print(f"  Model Accuracy: {accuracy:.2%}, Home Win Baseline: {home_win_accuracy:.2%}")
        
        
        # 9. (Optional) Print each match in the test set
        #    We'll show actual vs. predicted outcome + actual scores
        #    outcome = 2 (Home Win), 1 (Draw), 0 (Away Win)
        outcome_map = {0: "Away Win", 1: "Draw", 2: "Home Win"}

        # print("  Test Matches Detail:")
        # for idx, pred_label in zip(X_test.index, y_pred):
        #     row = df_test_metadata.loc[idx]
        #     actual_label = row["outcome"]
        #     start_time = row["start_time"]
        #     home_team = row["home_team"]
        #     away_team = row["away_team"]
        #     actual_home_goals = row["home_goals"]
        #     actual_away_goals = row["away_goals"]

        #     pred_outcome = outcome_map[pred_label]
        #     actual_outcome = outcome_map[actual_label]
        #     print(
        #         f"    {start_time}: {home_team} vs {away_team} | "
        #         f"Predicted: {pred_outcome}, Actual: {actual_outcome} | "
        #         f"Score: {actual_home_goals}-{actual_away_goals}"
        #     )
        # print("-"*60)

    # 10. After all runs, compute average accuracy
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    mean_home_win_acc = np.mean(home_win_accuracies)

    print(f"\nFinal Results Over {n_runs} Runs:")
    print(f"Mean Model Accuracy: {mean_acc:.2%}")
    print(f"Home Win Baseline:  {mean_home_win_acc:.2%}")
    print(f"Std Dev:            {std_acc:.2%}")


run_experiment_3_class(n_runs=10)