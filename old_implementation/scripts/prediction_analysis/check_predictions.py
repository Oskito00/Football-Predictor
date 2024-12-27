import json
import math

def load_fixtures(file_path):
    """Load fixtures from a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)['fixtures']

def load_predictions(file_path):
    """Load predictions from a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)['predictions']

def evaluate_predictions(fixtures, predictions):
    """Evaluate predictions against actual results."""
    correct_results = 0
    correct_scores = 0
    correct_over_under = 0
    total_evaluated = 0
    
    print("\nDetailed Predictions vs Actual Results:")
    print("----------------------------------------")

    for fixture in fixtures:
        fixture_id = fixture['fixture_id']
        actual_home_goals = int(fixture['home_team']['score'])
        actual_away_goals = int(fixture['away_team']['score'])

        # Find the corresponding prediction
        prediction = next((p for p in predictions if p['fixture']['fixture_id'] == fixture_id), None)
        if prediction:
            # Round predicted goals to nearest integer
            predicted_home_goals = round(prediction['predictions']['home_goals'])
            predicted_away_goals = round(prediction['predictions']['away_goals'])

            # Check if the predicted result matches the actual result
            actual_result = 'D' if actual_home_goals == actual_away_goals else ('H' if actual_home_goals > actual_away_goals else 'A')
            predicted_result = 'D' if predicted_home_goals == predicted_away_goals else ('H' if predicted_home_goals > predicted_away_goals else 'A')

            # Print match details
            print(f"\n{prediction['fixture']['home_team']} vs {prediction['fixture']['away_team']}")
            print(f"Predicted: {predicted_home_goals}-{predicted_away_goals} ({predicted_result})")
            print(f"Actual: {actual_home_goals}-{actual_away_goals} ({actual_result})")
            
            if actual_result == predicted_result:
                correct_results += 1
                print("✓ Correct result")
            else:
                print("✗ Incorrect result")

            # Check if the predicted score matches the actual score
            if actual_home_goals == predicted_home_goals and actual_away_goals == predicted_away_goals:
                correct_scores += 1
                print("✓ Correct score")
            else:
                print("✗ Incorrect score")
                
            # Check over/under 2.5 goals
            actual_total_goals = actual_home_goals + actual_away_goals
            predicted_total_goals = predicted_home_goals + predicted_away_goals
            actual_over_under = 'Over' if actual_total_goals > 2.5 else 'Under'
            predicted_over_under = 'Over' if predicted_total_goals > 2.5 else 'Under'
            
            if actual_over_under == predicted_over_under:
                correct_over_under += 1
                print(f"✓ Correct over/under 2.5 prediction ({predicted_over_under})")
            else:
                print(f"✗ Incorrect over/under 2.5 prediction (Predicted: {predicted_over_under}, Actual: {actual_over_under})")
                
            total_evaluated += 1

    return correct_results, correct_scores, correct_over_under, total_evaluated

if __name__ == "__main__":
    fixtures_file = 'data/2024/premier_league_results.json'
    predictions_file = 'predictions/fold_3_test_predictions_20241205_175414.json'
    fixtures = load_fixtures(fixtures_file)
    predictions = load_predictions(predictions_file)

    correct_results, correct_scores, correct_over_under, total_evaluated = evaluate_predictions(fixtures, predictions)

    print("\nSummary:")
    print("--------")
    print(f"Correct match results: {correct_results} out of {total_evaluated} ({(correct_results/total_evaluated)*100:.1f}%)")
    print(f"Correct exact scores: {correct_scores} out of {total_evaluated} ({(correct_scores/total_evaluated)*100:.1f}%)")
    print(f"Correct over/under 2.5 goals predictions: {correct_over_under} out of {total_evaluated} ({(correct_over_under/total_evaluated)*100:.1f}%)") 