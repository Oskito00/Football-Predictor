import json
import math
import pandas as pd
import numpy as np

def calculate_team_stats(match_stats):
    """Extract key statistics from a team's match data"""
    return {
        'possession': match_stats.get('possession_percentage', 0),
        'shots_on_target': match_stats.get('ontarget_scoring_att', 0),
        'passes_accurate': match_stats.get('accurate_pass', 0),
        'passes': match_stats.get('total_pass', 0),
        'big_chances_created': match_stats.get('big_chance_created', 0),
        'tackles': match_stats.get('total_tackle', 0),
        'saves': match_stats.get('saves', 0)
    }

#TODO Sort the improving/declining to make sure it works properly
def calculate_momentum(matches, team_id):
    """
    Calculate team momentum based on recent results
    Returns: 
        1 (declining form)
        2 (stable form)
        3 (improving form)
    """

    if not matches:
        return 2  # neutral momentum if no matches
        
    # Get points from each match in chronological order
    points = []
    for match in matches:
        is_home = match['home_team']['id'] == team_id
        team_score = match['home_team']['score'] if is_home else match['away_team']['score']
        opponent_score = match['away_team']['score'] if is_home else match['home_team']['score']
        
        if team_score > opponent_score:
            points.append(3)
        elif team_score == opponent_score:
            points.append(1)
        else:
            points.append(0)
    
    # Compare first half with second half of recent matches
    mid_point = len(points) // 2
    
    if len(points) % 2 == 0:
        recent_form = sum(points[:mid_point]) / max(1, mid_point)
        early_form = sum(points[mid_point:]) / max(1, len(points) - mid_point)
    else:
        recent_form = sum(points[:mid_point + 1]) / max(1, mid_point + 1)
        early_form = sum(points[mid_point:]) / max(1, len(points) - mid_point)
    
    # Determine trend
    if recent_form > early_form:  # Improving
        return 3
    elif recent_form < early_form:  # Declining
        return 1
    return 2  # Stable

def extract_team_recent_form(fixtures, team_id, before_timestamp, max_matches=5):
    """Get key recent form statistics including momentum"""
    previous_matches = [
        f for f in fixtures 
        if f['kickoff']['timestamp'] < before_timestamp and
        (f['home_team']['id'] == team_id or f['away_team']['id'] == team_id)
    ]
    
    previous_matches.sort(key=lambda x: x['kickoff']['timestamp'], reverse=True)
    recent_matches = previous_matches[:max_matches]
    
    # Calculate momentum first
    momentum = calculate_momentum(recent_matches, team_id)
    
    # Default values when no matches found
    if not recent_matches:
        return {
            'matches_played': 0,
            'points': 0,
            'goals_scored': 0,
            'goals_conceded': 0,
            'avg_possession': 0,
            'avg_shots_on_target': 0,
            'avg_pass_accuracy': 0,
            'avg_big_chances': 0,
            'avg_tackles': 0,
            'avg_saves': 0,
            'momentum': momentum
        }
    
    # Rest of the function remains the same
    total_points = 0
    total_goals_scored = 0
    total_goals_conceded = 0
    total_possession = 0
    total_shots_on_target = 0
    total_pass_accuracy = 0
    total_big_chances = 0
    total_tackles = 0
    total_saves = 0
    
    for match in recent_matches:
        is_home = match['home_team']['id'] == team_id
        team_stats = match['stats']['match_stats']['home' if is_home else 'away']
        opponent_stats = match['stats']['match_stats']['away' if is_home else 'home']
        
        # Basic match results
        team_score = match['home_team']['score'] if is_home else match['away_team']['score']
        opponent_score = match['away_team']['score'] if is_home else match['home_team']['score']
        
        total_goals_scored += team_score
        total_goals_conceded += opponent_score
        
        if team_score > opponent_score:
            total_points += 3
        elif team_score == opponent_score:
            total_points += 1
        
        if opponent_score == 0:
            total_possession += calculate_team_stats(team_stats)['possession']
            total_shots_on_target += calculate_team_stats(team_stats)['shots_on_target']
            total_pass_accuracy += (calculate_team_stats(team_stats)['passes_accurate'] / calculate_team_stats(team_stats)['passes'] * 100) if calculate_team_stats(team_stats)['passes'] > 0 else 0
            total_big_chances += calculate_team_stats(team_stats)['big_chances_created']
            total_tackles += calculate_team_stats(team_stats)['tackles']
            total_saves += calculate_team_stats(team_stats)['saves']
    
    matches_played = len(recent_matches)
    
    # Calculate averages
    avg_possession = total_possession / matches_played if matches_played > 0 else 0
    avg_shots_on_target = total_shots_on_target / matches_played if matches_played > 0 else 0
    avg_pass_accuracy = total_pass_accuracy / matches_played if matches_played > 0 else 0
    avg_big_chances = total_big_chances / matches_played if matches_played > 0 else 0
    avg_tackles = total_tackles / matches_played if matches_played > 0 else 0
    avg_saves = total_saves / matches_played if matches_played > 0 else 0
    
    return {
        'matches_played': matches_played,
        'points': total_points,
        'goals_scored': total_goals_scored,
        'goals_conceded': total_goals_conceded,
        'avg_possession': avg_possession,
        'avg_shots_on_target': avg_shots_on_target,
        'avg_pass_accuracy': avg_pass_accuracy,
        'avg_big_chances': avg_big_chances,
        'avg_tackles': avg_tackles,
        'avg_saves': avg_saves,
        'momentum': momentum
    }

def get_h2h_features(home_team_name, away_team_name):
    """Retrieve head-to-head statistics for a given fixture"""
    
    # Load H2H stats
    with open('team_data/all_h2h_stats.json', 'r') as f:
        h2h_stats = json.load(f)
    
    # Try both combinations of team names
    h2h_key = f"{home_team_name}_vs_{away_team_name}"
    if h2h_key not in h2h_stats:
        h2h_key = f"{away_team_name}_vs_{home_team_name}"
    
    if h2h_key in h2h_stats:
        h2h_data = h2h_stats[h2h_key]
        
        # Return H2H stats for both teams
        return {
            'h2h_home_wins': h2h_data[home_team_name]['wins'],
            'h2h_home_draws': h2h_data[home_team_name]['draws'],
            'h2h_home_losses': h2h_data[home_team_name]['losses'],
            'h2h_home_goals': h2h_data[home_team_name]['goals'],
            'h2h_home_clean_sheets': h2h_data[home_team_name]['clean_sheets'],
            'h2h_away_wins': h2h_data[away_team_name]['wins'],
            'h2h_away_draws': h2h_data[away_team_name]['draws'],
            'h2h_away_losses': h2h_data[away_team_name]['losses'],
            'h2h_away_goals': h2h_data[away_team_name]['goals'],
            'h2h_away_clean_sheets': h2h_data[away_team_name]['clean_sheets']
        }
    else:
        # Return zeros if no H2H data is found
        return {
            'h2h_home_wins': 0,
            'h2h_home_draws': 0,
            'h2h_home_losses': 0,
            'h2h_home_goals': 0,
            'h2h_home_clean_sheets': 0,
            'h2h_away_wins': 0,
            'h2h_away_draws': 0,
            'h2h_away_losses': 0,
            'h2h_away_goals': 0,
            'h2h_away_clean_sheets': 0
        }

def prepare_training_data(historical_fixtures, future_fixtures=None, is_training=True):
    """
    Prepare data by combining fixtures with historical form
    
    Args:
        historical_fixtures: Dictionary containing past match data
        future_fixtures: Optional list of future fixtures (for test data)
        is_training: Boolean indicating if this is for training (True) or prediction (False)
    """
    data = []
    
    # Convert historical fixtures to a format easier to search
    historical_data = historical_fixtures['fixtures']
    historical_data.sort(key=lambda x: x['kickoff']['timestamp'])
    
    # Determine which fixtures to process
    fixtures_to_process = historical_data if is_training else future_fixtures
    
    for fixture in fixtures_to_process:
        # Get team IDs and timestamp (handle different formats for historical vs future)
        if is_training:
            home_team_id = fixture['home_team']['id']
            away_team_id = fixture['away_team']['id']
            timestamp = fixture['kickoff']['timestamp']
        else:
            home_team_id = fixture['home_team']['id']
            away_team_id = fixture['away_team']['id']
            timestamp = fixture['kickoff']['timestamp']
        
        # Get recent matches for both teams before this fixture
        home_form = extract_team_recent_form(historical_data, home_team_id, timestamp)
        away_form = extract_team_recent_form(historical_data, away_team_id, timestamp)
        
        # Get H2H features
        h2h_features = get_h2h_features(fixture['home_team']['name'], fixture['away_team']['name'])
        
        row = {
            # Fixture information
            'fixture_id': fixture['fixture_id'],
            'home_team': fixture['home_team']['name'],
            'away_team': fixture['away_team']['name'],
            
            # Home team features
            'home_matches_played': home_form['matches_played'],
            'home_goals_scored': home_form['goals_scored'],
            'home_goals_conceded': home_form['goals_conceded'],
            'home_avg_possession': home_form['avg_possession'],
            'home_avg_shots_on_target': home_form['avg_shots_on_target'],
            'home_avg_pass_accuracy': home_form['avg_pass_accuracy'],
            'home_avg_big_chances': home_form['avg_big_chances'],
            'home_avg_tackles': home_form['avg_tackles'],
            'home_avg_saves': home_form['avg_saves'],
            'home_momentum': home_form['momentum'],
            
            # Away team features
            'away_matches_played': away_form['matches_played'],
            'away_goals_scored': away_form['goals_scored'],
            'away_goals_conceded': away_form['goals_conceded'],
            'away_avg_possession': away_form['avg_possession'],
            'away_avg_shots_on_target': away_form['avg_shots_on_target'],
            'away_avg_pass_accuracy': away_form['avg_pass_accuracy'],
            'away_avg_big_chances': away_form['avg_big_chances'],
            'away_avg_tackles': away_form['avg_tackles'],
            'away_avg_saves': away_form['avg_saves'],
            'away_momentum': away_form['momentum'],
            
            # H2H features
            **h2h_features
        }
        
        # Add actual results for training data
        if is_training:
            row.update({
                'home_goals': fixture['home_team']['score'],
                'away_goals': fixture['away_team']['score']
            })
            
        data.append(row)
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare training or test data')
    parser.add_argument('--mode', choices=['train', 'test'], default='train',
                      help='Whether to prepare training or test data')
    args = parser.parse_args()
    
    # Load historical fixtures
    with open('premier_league_fixtures.json', 'r') as f:
        historical_fixtures = json.load(f)
    
    if args.mode == 'train':
        # Prepare training data
        df = prepare_training_data(historical_fixtures, is_training=True)
        output_file = 'data/training_data.csv'
    else:
        # Load future fixtures and prepare test data
        with open('future_fixtures.json', 'r') as f:
            future_fixtures = json.load(f)
        df = prepare_training_data(historical_fixtures, future_fixtures, is_training=False)
        output_file = 'data/test_data.csv'
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Data prepared and saved to {output_file}")