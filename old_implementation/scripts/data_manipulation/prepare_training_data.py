import json
import math
import pandas as pd
import numpy as np
import os

def load_future_fixtures(json_file):
    """Load future fixtures from a JSON file."""
    with open(json_file, 'r') as f:
        return json.load(f)

def calculate_team_stats(match_stats):
    """Extract key statistics from a team's match data"""

    # === Possession and Passing Performance ===
    possession_percentage = match_stats.get('possession_percentage', 0)

    total_passes = match_stats.get('total_pass', 0)
    accurate_passes = match_stats.get('accurate_pass', 0)
    pass_accuracy = (accurate_passes / total_passes * 100) if total_passes else 0

    total_final_third_passes = match_stats.get('total_final_third_passes', 0)
    successful_final_third_passes = match_stats.get('successful_final_third_passes', 0)
    final_third_pass_accuracy = (successful_final_third_passes / total_final_third_passes * 100) if total_final_third_passes else 0

    total_long_balls = match_stats.get('total_long_balls', 0)
    accurate_long_balls = match_stats.get('accurate_long_balls', 0)
    long_ball_accuracy = (accurate_long_balls / total_long_balls * 100) if total_long_balls else 0

    total_crosses = match_stats.get('total_cross', 0)
    accurate_crosses = match_stats.get('accurate_cross', 0)
    cross_accuracy = (accurate_crosses / total_crosses * 100) if total_crosses else 0

    # === Combine Passing Metrics into 'pass_effectiveness' ===
    # Weighting final third pass accuracy higher because it's more important
    pass_accuracy_weight = 1
    final_third_pass_accuracy_weight = 2  # Higher weight
    long_ball_accuracy_weight = 1
    cross_accuracy_weight = 1

    total_pass_weights = (pass_accuracy_weight +
                          final_third_pass_accuracy_weight +
                          long_ball_accuracy_weight +
                          cross_accuracy_weight)

    pass_effectiveness = (
        (pass_accuracy * pass_accuracy_weight) +
        (final_third_pass_accuracy * final_third_pass_accuracy_weight) +
        (long_ball_accuracy * long_ball_accuracy_weight) +
        (cross_accuracy * cross_accuracy_weight)
    ) / total_pass_weights

    # === Attacking Performance ===
    goals_scored = match_stats.get('goals', 0)
    total_scoring_att = match_stats.get('total_scoring_att', 0)
    shots_on_target = match_stats.get('ontarget_scoring_att', 0)
    shot_accuracy = (shots_on_target / total_scoring_att * 100) if total_scoring_att else 0
    conversion_rate = (goals_scored / total_scoring_att * 100) if total_scoring_att else 0

    big_chance_scored = match_stats.get('big_chance_scored', 0)
    big_chance_missed = match_stats.get('big_chance_missed', 0)
    total_big_chances = big_chance_scored + big_chance_missed
    big_chance_conversion = (big_chance_scored / total_big_chances * 100) if total_big_chances else 0

    # === Defensive Performance ===
    goals_conceded = match_stats.get('goals_conceded', 0)
    goals_conceded_inside_box = match_stats.get('goals_conceded_ibox', 0)
    goals_conceded_outside_box = goals_conceded - goals_conceded_inside_box

    total_tackles = match_stats.get('total_tackle', 0)
    won_tackles = match_stats.get('won_tackle', 0)
    tackle_success_rate = (won_tackles / total_tackles * 100) if total_tackles else 0

    duel_won = match_stats.get('duel_won', 0)
    duel_lost = match_stats.get('duel_lost', 0)
    total_duels = duel_won + duel_lost
    duel_success_rate = (duel_won / total_duels * 100) if total_duels else 0

    aerial_won = match_stats.get('aerial_won', 0)
    aerial_lost = match_stats.get('aerial_lost', 0)
    total_aerials = aerial_won + aerial_lost
    aerial_success_rate = (aerial_won / total_aerials * 100) if total_aerials else 0

    # === Combine Defensive Metrics into 'defensive_success_rate' ===
    defensive_success_rate = (
        tackle_success_rate +
        duel_success_rate +
        aerial_success_rate
    ) / 3  # Equal weighting

    # === Keeper Performance ===
    saves_made = match_stats.get('saves', 0)
    shots_on_target_faced = saves_made + goals_conceded
    save_percentage = (saves_made / shots_on_target_faced * 100) if shots_on_target_faced else 0

    # === Assemble Metrics ===
    team_stats = {
        # Possession and Passing Performance
        'possession_percentage': possession_percentage,
        'pass_effectiveness': round(pass_effectiveness, 2),

        # Attacking Performance
        'shot_accuracy': round(shot_accuracy, 2),
        'conversion_rate': round(conversion_rate, 2),
        'big_chance_conversion_rate': round(big_chance_conversion, 2),

        # Defensive Performance
        'defensive_success_rate': round(defensive_success_rate, 2),

        # Keeper Performance
        'save_percentage': round(save_percentage, 2),
    }

    return team_stats

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
    print(f"\nDebug: Processing recent form for team_id {team_id}")
    print(f"Debug: Looking for matches before timestamp {before_timestamp}")
    
    previous_matches = [
        f for f in fixtures 
        if f['kickoff']['timestamp'] < before_timestamp and
        (f['home_team']['id'] == team_id or f['away_team']['id'] == team_id)
    ]
    
    print(f"Debug: Found {len(previous_matches)} previous matches for team")
    
    previous_matches.sort(key=lambda x: x['kickoff']['timestamp'], reverse=True)
    recent_matches = previous_matches[:max_matches]
    print(f"Debug: Using {len(recent_matches)} most recent matches")
    
    # Calculate momentum first
    momentum = calculate_momentum(recent_matches, team_id)
    
    # Default values when no matches found
    if not recent_matches:
        print("Debug: No recent matches found, returning default values")
        return {
            'matches_played': 0,
            'goals_scored': 0,
            'goals_conceded': 0,
            'possession_percentage': 0,
            'pass_effectiveness': 0,
            'shot_accuracy': 0,
            'conversion_rate': 0,
            'big_chance_conversion_rate': 0,
            'defensive_success_rate': 0,
            'save_percentage': 0,
            'momentum': momentum
        }
    
    total_goals_scored = 0
    total_goals_conceded = 0
    total_possession = 0
    total_pass_effectiveness = 0
    total_shot_accuracy = 0
    total_conversion_rate = 0
    total_big_chance_conversion = 0
    total_defensive_success = 0
    total_save_percentage = 0
    matches_with_stats = 0
    
    for match in recent_matches:
        print(f"\nDebug: Processing match stats")
        stats = match.get('stats', {})
        if not stats:
            print("Debug: No stats found for match, skipping")
            continue
            
        match_stats = stats.get('match_stats', {})
        if not match_stats:
            print("Debug: No match_stats found, skipping")
            continue
            
        is_home = match['home_team']['id'] == team_id
        print(f"Debug: Team playing as {'home' if is_home else 'away'}")
        
        team_stats = match_stats.get('home' if is_home else 'away')
        if not team_stats:
            print(f"Debug: No {'home' if is_home else 'away'} stats found, skipping")
            continue
            
        # Basic match results
        team_score = match['home_team']['score'] if is_home else match['away_team']['score']
        opponent_score = match['away_team']['score'] if is_home else match['home_team']['score']
        
        total_goals_scored += team_score
        total_goals_conceded += opponent_score
        
        # Calculate and aggregate advanced stats
        calculated_stats = calculate_team_stats(team_stats)
        print(f"Debug: Calculated stats: {calculated_stats}")
        
        total_possession += calculated_stats['possession_percentage']
        total_pass_effectiveness += calculated_stats['pass_effectiveness']
        total_shot_accuracy += calculated_stats['shot_accuracy']
        total_conversion_rate += calculated_stats['conversion_rate']
        total_big_chance_conversion += calculated_stats['big_chance_conversion_rate']
        total_defensive_success += calculated_stats['defensive_success_rate']
        total_save_percentage += calculated_stats['save_percentage']
        matches_with_stats += 1
    
    matches_played = len(recent_matches)
    print(f"Debug: Processed {matches_played} matches successfully")
    
    # Calculate averages using matches_with_stats instead of matches_played
    divisor = max(1, matches_with_stats)  # Avoid division by zero
    return {
        'matches_played': matches_played,
        'goals_scored': total_goals_scored,
        'goals_conceded': total_goals_conceded,
        'possession_percentage': total_possession / divisor,
        'pass_effectiveness': total_pass_effectiveness / divisor,
        'shot_accuracy': total_shot_accuracy / divisor,
        'conversion_rate': total_conversion_rate / divisor,
        'big_chance_conversion_rate': total_big_chance_conversion / divisor,
        'defensive_success_rate': total_defensive_success / divisor,
        'save_percentage': total_save_percentage / divisor,
        'momentum': momentum
    }

def get_h2h_features(home_team_name, away_team_name, year):
    """Retrieve normalized head-to-head statistics for a given fixture"""
    
    # Load H2H stats for the specific year
    try:
        with open(f'data/{year}/all_h2h_stats.json', 'r') as f:
            h2h_stats = json.load(f)
    except FileNotFoundError:
        print(f"H2H stats file not found for year {year}")
        return {
            'h2h_home_points': 0,
            'h2h_home_goals': 0,
            'h2h_home_clean_sheets': 0,
            'h2h_away_points': 0,
            'h2h_away_goals': 0,
            'h2h_away_clean_sheets': 0
        }
    
    # Try both combinations of team names
    h2h_key = f"{home_team_name}_vs_{away_team_name}"
    reverse_key = f"{away_team_name}_vs_{home_team_name}"
    
    if h2h_key in h2h_stats:
        h2h_data = h2h_stats[h2h_key]
        home_team_data = h2h_data[home_team_name]
        away_team_data = h2h_data[away_team_name]
    elif reverse_key in h2h_stats:
        h2h_data = h2h_stats[reverse_key]
        home_team_data = h2h_data[home_team_name]
        away_team_data = h2h_data[away_team_name]
    else:
        return {
            'h2h_home_points': 0,
            'h2h_home_goals': 0,
            'h2h_home_clean_sheets': 0,
            'h2h_away_points': 0,
            'h2h_away_goals': 0,
            'h2h_away_clean_sheets': 0
        }
    
    # Calculate total matches for each team
    home_matches = home_team_data['wins'] + home_team_data['draws'] + home_team_data['losses']
    away_matches = away_team_data['wins'] + away_team_data['draws'] + away_team_data['losses']
    
    # Calculate normalized statistics
    # Points: total points / maximum possible points (3 * number of matches)
    home_points_normalized = ((home_team_data['wins'] * 3) + home_team_data['draws']) / (home_matches * 3) if home_matches > 0 else 0
    away_points_normalized = ((away_team_data['wins'] * 3) + away_team_data['draws']) / (away_matches * 3) if away_matches > 0 else 0
    
    # Goals: average goals per game
    home_goals_normalized = home_team_data['goals'] / home_matches if home_matches > 0 else 0
    away_goals_normalized = away_team_data['goals'] / away_matches if away_matches > 0 else 0
    
    # Clean sheets: proportion of games with clean sheets
    home_clean_sheets_normalized = home_team_data['clean_sheets'] / home_matches if home_matches > 0 else 0
    away_clean_sheets_normalized = away_team_data['clean_sheets'] / away_matches if away_matches > 0 else 0
    
    return {
        'h2h_home_points': home_points_normalized,
        'h2h_home_goals': home_goals_normalized,
        'h2h_home_clean_sheets': home_clean_sheets_normalized,
        'h2h_away_points': away_points_normalized,
        'h2h_away_goals': away_goals_normalized,
        'h2h_away_clean_sheets': away_clean_sheets_normalized
    }

def prepare_training_data(year, future_fixtures=None, is_training=True):
    """
    Prepare data by combining fixtures with historical form
    
    Args:
        year: The year to prepare data for
        future_fixtures: Optional list of future fixtures (for test data)
        is_training: Boolean indicating if this is for training (True) or prediction (False)
    """
    print(f"\nDebug: Starting prepare_training_data for year {year}")
    
    # Load historical fixtures for the specific year
    try:
        with open(f'data/{year}/premier_league_results.json', 'r') as f:
            historical_fixtures = json.load(f)
            print(f"Debug: Successfully loaded fixtures file")
            print(f"Debug: Keys in historical_fixtures: {historical_fixtures.keys()}")
    except FileNotFoundError:
        print(f"Debug: Fixtures file not found for year {year}")
        return pd.DataFrame()
    
    if 'fixtures' not in historical_fixtures:
        print(f"Debug: No fixtures data found for year {year}")
        return pd.DataFrame()
    
    data = []
    
    # Convert historical fixtures to a format easier to search
    historical_data = historical_fixtures['fixtures']
    print(f"Debug: Number of fixtures found: {len(historical_data)}")
    historical_data.sort(key=lambda x: x['kickoff']['timestamp'])
    
    # Determine which fixtures to process
    fixtures_to_process = historical_data if is_training else future_fixtures
    print(f"Debug: Number of fixtures to process: {len(fixtures_to_process)}")
    
    for i, fixture in enumerate(fixtures_to_process):
        try:
            print(f"\nDebug: Processing fixture {i+1}/{len(fixtures_to_process)}")
            print(f"Debug: Fixture keys available: {fixture.keys()}")
            print(f"Debug: Home team: {fixture['home_team']['name']} vs Away team: {fixture['away_team']['name']}")
            
            # Get team IDs and timestamp
            home_team_id = fixture['home_team']['id']
            away_team_id = fixture['away_team']['id']
            timestamp = fixture['kickoff']['timestamp']
            
            # Get recent matches for both teams before this fixture
            home_form = extract_team_recent_form(historical_data, home_team_id, timestamp)
            away_form = extract_team_recent_form(historical_data, away_team_id, timestamp)
            
            # Get H2H features for the specific year
            h2h_features = get_h2h_features(fixture['home_team']['name'], 
                                          fixture['away_team']['name'], 
                                          year)
            
            row = {
                # Fixture information
                'fixture_id': fixture['fixture_id'],
                'home_team': fixture['home_team']['name'],
                'away_team': fixture['away_team']['name'],
                
                # Home team features
                'home_matches_played': home_form['matches_played'],
                'home_goals_scored': home_form['goals_scored'],
                'home_goals_conceded': home_form['goals_conceded'],
                'home_possession': home_form['possession_percentage'],
                'home_pass_effectiveness': home_form['pass_effectiveness'],
                'home_shot_accuracy': home_form['shot_accuracy'],
                'home_conversion_rate': home_form['conversion_rate'],
                'home_big_chance_conversion': home_form['big_chance_conversion_rate'],
                'home_defensive_success': home_form['defensive_success_rate'],
                'home_save_percentage': home_form['save_percentage'],
                'home_momentum': home_form['momentum'],
                
                # Away team features
                'away_matches_played': away_form['matches_played'],
                'away_goals_scored': away_form['goals_scored'],
                'away_goals_conceded': away_form['goals_conceded'],
                'away_possession': away_form['possession_percentage'],
                'away_pass_effectiveness': away_form['pass_effectiveness'],
                'away_shot_accuracy': away_form['shot_accuracy'],
                'away_conversion_rate': away_form['conversion_rate'],
                'away_big_chance_conversion': away_form['big_chance_conversion_rate'],
                'away_defensive_success': away_form['defensive_success_rate'],
                'away_save_percentage': away_form['save_percentage'],
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
            
        except Exception as e:
            print(f"Debug: Error processing fixture: {str(e)}")
            print(f"Debug: Fixture data: {fixture}")
            raise
    
    print(f"Debug: Successfully processed {len(data)} fixtures")
    return pd.DataFrame(data)

def process_all_years(start_year=2024, end_year=2024):
    """Process data for multiple years, from start_year to end_year inclusive"""
    for year in range(start_year, end_year + 1):  # Changed from end_year-1 to end_year + 1
        try:
            year_str = str(year)
            print(f"Processing year {year_str}...")
            
            # Create output directory if it doesn't exist
            output_dir = f'LinearRegression/data/{year_str}'
            os.makedirs(output_dir, exist_ok=True)
            
            # Prepare and save training data
            df = prepare_training_data(year_str, is_training=True)
            if df.empty:
                print(f"No data prepared for year {year_str}")
                continue
            
            output_file = f'{output_dir}/training_data.csv'
            df.to_csv(output_file, index=False)
            
            print(f"Data prepared and saved to {output_file}")
            
        except FileNotFoundError as e:
            print(f"No data found for year {year_str}, skipping... ({str(e)})")
        except Exception as e:
            print(f"Error processing year {year_str}: {str(e)}")

def process_future_fixtures():
    # Load future fixtures
    future_fixtures_file = 'data/2024/future_fixtures.json'
    future_fixtures = load_future_fixtures(future_fixtures_file)
    
    # Prepare future fixtures data
    print("Preparing future fixtures data...")
    future_data = prepare_training_data(year=2024, future_fixtures=future_fixtures, is_training=False)
    
    # Save to CSV
    future_data_csv = 'LinearRegression/data/2024/future_fixtures_prepared.csv'
    future_data.to_csv(future_data_csv, index=False)
    print(f"Future fixtures data saved to {future_data_csv}")



if __name__ == "__main__":
    process_all_years()
    # process_future_fixtures()
    