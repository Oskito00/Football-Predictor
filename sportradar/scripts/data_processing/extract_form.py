import sqlite3
import pandas as pd
import numpy as np
import os
from pathlib import Path
from datetime import datetime
import time
from constants import PREVIOUS_MATCHES_QUERY, ENDED_MATCHES_QUERY, STATS_CHECK_QUERY, DEBUG_ENDED_MATCHES_QUERY

def get_previous_matches(conn, team_name, before_date, limit=5):
    """Get previous matches for a team before a specific date"""
    print(f"\nGetting previous matches for {team_name} before {before_date}")
    
    params = [team_name, team_name, team_name, before_date, limit]
    result = pd.read_sql_query(PREVIOUS_MATCHES_QUERY, conn, params=params)
    
    # Debug output
    print(f"Retrieved {len(result)} matches with stats")
    if len(result) > 0:
        print("\nScore columns:")
        for idx, row in result.iterrows():
            if row['team_position'] == 'home':
                print(f"Match {idx+1}: {row['home_team_name']} {row['home_score']} - {row['away_score']} {row['away_team_name']}")
            else:
                print(f"Match {idx+1}: {row['home_team_name']} {row['home_score']} - {row['away_score']} {row['away_team_name']}")
        
    
    return result

def calculate_team_stats(matches_df):
    """Calculate team performance metrics from previous matches"""
    print("\nCalculating team statistics...")
    print(f"Working with {len(matches_df)} matches")
    
    if len(matches_df) == 0:
        print("No matches found, returning default values")
        return get_default_metrics()
    
    metrics = calculate_metrics(matches_df) 
    
    print("\nCalculated metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    
    return metrics

def calculate_metrics(matches_df):
    print(matches_df)
    """Calculate performance metrics from match data"""
    # Convert None to NaN for numeric operations
    matches_df = matches_df.replace({None: np.nan})
    
    # Helper function to safely calculate ratios
    def safe_ratio(numerator, denominator, default=0):
        if denominator == 0 or pd.isna(denominator):
            return default
        return numerator / denominator
    
    # Basic metrics (always available)

    print("\nTeam positions:")
    print(matches_df['team_position'].tolist())
    print(matches_df)

    matches_played = 0
    goals_scored = 0
    goals_conceded = 0
    win_rate = 0
    clean_sheets = 0

    for i in range(len(matches_df)):
        team_position = matches_df['team_position'].iloc[i]
        matches_played += 1
        if team_position == 'home':
            goals_scored += matches_df['home_score'].iloc[i]
            goals_conceded += matches_df['away_score'].iloc[i]
            win_rate += 1 if matches_df['home_score'].iloc[i] > matches_df['away_score'].iloc[i] else 0
            clean_sheets += 1 if matches_df['away_score'].iloc[i] == 0 else 0
        else:
            goals_scored += matches_df['away_score'].iloc[i]
            goals_conceded += matches_df['home_score'].iloc[i]
            win_rate += 1 if matches_df['away_score'].iloc[i] > matches_df['home_score'].iloc[i] else 0
            clean_sheets += 1 if matches_df['home_score'].iloc[i] == 0 else 0


    metrics = {
        'matches_played': matches_played,
        'average_goals_scored': goals_scored/matches_played,
        'average_goals_conceded': goals_conceded/matches_played,
        'average_win_rate': win_rate/matches_played,
        'average_clean_sheets': clean_sheets/matches_played
    }
    
    #Get the average advanced stats data with the sum of the stats divided by the number of matches that had respectice advanced stats
    if not matches_df['passes_successful'].isna().all() and not matches_df['passes_total'].isna().all():
        print("has passes, successful and total:")
        print("Passes successful:")
        print(matches_df['passes_successful'])
        print("Passes total:")
        print(matches_df['passes_total'])
        metrics['pass_effectiveness'] = safe_ratio(
            matches_df['passes_successful'].sum(),
            matches_df['passes_total'].sum()
        )
        
    if not matches_df['shots_on_target'].isna().all() and not matches_df['shots_total'].isna().all():
        print("has shots")
        print("Shots on target:")
        print(matches_df['shots_on_target'])
        print("Shots total:")
        print(matches_df['shots_total'])
        metrics['shot_accuracy'] = safe_ratio(
            matches_df['shots_on_target'].sum(),
            matches_df['shots_total'].sum()
        )
        
    if not matches_df['chances_created'].isna().all() and not matches_df['shots_on_target'].isna().all():
        print("has chances")
        print("Chances created:")
        print(matches_df['chances_created'])
        print("Shots on target:")
        print(matches_df['shots_on_target'])
        metrics['conversion_rate'] = safe_ratio(
            matches_df['shots_on_target'].sum(),
            matches_df['chances_created'].sum()
        )
        
    if not matches_df['tackles_successful'].isna().all() and not matches_df['tackles_total'].isna().all():
        print("has tackles")
        print("Tackles successful:")
        print(matches_df['tackles_successful'])
        print("Tackles total:")
        print(matches_df['tackles_total'])
        metrics['defensive_success'] = safe_ratio(
            matches_df['tackles_successful'].sum(),
            matches_df['tackles_total'].sum()
        )
    
    #Only set has_advanced_stats to true if all advanced stats are present
    if 'pass_effectiveness' in metrics and 'shot_accuracy' in metrics and 'conversion_rate' in metrics and 'defensive_success' in metrics:
        metrics['has_advanced_stats'] = True
    else:
        metrics['has_advanced_stats'] = False

    return metrics

def get_default_metrics():
    """Return default metrics when no matches are found"""
    return {
        'matches_played': 0,
        'average_goals_scored': 0,
        'average_goals_conceded': 0,
        'average_win_rate': 0,
        'average_clean_sheets': 0,
        'pass_effectiveness': 0,
        'shot_accuracy': 0,
        'conversion_rate': 0,
        'defensive_success': 0,
        'has_advanced_stats': False
    }

def create_training_data(db_path, output_dir, debug_mode=False):
    """Create both basic and advanced training datasets from match database"""
    print(f"\n=== Starting training data creation at {datetime.now()} ===")
    try:
        # Ensure database exists
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database not found at: {db_path}")
        print(f"Using database: {db_path}")
        print(f"Debug mode: {'ON' if debug_mode else 'OFF'}")
            
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Connect to database
        conn = sqlite3.connect(db_path)
        print("Successfully connected to database")
        
        # Get completed matches
        print("\nFetching completed matches...")
        if debug_mode:
            matches_query = DEBUG_ENDED_MATCHES_QUERY
        else:
            matches_query = ENDED_MATCHES_QUERY
                
        matches_df = pd.read_sql_query(matches_query, conn)
        print(f"Found {len(matches_df)} completed matches")
        
        # Debug: Check team_stats table
        if debug_mode:
            print("\nChecking team_stats table...")
            stats_check = pd.read_sql_query(STATS_CHECK_QUERY, conn)
            print("Team stats summary:")
            print(stats_check)
        
        # Rest of the function remains the same...
        basic_data = []
        advanced_data = []
        total_matches = len(matches_df)
        
        for idx, match in matches_df.iterrows():
            try:
                if debug_mode or idx % 100 == 0:
                    print(f"\nProcessing match {idx + 1} of {total_matches} ({(idx + 1)/total_matches*100:.1f}%)")
                    print(f"Match: {match['home_team']} vs {match['away_team']}")
                
                
                # Get previous matches for both teams
                home_prev = get_previous_matches(conn, match['home_team'], match['start_time'])
                away_prev = get_previous_matches(conn, match['away_team'], match['start_time'])
                
                # Calculate form metrics
                home_metrics = calculate_team_stats(home_prev)
                away_metrics = calculate_team_stats(away_prev)
                
                # Now create basic_row with all context
                basic_row = {
                    'fixture_id': match['fixture_id'],
                    'start_time': match['start_time'],
                    'home_team': match['home_team'],
                    'away_team': match['away_team'],
                    'competition_name': match['competition_name'],
                    'home_matches_played': home_metrics['matches_played'],
                    'average_home_goals_scored': home_metrics['average_goals_scored'],
                    'average_home_goals_conceded': home_metrics['average_goals_conceded'],
                    'average_home_win_rate': home_metrics['average_win_rate'],
                    'average_home_clean_sheets': home_metrics['average_clean_sheets'],
                    'away_matches_played': away_metrics['matches_played'],
                    'average_away_goals_scored': away_metrics['average_goals_scored'],
                    'average_away_goals_conceded': away_metrics['average_goals_conceded'],
                    'average_away_win_rate': away_metrics['average_win_rate'],
                    'average_away_clean_sheets': away_metrics['average_clean_sheets'],
                    'home_goals': match['home_goals'],
                    'away_goals': match['away_goals'],
                }
                
                basic_data.append(basic_row)
                
                # Advanced stats processing
                if home_metrics['has_advanced_stats'] and away_metrics['has_advanced_stats']:
                    advanced_row = basic_row.copy()
                    advanced_row.update({
                        'home_pass_effectiveness': home_metrics['pass_effectiveness'],
                        'home_shot_accuracy': home_metrics['shot_accuracy'],
                        'home_conversion_rate': home_metrics['conversion_rate'],
                        'home_defensive_success': home_metrics['defensive_success'],
                        'away_pass_effectiveness': away_metrics['pass_effectiveness'],
                        'away_shot_accuracy': away_metrics['shot_accuracy'],
                        'away_conversion_rate': away_metrics['conversion_rate'],
                        'away_defensive_success': away_metrics['defensive_success']
                    })
                    advanced_data.append(advanced_row)
                    
                    if debug_mode:
                        print("\nAdvanced metrics available for this match")
                
            except Exception as e:
                print(f"\nError processing match {match['fixture_id']}: {str(e)}")
                if debug_mode:
                    raise  # In debug mode, raise the exception for detailed traceback
                continue
        
        # Save datasets
        basic_df = pd.DataFrame(basic_data)
        advanced_df = pd.DataFrame(advanced_data)
        
        # Add debug suffix to filenames in debug mode
        suffix = '_debug' if debug_mode else ''
        basic_output = os.path.join(output_dir, f'training_data_basic{suffix}.csv')
        advanced_output = os.path.join(output_dir, f'training_data_advanced{suffix}.csv')
        
        basic_df.to_csv(basic_output, index=False)
        advanced_df.to_csv(advanced_output, index=False)
        
        print(f"\nSaved basic dataset with {len(basic_df)} matches to {basic_output}")
        print(f"Saved advanced dataset with {len(advanced_df)} matches to {advanced_output}")
        
        return basic_df, advanced_df
        
    except Exception as e:
        print(f"\nFatal error: {str(e)}")
        raise
    finally:
        if 'conn' in locals():
            conn.close()
            print("\nDatabase connection closed")


# def print_matches_db():
#     conn = sqlite3.connect('football_data.db')
#     matches_df = pd.read_sql_query(MATCHES_QUERY, conn)
#     print("matches_df: ")
#     print(matches_df)

if __name__ == "__main__":
    try:
        output_dir = 'sportradar/data/processed_data'
        debug_mode = True  # Set to False for full processing
        basic_df, advanced_df = create_training_data('football_data.db', output_dir, debug_mode=True)
        # print_matches_db()
    except Exception as e:
        print(f"\nScript failed: {str(e)}")