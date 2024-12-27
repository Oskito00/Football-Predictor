import sqlite3
import os
from datetime import datetime
import pandas as pd

# Local imports
from constants import (
    PREVIOUS_MATCHES_QUERY,
    ENDED_MATCHES_QUERY,
    STATS_CHECK_QUERY,
    DEBUG_ENDED_MATCHES_QUERY
)
from match_helpers import get_previous_matches
from team_stats import calculate_team_points, calculate_team_stats

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
                
                #use current season for points calculation
                current_season = match['season_id']

                home_points = calculate_team_points(conn, match['home_team'], match['start_time'], current_season)
                away_points = calculate_team_points(conn, match['away_team'], match['start_time'], current_season)
        
                # Get previous matches for both teams
                home_prev = get_previous_matches(conn, match['home_team'], match['start_time'], 5)
                away_prev = get_previous_matches(conn, match['away_team'], match['start_time'], 5)
                
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

if __name__ == "__main__":
    try:
        output_dir = 'sportradar/data/processed_data'
        debug_mode = True  # Set to False for full processing
        basic_df, advanced_df = create_training_data('football_data.db', output_dir, debug_mode)
    except Exception as e:
        print(f"\nScript failed: {str(e)}") 