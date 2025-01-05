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
from team_processing import add_points_for_team, add_team_stats, calculate_match_importance, calculate_form, get_league_positions, get_stats_coverage, get_team_points, initialize_database, get_previous_matches


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
        
        # Connect to database and initialize
        conn = sqlite3.connect(db_path)
        print("Successfully connected to database")
        print("Initializing database...")
        initialize_database(conn)

        # Get completed matches
        print("\nFetching completed matches...")
        matches_query = DEBUG_ENDED_MATCHES_QUERY if debug_mode else ENDED_MATCHES_QUERY
        matches_df = pd.read_sql_query(matches_query, conn)
        print(f"Found {len(matches_df)} completed matches")
        
        # Debug: Check team_stats table
        if debug_mode:
            print("\nChecking team_stats table...")
            stats_check = pd.read_sql_query(STATS_CHECK_QUERY, conn)
            print("Team stats summary:")
            print(stats_check)
        
        basic_data = []
        advanced_data = []
        total_matches = len(matches_df)
        
        for idx, match in matches_df.iterrows():
            try:
                # Process match stats
                add_team_stats(conn, match)
                average_home_stats, average_away_stats = calculate_form(conn, match)
                competition_id = match['competition_id']
                
                # Calculate match importance and update points
                match_importance = calculate_match_importance(conn, match)
                add_points_for_team(conn, match)

                # Create basic row data
                basic_row = {
                    'fixture_id': match['fixture_id'],
                    'start_time': match['start_time'],
                    'home_team': match['home_team'],
                    'away_team': match['away_team'],
                    'competition_id': competition_id,
                    'match_importance': match_importance,
                    'average_home_goals_scored': average_home_stats['average_goals_scored'],
                    'average_home_goals_conceded': average_home_stats['average_goals_conceded'],
                    'average_home_win_rate': average_home_stats['average_win_rate'],
                    'average_home_clean_sheets': average_home_stats['average_clean_sheets'],
                    'average_away_goals_scored': average_away_stats['average_goals_scored'],
                    'average_away_goals_conceded': average_away_stats['average_goals_conceded'],
                    'average_away_win_rate': average_away_stats['average_win_rate'],
                    'average_away_clean_sheets': average_away_stats['average_clean_sheets'],
                }
                if (average_home_stats.get('has_advanced_stats') == 0 and average_away_stats.get('has_advanced_stats') == 0):
                    basic_data.append(basic_row)
                
                # Process advanced stats if available
                if average_home_stats.get('has_advanced_stats') == 1 and average_away_stats.get('has_advanced_stats') == 1:
                    advanced_row = basic_row.copy()
                    advanced_row.update({
                        'home_pass_effectiveness': average_home_stats['pass_effectiveness'],
                        'home_shot_accuracy': average_home_stats['shot_accuracy'],
                        'home_conversion_rate': average_home_stats['conversion_rate'],
                        'home_defensive_success': average_home_stats['defensive_success'],
                        'away_pass_effectiveness': average_away_stats['pass_effectiveness'],
                        'away_shot_accuracy': average_away_stats['shot_accuracy'],
                        'away_conversion_rate': average_away_stats['conversion_rate'],
                        'away_defensive_success': average_away_stats['defensive_success'],
                        'home_goals': match['home_goals'],
                        'away_goals': match['away_goals'],
                    })
                    advanced_data.append(advanced_row)

                conn.commit()
                print(f"Matches processed: {idx + 1} of {total_matches}")

            except Exception as e:
                print(f"\nError processing match {match['fixture_id']}: {str(e)}")
                if debug_mode:
                    raise
                continue
        
        # Save datasets
        basic_df = pd.DataFrame(basic_data)
        advanced_df = pd.DataFrame(advanced_data)
        
        # Save to files
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