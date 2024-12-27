from datetime import datetime
import sqlite3
import pandas as pd
import numpy as np

from match_helpers import get_previous_matches

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

def calculate_team_points(conn, team_name, before_date, season):
    """Caclulates the teams points for a given season"""
    #Before date should be date and time of match in question
    all_prev_matches = get_previous_matches(conn, team_name, before_date, 50)
    team_points = 0
    
    if len(all_prev_matches) == 0:
        return 0
    
    for _, match in all_prev_matches.iterrows():
        if match['season_id'] == season: #Current season is the season of the match in question
            print("Adding points for match in:", match['competition_name'])
            if match['team_position'] == 'home':
                team_points += 3 if match['home_score'] > match['away_score'] else 1 if match['home_score'] == match['away_score'] else 0
            else:
                team_points += 3 if match['away_score'] > match['home_score'] else 1 if match['away_score'] == match['home_score'] else 0
    
    return team_points