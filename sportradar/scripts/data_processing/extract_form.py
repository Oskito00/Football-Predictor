import sqlite3
import pandas as pd
import numpy as np
import os
from pathlib import Path
from datetime import datetime
import time

derbies = {
    'Premier League': [
        ('Arsenal', 'Tottenham Hotspur'),  # North London Derby
        ('Liverpool', 'Everton'),  # Merseyside Derby
        ('Manchester United', 'Manchester City'),  # Manchester Derby
        ('Chelsea', 'Tottenham Hotspur'),  # London Derby
        ('Arsenal', 'Chelsea'),  # London Derby
    ],
    'LaLiga': [
        ('Real Madrid', 'Barcelona'),  # El Clásico
        ('Atletico Madrid', 'Real Madrid'),  # Madrid Derby
        ('Sevilla', 'Real Betis'),  # Seville Derby
        ('Athletic Bilbao', 'Real Sociedad'),  # Basque Derby
        ('Valencia', 'Villarreal'),  # Valencian Community Derby
    ],
    'Bundesliga': [
        ('Borussia Dortmund', 'Schalke 04'),  # Revierderby
        ('Bayern Munich', 'Borussia Dortmund'),  # Der Klassiker
        ('Hamburger SV', 'Werder Bremen'),  # Nordderby
        ('Bayern Munich', '1860 Munich'),  # Munich Derby (historical)
    ],
    'Serie A': [
        ('Inter Milan', 'AC Milan'),  # Derby della Madonnina
        ('Roma', 'Lazio'),  # Derby della Capitale
        ('Juventus', 'Torino'),  # Derby della Mole
        ('Napoli', 'Roma'),  # Derby del Sole
        ('Genoa', 'Sampdoria'),  # Derby della Lanterna
    ],
    'Ligue 1': [
        ('Paris Saint-Germain', 'Marseille'),  # Le Classique
        ('Lyon', 'Saint-Etienne'),  # Derby Rhône-Alpes
        ('Nice', 'Monaco'),  # Côte d'Azur Derby
    ],
    'Eredivisie': [
        ('Ajax', 'Feyenoord'),  # De Klassieker
        ('PSV Eindhoven', 'Ajax'),  # Dutch Derby
        ('Feyenoord', 'Sparta Rotterdam'),  # Rotterdam Derby
    ],
    'Swiss Super League': [
        ('FC Basel', 'FC Zürich'),  # Swiss Classic
        ('FC Zürich', 'Grasshopper Club Zürich'),  # Zurich Derby
        ('Young Boys', 'FC Basel'),  # Key Rivalry
    ],
    'Austrian Bundesliga': [
        ('Rapid Wien', 'Austria Wien'),  # Vienna Derby
        ('RB Salzburg', 'Rapid Wien'),  # Top Clash
    ],
    'Danish Superliga': [
        ('FC Copenhagen', 'Brøndby IF'),  # Copenhagen Derby
    ],
    'Norwegian Eliteserien': [
        ('Rosenborg', 'Molde'),  # Norwegian Classic
    ],
    'Swedish Allsvenskan': [
        ('AIK', 'Djurgården'),  # Stockholm Derby
        ('Malmö FF', 'Helsingborg'),  # Skåne Derby
    ],
    'UEFA Champions League': [],
    'UEFA Europa League': [],
    'UEFA Conference League': [],
    'UEFA Super Cup': [],
    'FIFA Club World Cup': [],
    'FA Cup': [],
    'EFL Cup': [],
    'Community Shield': [],
    'Copa del Rey': [],
    'Supercopa': [],
    'Coppa Italia': [],
    'Supercoppa Italiana': [],
    'Coupe de France': [],
}

def get_previous_matches(conn, team_name, before_date, limit=5):
    """Get previous matches for a team before a specific date"""
    print(f"\nGetting previous matches for {team_name} before {before_date}")
    
    query = """
    WITH TeamMatches AS (
        SELECT 
            m.match_id,
            m.start_time,
            m.home_team_name,
            m.away_team_name,
            m.home_score,
            m.away_score,
            CASE 
                WHEN m.home_team_name = ? THEN 'home'
                ELSE 'away'
            END as team_position
        FROM matches m
        WHERE (m.home_team_name = ? OR m.away_team_name = ?)
            AND m.start_time < ?
            AND m.match_status = 'ended'
        ORDER BY m.start_time DESC
        LIMIT ?
    )
    SELECT 
        tm.*,
        ts.ball_possession,
        ts.passes_successful,
        ts.passes_total,
        ts.shots_total,
        ts.shots_on_target,
        ts.chances_created,
        ts.tackles_successful,
        ts.tackles_total,
        ts.shots_saved
    FROM TeamMatches tm
    LEFT JOIN team_stats ts ON tm.match_id = ts.match_id 
        AND ((tm.team_position = 'home' AND ts.qualifier = 'home')
         OR (tm.team_position = 'away' AND ts.qualifier = 'away'))
    """
    
    params = [team_name, team_name, team_name, before_date, limit]
    result = pd.read_sql_query(query, conn, params=params)
    
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
    
    # Advanced metrics (only if available)
    if not matches_df['passes_successful'].isna().all() and not matches_df['passes_total'].isna().all():
        metrics['pass_effectiveness'] = safe_ratio(
            matches_df['passes_successful'].sum(),
            matches_df['passes_total'].sum()
        )
        
    if not matches_df['shots_on_target'].isna().all() and not matches_df['shots_total'].isna().all():
        print("has shots")
        metrics['shot_accuracy'] = safe_ratio(
            matches_df['shots_on_target'].sum(),
            matches_df['shots_total'].sum()
        )
        
    if not matches_df['chances_created'].isna().all() and not matches_df['shots_on_target'].isna().all():
        print("has chances")
        metrics['conversion_rate'] = safe_ratio(
            matches_df['shots_on_target'].sum(),
            matches_df['chances_created'].sum()
        )
        
    if not matches_df['tackles_successful'].isna().all() and not matches_df['tackles_total'].isna().all():
        print("has tackles")
        metrics['defensive_success'] = safe_ratio(
            matches_df['tackles_successful'].sum(),
            matches_df['tackles_total'].sum()
        )
    
    if 'pass_effectiveness' in metrics or 'shot_accuracy' in metrics or 'conversion_rate' in metrics or 'defensive_success' in metrics:
        metrics['has_advanced_stats'] = True
    else:
        metrics['has_advanced_stats'] = False

    # Set metrics to NaN if they don't exist
    if not metrics.get('pass_effectiveness'):
        metrics['pass_effectiveness'] = float('nan')
    if not metrics.get('shot_accuracy'): 
        metrics['shot_accuracy'] = float('nan')
    if not metrics.get('conversion_rate'):
        metrics['conversion_rate'] = float('nan')
    if not metrics.get('defensive_success'):
        metrics['defensive_success'] = float('nan')

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
            matches_query = """
            WITH numbered_matches AS (
                SELECT 
                    match_id as fixture_id,
                    start_time,
                    home_team_name as home_team,
                    away_team_name as away_team,
                    home_score as home_goals,
                    away_score as away_goals,
                    ROW_NUMBER() OVER (ORDER BY start_time) as row_num
                FROM matches
                WHERE match_status = 'ended'
            )
            SELECT 
                fixture_id,
                start_time,
                home_team,
                away_team,
                home_goals,
                away_goals
            FROM numbered_matches
            WHERE row_num % 50 = 0
            ORDER BY start_time
            """
        else:
            matches_query = """
            SELECT 
                match_id as fixture_id,
                start_time,
                home_team_name as home_team,
                away_team_name as away_team,
                home_score as home_goals,
                away_score as away_goals
            FROM matches
            WHERE match_status = 'ended'
            ORDER BY start_time
            """
        
        matches_df = pd.read_sql_query(matches_query, conn)
        print(f"Found {len(matches_df)} completed matches")
        
        # Debug: Check team_stats table
        if debug_mode:
            print("\nChecking team_stats table...")
            stats_check = pd.read_sql_query("""
                SELECT COUNT(*) as count, 
                       COUNT(DISTINCT match_id) as unique_matches,
                       COUNT(DISTINCT team_name) as unique_teams
                FROM team_stats
            """, conn)
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
        basic_df, advanced_df = create_training_data('football_data.db', output_dir, debug_mode=debug_mode)
        
    except Exception as e:
        print(f"\nScript failed: {str(e)}")