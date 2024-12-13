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
    
    # First, let's check if we can find the team in matches
    check_query = """
    SELECT COUNT(*) as match_count
    FROM matches m
    WHERE (home_team_name = ? OR away_team_name = ?)
        AND start_time < ?
        AND match_status = 'ended'
    """
    match_count = pd.read_sql_query(check_query, conn, params=[team_name, team_name, before_date]).iloc[0]['match_count']
    print(f"Found {match_count} matches for {team_name}")

    # Modified query to handle team_stats join better
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
        print("\nSample stats for first match:")
        for col in result.columns:
            print(f"{col}: {result.iloc[0][col]}")
    
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
    """Calculate performance metrics from match data"""
    # Convert None to NaN for numeric operations
    matches_df = matches_df.replace({None: np.nan})
    
    # Helper function to safely calculate ratios
    def safe_ratio(numerator, denominator, default=0):
        if denominator == 0 or pd.isna(denominator):
            return default
        return numerator / denominator
    
    # Basic metrics (always available)
    metrics = {
        'matches_played': len(matches_df),
        'goals_scored': matches_df['home_score' if matches_df['team_position'].iloc[0] == 'home' else 'away_score'].mean(),
        'goals_conceded': matches_df['away_score' if matches_df['team_position'].iloc[0] == 'home' else 'home_score'].mean(),
        'win_rate': len(matches_df[matches_df['home_score' if matches_df['team_position'].iloc[0] == 'home' else 'away_score'] > 
                                    matches_df['away_score' if matches_df['team_position'].iloc[0] == 'home' else 'home_score']]) / len(matches_df),
        'clean_sheets': len(matches_df[matches_df['away_score' if matches_df['team_position'].iloc[0] == 'home' else 'home_score'] == 0]) / len(matches_df)
    }
    
    # Advanced metrics (only if available)
    if not matches_df['passes_successful'].isna().all() and not matches_df['passes_total'].isna().all():
        metrics['pass_effectiveness'] = safe_ratio(
            matches_df['passes_successful'].sum(),
            matches_df['passes_total'].sum()
        )
        
    if not matches_df['shots_on_target'].isna().all() and not matches_df['shots_total'].isna().all():
        metrics['shot_accuracy'] = safe_ratio(
            matches_df['shots_on_target'].sum(),
            matches_df['shots_total'].sum()
        )
        
    if not matches_df['chances_created'].isna().all():
        metrics['conversion_rate'] = safe_ratio(
            matches_df['shots_on_target'].sum(),
            matches_df['chances_created'].sum()
        )
        
    if not matches_df['tackles_successful'].isna().all() and not matches_df['tackles_total'].isna().all():
        metrics['defensive_success'] = safe_ratio(
            matches_df['tackles_successful'].sum(),
            matches_df['tackles_total'].sum()
        )
    
    # Flag for complete advanced stats
    metrics['has_advanced_stats'] = not (
        matches_df['passes_successful'].isna().all() or
        matches_df['shots_on_target'].isna().all() or
        matches_df['chances_created'].isna().all() or
        matches_df['tackles_successful'].isna().all()
    )
    
    # Replace any remaining NaN with 0
    metrics = {k: 0 if pd.isna(v) else v for k, v in metrics.items()}
    
    return metrics

def get_default_metrics():
    """Return default metrics when no matches are found"""
    return {
        'matches_played': 0,
        'goals_scored': 0,
        'goals_conceded': 0,
        'win_rate': 0,
        'clean_sheets': 0,
        'pass_effectiveness': 0,
        'shot_accuracy': 0,
        'conversion_rate': 0,
        'defensive_success': 0,
        'has_advanced_stats': False
    }

def get_league_standings(conn, competition_name, before_date):
    """Get league standings before a specific date"""
    query = """
    WITH PreviousMatches AS (
        SELECT 
            home_team_name,
            away_team_name,
            home_score,
            away_score,
            start_time
        FROM matches
        WHERE competition_name = ?
            AND start_time < ?
            AND match_status = 'ended'
    ),
    TeamPoints AS (
        SELECT 
            team_name,
            SUM(points) as total_points,
            COUNT(*) as matches_played
        FROM (
            SELECT 
                home_team_name as team_name,
                CASE 
                    WHEN home_score > away_score THEN 3
                    WHEN home_score = away_score THEN 1
                    ELSE 0
                END as points
            FROM PreviousMatches
            UNION ALL
            SELECT 
                away_team_name as team_name,
                CASE 
                    WHEN away_score > home_score THEN 3
                    WHEN home_score = away_score THEN 1
                    ELSE 0
                END as points
            FROM PreviousMatches
        )
        GROUP BY team_name
    )
    SELECT 
        team_name,
        total_points,
        matches_played,
        ROW_NUMBER() OVER (ORDER BY total_points DESC, matches_played) as position
    FROM TeamPoints
    ORDER BY total_points DESC, matches_played
    """
    return pd.read_sql_query(query, conn, params=[competition_name, before_date])

def calculate_match_importance(context, standings=None):
    """Calculate match importance (1-10) based on various factors"""
    comp_name = context['competition_name'].iloc[0]
    round_num = context['round_number'].iloc[0]
    
    # Base importance by competition tier
    competition_tiers = {
        'UEFA Champions League': 1,
        'Premier League': 2, 'LaLiga': 2, 'Bundesliga': 2, 'Serie A': 2, 'Ligue 1': 2,
        'UEFA Europa League': 3, 'FIFA Club World Cup': 3,
        'FA Cup': 4, 'Copa del Rey': 4, 'DFB-Pokal': 4, 'Coppa Italia': 4,
        'Coupe de France': 4, 'Eredivisie': 4, 'UEFA Europa Conference League': 4,
        'EFL Cup': 5, 'Swiss Super League': 5, 'Austrian Bundesliga': 5,
        'Danish Superliga': 5, 'Norwegian Eliteserien': 5, 'Swedish Allsvenskan': 5,
        'Eliteserien': 5
    }
    
    base_importance = 11 - competition_tiers.get(comp_name, 6)
    
    # Knockout stage bonus
    if any(cup in comp_name for cup in ['Cup', 'UEFA', 'FIFA']):
        if round_num:
            if round_num >= 7:  # Final stages
                base_importance += 4
            elif round_num >= 6:  # Semi
                base_importance += 3
            elif round_num >= 5:  # Quarter
                base_importance += 2
            elif round_num >= 4:  # Early knockout
                base_importance += 1
    
    # League position importance
    position_importance = 0
    if standings is not None and not standings.empty:
        home_team = context['home_team_name'].iloc[0]
        away_team = context['away_team_name'].iloc[0]
        
        home_pos = standings[standings['team_name'] == home_team]['position'].iloc[0] if not standings[standings['team_name'] == home_team].empty else 0
        away_pos = standings[standings['team_name'] == away_team]['position'].iloc[0] if not standings[standings['team_name'] == away_team].empty else 0
        
        # Title race
        if home_pos <= 3 or away_pos <= 3:
            position_importance += 2
        # European spots
        elif home_pos <= 6 or away_pos <= 6:
            position_importance += 1
        # Relegation battle
        elif home_pos >= len(standings) - 3 or away_pos >= len(standings) - 3:
            position_importance += 1
            
        # Close positions
        if abs(home_pos - away_pos) <= 2:
            position_importance += 1
    
    final_importance = min(10, base_importance + position_importance)
    
    return final_importance

def get_match_context(conn, match_id, derbies):
    """Get competition context for a match"""
    query = """
    SELECT 
        m.competition_name,
        m.venue_country as country,
        m.season_name,
        m.round_number,
        m.home_team_name,
        m.away_team_name,
        m.start_time
    FROM matches m
    WHERE m.match_id = ?
    """
    context = pd.read_sql_query(query, conn, params=[match_id])
    
    if context.empty:
        return default_context()
    
    # Get standings for league matches
    standings = None
    comp_name = context['competition_name'].iloc[0]
    if not any(cup in comp_name for cup in ['Cup', 'UEFA', 'FIFA']):
        standings = get_league_standings(conn, comp_name, context['start_time'].iloc[0])
    
    # Calculate importance
    importance = calculate_match_importance(context, standings)
    
    # Rest of the context calculations...
    is_knockout = bool(context['round_number'].iloc[0]) and any(cup in comp_name for cup in ['Cup', 'UEFA', 'FIFA'])
    
    international_competitions = [
        'UEFA Champions League',
        'UEFA Europa League',
        'UEFA Europa Conference League',
        'UEFA Super Cup',
        'FIFA Club World Cup'
    ]
    is_domestic = 0 if comp_name in international_competitions else 1
    
    # Derby check
    home_team = context['home_team_name'].iloc[0]
    away_team = context['away_team_name'].iloc[0]
    is_derby = int((home_team, away_team) in derbies.get(comp_name, []) or 
                   (away_team, home_team) in derbies.get(comp_name, []))
    
    return {
        'competition_name': comp_name,
        'competition_tier': competition_tiers.get(comp_name, 6),
        'is_knockout': int(is_knockout),
        'is_domestic': is_domestic,
        'match_importance': importance,
        'is_derby': is_derby
    }

def get_match_importance(context, league_context, derbies):
    """Calculate match importance (1-10) based on various factors"""
    try:
        # Initialize base importance
        importance = 5  # Default league match importance
        
        comp_name = context['competition_name'].iloc[0]
        round_num = context['round_number'].iloc[0]
        
        # Define competition tiers
        competition_tiers = {
            'UEFA Champions League': 1,
            'Premier League': 2, 'LaLiga': 2, 'Bundesliga': 2, 'Serie A': 2, 'Ligue 1': 2,
            'UEFA Europa League': 3, 'FIFA Club World Cup': 3,
            'FA Cup': 4, 'Copa del Rey': 4, 'DFB-Pokal': 4, 'Coppa Italia': 4,
            'Coupe de France': 4, 'Eredivisie': 4, 'UEFA Europa Conference League': 4,
            'EFL Cup': 5, 'Swiss Super League': 5, 'Austrian Bundesliga': 5,
            'Danish Superliga': 5, 'Norwegian Eliteserien': 5, 'Swedish Allsvenskan': 5,
            'Eliteserien': 5  # Added Norwegian league
        }
        
        # Get competition tier (default to 6 if not found)
        comp_tier = competition_tiers.get(comp_name, 6)
        
        # Check if it's a knockout match
        is_knockout = bool(round_num) and any(cup in comp_name for cup in ['Cup', 'UEFA', 'FIFA'])
        
        # Knockout stage importance
        if is_knockout:
            if round_num >= 7:  # Likely final stages
                importance = 10
            elif round_num >= 6:  # Likely semi
                importance = 9
            elif round_num >= 5:  # Likely quarter
                importance = 8
            elif round_num >= 4:  # Early knockout
                importance = 7
            else:  # Group/Early stages
                importance = 6
        
        # League context importance
        if not is_knockout:
            if league_context['title_race']:
                if league_context['matches_remaining'] <= 5:
                    importance += 3
                elif league_context['matches_remaining'] <= 10:
                    importance += 2
                else:
                    importance += 1
            
            if league_context['relegation_battle']:
                if league_context['matches_remaining'] <= 5:
                    importance += 2
                elif league_context['matches_remaining'] <= 10:
                    importance += 1
            
            if league_context['points_gap'] <= 3:
                importance += 1
        
        # Derby importance
        if league_context['is_derby']:
            importance += 1
        
        # Competition tier adjustment
        importance = min(10, importance + (7 - comp_tier))
        
        # Is domestic or European
        is_domestic = 1 if context['country'].iloc[0] != 'international' else 0
        
        result = {
            'match_importance': min(10, importance),
            'is_knockout': 1 if is_knockout else 0,
            'is_domestic': is_domestic,
            'competition_tier': comp_tier,
            'competition_name': comp_name
        }
        
        print("\nDebug - Match importance calculation result:", result)
        return result
        
    except Exception as e:
        print(f"\nError in get_match_importance: {str(e)}")
        # Return default values if there's an error
        return {
            'match_importance': 5,
            'is_knockout': 0,
            'is_domestic': 1,
            'competition_tier': 6,
            'competition_name': context['competition_name'].iloc[0] if not context.empty else 'Unknown'
        }

def get_league_context(conn, match_id, home_team, away_team, competition_name, date, derbies):
    """Get league standings context before the match"""
    # Skip for non-league matches
    if any(cup in competition_name for cup in ['Cup', 'UEFA', 'FIFA']):
        return {
            'is_derby': 1 if (home_team, away_team) in derbies.get(competition_name, []) else 0,
            'title_race': 0,
            'relegation_battle': 0,
            'points_gap': 0,
            'home_position': 0,
            'away_position': 0,
            'matches_remaining': 0
        }
    
    # For league matches, return basic context
    return {
        'is_derby': 1 if (home_team, away_team) in derbies.get(competition_name, []) else 0,
        'title_race': 0,
        'relegation_battle': 0,
        'points_gap': 0,
        'home_position': 1,
        'away_position': 2,
        'matches_remaining': 38
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
            WHERE row_num % 25 = 0
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
                
                # Get match context first
                match_context = get_match_context(conn, match['fixture_id'], derbies)
                if debug_mode:
                    print("\nMatch context:", match_context)
                
                # Get previous matches for both teams
                home_prev = get_previous_matches(conn, match['home_team'], match['start_time'])
                away_prev = get_previous_matches(conn, match['away_team'], match['start_time'])
                
                # Calculate form metrics
                home_metrics = calculate_team_stats(home_prev)
                away_metrics = calculate_team_stats(away_prev)
                
                # Now create basic_row with all context
                basic_row = {
                    'fixture_id': match['fixture_id'],
                    'home_team': match['home_team'],
                    'away_team': match['away_team'],
                    'home_matches_played': home_metrics['matches_played'],
                    'home_goals_scored': home_metrics['goals_scored'],
                    'home_goals_conceded': home_metrics['goals_conceded'],
                    'home_win_rate': home_metrics['win_rate'],
                    'home_clean_sheets': home_metrics['clean_sheets'],
                    'away_matches_played': away_metrics['matches_played'],
                    'away_goals_scored': away_metrics['goals_scored'],
                    'away_goals_conceded': away_metrics['goals_conceded'],
                    'away_win_rate': away_metrics['win_rate'],
                    'away_clean_sheets': away_metrics['clean_sheets'],
                    'home_goals': match['home_goals'],
                    'away_goals': match['away_goals'],
                    
                    # Simplified context features
                    'competition_name': match_context['competition_name'],
                    'competition_tier': match_context['competition_tier'],
                    'is_knockout': match_context['is_knockout'],
                    'is_domestic': match_context['is_domestic'],
                    'match_importance': match_context['match_importance'],
                    'is_derby': match_context['is_derby']
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