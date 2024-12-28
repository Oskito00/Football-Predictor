from datetime import datetime
import sqlite3
import pandas as pd
import numpy as np

from constants import DERBIES

#Need to implement: 
#Update team_table
#Get team_table

def initialize_database(conn):
    """Create necessary tables if they don't exist and clean existing data"""
    cursor = conn.cursor()
    
    # Drop existing tables to ensure we have the correct schema
    cursor.execute("DROP TABLE IF EXISTS team_running_stats")
    cursor.execute("DROP TABLE IF EXISTS season_points")
    
    # Create team_running_stats table
    cursor.execute('''CREATE TABLE team_running_stats (
        team_name TEXT,
        start_time TEXT,
        season_id TEXT,
        competition_id TEXT,
        match_id TEXT,
        match_status TEXT DEFAULT 'ended',
        
        -- Stats for the game
        goals_scored INTEGER,
        goals_conceded INTEGER,
        match_outcome TEXT,
        clean_sheet BOOLEAN,
        
        -- Advanced stats totals
        passes_successful INTEGER,
        passes_total INTEGER,
        shots_on_target INTEGER,
        shots_total INTEGER,
        chances_created INTEGER,
        tackles_successful INTEGER,
        tackles_total INTEGER,
        
        PRIMARY KEY (team_name, start_time)
    )''')
    
    # Create season_points table
    cursor.execute('''CREATE TABLE season_points (
        team_name TEXT,
        season_id TEXT,
        competition_id TEXT,
        points INTEGER DEFAULT 0,
        matches_played INTEGER DEFAULT 0,
        wins INTEGER DEFAULT 0,
        draws INTEGER DEFAULT 0,
        losses INTEGER DEFAULT 0,
        PRIMARY KEY (team_name, season_id, competition_id)
    )''')
    
    conn.commit()
    print("Tables dropped and recreated with new schema")


def get_previous_matches(conn, team_name, before_match_date):
    cursor = conn.cursor()
    cursor.execute("""
        SELECT *
        FROM team_running_stats 
        WHERE team_name = ?
        AND start_time < ?
        AND match_status = 'ended'
        ORDER BY start_time DESC
        LIMIT 5
    """, (team_name, before_match_date))
    return cursor.fetchall()

#TODO:Get this working so that it extracts useful form information from previous 5 matches
def calculate_form(conn, before_match_date):
    cursor = conn.cursor()

    home_previous_5_matches = get_previous_matches(conn, before_match_date['home_team'], before_match_date['start_time'])
    away_previous_5_matches = get_previous_matches(conn, before_match_date['away_team'], before_match_date['start_time'])

    # Initialize home stats counters
    home_goals_scored = home_goals_conceded = home_wins = home_clean_sheets = 0
    home_passes_successful = home_passes_total = home_shots_on_target = home_shots_total = 0
    home_chances_created = home_tackles_successful = home_tackles_total = 0

    # Initialize away stats counters  
    away_goals_scored = away_goals_conceded = away_wins = away_clean_sheets = 0
    away_passes_successful = away_passes_total = away_shots_on_target = away_shots_total = 0 
    away_chances_created = away_tackles_successful = away_tackles_total = 0
    # Initialize counters

    # Get number of available matches
    home_num_matches = len(home_previous_5_matches)  # Will be between 0 and 5
    away_num_matches = len(away_previous_5_matches)  # Will be between 0 and 5

    for stats in home_previous_5_matches:
        home_goals_scored += stats[6] if stats[6] is not None else 0
        home_goals_conceded += stats[7] if stats[7] is not None else 0
        home_wins += 1 if stats[8] == 'win' else 0
        home_clean_sheets += 1 if stats[9] == 1 else 0
        home_passes_successful += stats[10] if stats[10] is not None else 0
        home_passes_total += stats[11] if stats[11] is not None else 0
        home_shots_on_target += stats[12] if stats[12] is not None else 0
        home_shots_total += stats[13] if stats[13] is not None else 0
        home_tackles_successful += stats[14] if stats[14] is not None else 0
        home_tackles_total += stats[15] if stats[15] is not None else 0
        home_chances_created += stats[16] if stats[16] is not None else 0
    
    home_divisor = max(1, min(5, home_num_matches))
    away_divisor = max(1, min(5, away_num_matches))
    
    home_metrics = {
        'average_goals_scored': home_goals_scored / home_divisor,
        'average_goals_conceded': home_goals_conceded / home_divisor,
        'average_win_rate': home_wins / home_divisor,
        'average_clean_sheets': home_clean_sheets / home_divisor,
        'average_passes_successful': home_passes_successful / home_divisor,
        'average_passes_total': home_passes_total / home_divisor,
        'average_shots_on_target': home_shots_on_target / home_divisor,
        'average_shots_total': home_shots_total / home_divisor,
        'average_chances_created': home_chances_created / home_divisor,
        'average_tackles_successful': home_tackles_successful / home_divisor,
        'average_tackles_total': home_tackles_total / home_divisor
    }

    for stats in away_previous_5_matches:
        away_goals_scored += stats[6] if stats[6] is not None else 0
        away_goals_conceded += stats[7] if stats[7] is not None else 0
        away_wins += 1 if stats[8] == 'win' else 0
        away_clean_sheets += 1 if stats[9] == 1 else 0
        away_passes_successful += stats[10] if stats[10] is not None else 0
        away_passes_total += stats[11] if stats[11] is not None else 0
        away_shots_on_target += stats[12] if stats[12] is not None else 0
        away_shots_total += stats[13] if stats[13] is not None else 0
        away_tackles_successful += stats[14] if stats[14] is not None else 0
        away_tackles_total += stats[15] if stats[15] is not None else 0
        away_chances_created += stats[16] if stats[16] is not None else 0

    away_average_goals_scored = away_goals_scored / away_divisor
    away_average_goals_conceded = away_goals_conceded / away_divisor
    away_average_win_rate = away_wins / away_divisor
    away_average_clean_sheets = away_clean_sheets / away_divisor
    away_average_passes_successful = away_passes_successful / away_divisor
    away_average_passes_total = away_passes_total / away_divisor
    away_average_shots_on_target = away_shots_on_target / away_divisor
    away_average_shots_total = away_shots_total / away_divisor
    away_average_chances_created = away_chances_created / away_divisor
    away_average_tackles_successful = away_tackles_successful / away_divisor
    away_average_tackles_total = away_tackles_total / away_divisor

    #Final Away Metrics
    away_metrics = {'average_goals_scored': away_average_goals_scored, 'average_goals_conceded': away_average_goals_conceded, 'average_win_rate': away_average_win_rate, 'average_clean_sheets': away_average_clean_sheets, 'average_passes_successful': away_average_passes_successful, 'average_passes_total': away_average_passes_total, 'average_shots_on_target': away_average_shots_on_target, 'average_shots_total': away_average_shots_total, 'average_chances_created': away_average_chances_created, 'average_tackles_successful': away_average_tackles_successful, 'average_tackles_total': away_average_tackles_total}
    return home_metrics, away_metrics
def add_team_stats(conn, match):
    cursor = conn.cursor()
    
    try:
        # Map the expected column names to what's in your database
        home_stats = {
            'team_name': match['home_team'],
            'start_time': match['start_time'],
            'season_id': match['season_id'],
            'competition_id': match['competition_id'],
            'match_id': match['fixture_id'],
            'match_status': 'ended',
            'goals_scored': match['home_goals'],
            'goals_conceded': match['away_goals'],
            'match_outcome': 'win' if match['home_goals'] > match['away_goals'] else 'loss' if match['home_goals'] < match['away_goals'] else 'draw',
            'clean_sheet': match['away_goals'] == 0,
            'passes_successful': match.get('home_passes_successful'),
            'passes_total': match.get('home_passes_total'),
            'shots_on_target': match.get('home_shots_on_target'),
            'shots_total': match.get('home_shots_total'),
            'chances_created': match.get('home_chances_created'),
            'tackles_successful': match.get('home_tackles_successful'),
            'tackles_total': match.get('home_tackles_total')
        }

        away_stats = {
            'team_name': match['away_team'],
            'start_time': match['start_time'],
            'season_id': match['season_id'],
            'competition_id': match['competition_id'],
            'match_id': match['fixture_id'],
            'match_status': 'ended',
            'goals_scored': match['away_goals'],
            'goals_conceded': match['home_goals'],
            'match_outcome': 'win' if match['away_goals'] > match['home_goals'] else 'loss' if match['away_goals'] < match['home_goals'] else 'draw',
            'clean_sheet': match['home_goals'] == 0,
            'passes_successful': match.get('away_passes_successful'),
            'passes_total': match.get('away_passes_total'),
            'shots_on_target': match.get('away_shots_on_target'),
            'shots_total': match.get('away_shots_total'),
            'chances_created': match.get('away_chances_created'),
            'tackles_successful': match.get('away_tackles_successful'),
            'tackles_total': match.get('away_tackles_total')
        }

        cursor.execute("""
            INSERT INTO team_running_stats (
                team_name, start_time, season_id, competition_id, match_id,
                match_status, goals_scored, goals_conceded, match_outcome, clean_sheet,
                passes_successful, passes_total, shots_on_target, shots_total,
                chances_created, tackles_successful, tackles_total
            ) VALUES (
                :team_name, :start_time, :season_id, :competition_id, :match_id,
                :match_status, :goals_scored, :goals_conceded, :match_outcome, :clean_sheet,
                :passes_successful, :passes_total, :shots_on_target, :shots_total,
                :chances_created, :tackles_successful, :tackles_total
            )
        """, home_stats)

        cursor.execute("""
            INSERT INTO team_running_stats (
                team_name, start_time, season_id, competition_id, match_id,
                match_status, goals_scored, goals_conceded, match_outcome, clean_sheet,
                passes_successful, passes_total, shots_on_target, shots_total,
                chances_created, tackles_successful, tackles_total
            ) VALUES (
                :team_name, :start_time, :season_id, :competition_id, :match_id,
                :match_status, :goals_scored, :goals_conceded, :match_outcome, :clean_sheet,
                :passes_successful, :passes_total, :shots_on_target, :shots_total,
                :chances_created, :tackles_successful, :tackles_total
            )
        """, away_stats)

        print("Added team stats to database")

        conn.commit()        
    except Exception as e:
        print(f"Error adding match: {str(e)}")
        print(f"Match data: {match}")
        conn.rollback()
        raise

    return 0

def add_points_for_team(conn, match):
    """Update season points for both teams based on match outcome"""
    cursor = conn.cursor()
    
    try:
        # Determine points for each team
        if match['home_goals'] > match['away_goals']:
            home_points = 3
            away_points = 0
        elif match['home_goals'] < match['away_goals']:
            home_points = 0
            away_points = 3
        else:
            home_points = 1
            away_points = 1

        # Update home team points
        cursor.execute("""
            INSERT INTO season_points (
                team_name, season_id, competition_id, 
                points, matches_played, wins, draws, losses
            ) VALUES (
                ?, ?, ?,
                ?, 1,
                CASE WHEN ? = 3 THEN 1 ELSE 0 END,
                CASE WHEN ? = 1 THEN 1 ELSE 0 END,
                CASE WHEN ? = 0 THEN 1 ELSE 0 END
            )
            ON CONFLICT(team_name, season_id, competition_id) DO UPDATE SET
                points = points + ?,
                matches_played = matches_played + 1,
                wins = wins + CASE WHEN ? = 3 THEN 1 ELSE 0 END,
                draws = draws + CASE WHEN ? = 1 THEN 1 ELSE 0 END,
                losses = losses + CASE WHEN ? = 0 THEN 1 ELSE 0 END
        """, (
            match['home_team'], match['season_id'], match['competition_id'],
            home_points,  # Initial points
            home_points, home_points, home_points,  # For CASE statements in INSERT
            home_points,  # For points addition in UPDATE
            home_points, home_points, home_points   # For CASE statements in UPDATE
        ))

        # Update away team points
        cursor.execute("""
            INSERT INTO season_points (
                team_name, season_id, competition_id,
                points, matches_played, wins, draws, losses
            ) VALUES (
                ?, ?, ?,
                ?, 1,
                CASE WHEN ? = 3 THEN 1 ELSE 0 END,
                CASE WHEN ? = 1 THEN 1 ELSE 0 END,
                CASE WHEN ? = 0 THEN 1 ELSE 0 END
            )
            ON CONFLICT(team_name, season_id, competition_id) DO UPDATE SET
                points = points + ?,
                matches_played = matches_played + 1,
                wins = wins + CASE WHEN ? = 3 THEN 1 ELSE 0 END,
                draws = draws + CASE WHEN ? = 1 THEN 1 ELSE 0 END,
                losses = losses + CASE WHEN ? = 0 THEN 1 ELSE 0 END
        """, (
            match['away_team'], match['season_id'], match['competition_id'],
            away_points,  # Initial points
            away_points, away_points, away_points,  # For CASE statements in INSERT
            away_points,  # For points addition in UPDATE
            away_points, away_points, away_points   # For CASE statements in UPDATE
        ))

        conn.commit()

    except Exception as e:
        print(f"Error updating points: {str(e)}")
        conn.rollback()
        raise

def get_team_points(conn, match):
    """Get current points for both teams in the match for their current season"""
    cursor = conn.cursor()
    
    try:
        # Query points for both teams in one go
        cursor.execute("""
            SELECT team_name, points, matches_played
            FROM season_points
            WHERE team_name IN (?, ?)
            AND season_id = ?
            AND competition_id = ?
        """, (
            match['home_team'],
            match['away_team'],
            match['season_id'],
            match['competition_id']
        ))
        
        results = cursor.fetchall()
        
        # Initialize default values
        home_points = 0
        away_points = 0
        
        # Process results
        for team, points, matches in results:
            if team == match['home_team']:
                home_points = points
            elif team == match['away_team']:
                away_points = points
                
        return home_points, away_points

    except Exception as e:
        print(f"Error getting team points: {str(e)}")
        return 0, 0  # Return default values if there's an error

def get_league_positions(conn, match):
    """Get current league positions for both teams in the match"""
    cursor = conn.cursor()
    
    # Query that ranks teams by points (and goal difference if we had it)
    cursor.execute("""
        WITH ranked_teams AS (
            SELECT 
                team_name,
                points,
                matches_played,
                wins,
                ROW_NUMBER() OVER (
                    ORDER BY 
                        points DESC,
                        wins DESC,
                        matches_played ASC
                ) as position
            FROM season_points
            WHERE season_id = ? 
            AND competition_id = ?
        )
        SELECT 
            team_name,
            position,
            points,
            matches_played
        FROM ranked_teams
        WHERE team_name IN (?, ?)
        ORDER BY position ASC
    """, (
        match['season_id'],
        match['competition_id'],
        match['home_team'],
        match['away_team']
    ))
    
    results = cursor.fetchall()
    positions = {}
    
    for team, pos, pts, matches in results:
        positions[team] = {
            'position': pos,
            'points': pts,
            'matches_played': matches
        }
    
    return {
        'home': positions.get(match['home_team'], {'position': None, 'points': 0, 'matches_played': 0}),
        'away': positions.get(match['away_team'], {'position': None, 'points': 0, 'matches_played': 0})
    }

#TODO: Implement more features
def calculate_match_importance(conn, match):
    """Calculate the importance of a match based on various factors"""
    
    BASE_LEAGUE_IMPORTANCE = 5.0
    BASE_CUP_IMPORTANCE = 5.5
    DERBY_BONUS = 2.0
    TITLE_RACE_BONUS = 2.5
    RELEGATION_BATTLE_BONUS = 2.0
    POSITION_PROXIMITY_MAX_BONUS = 1.5
    POINTS_PROXIMITY_MAX_BONUS = 1.5

    CUP_IMPORTANCE = {
    'UEFA Champions League': 6.0,
    'UEFA Europa League': 5.5,
    'UEFA Conference League': 5.5,
    'FA Cup': 6.0,
    'EFL Cup': 5.5,
    'Coppa Italia': 5.5,
    'Coupe de France': 5.5,
    'Copa del Rey': 5.5,
}
    
    importance = 0.0
    
    # Get current league positions and points
    home_points, away_points = get_team_points(conn, match)
    positions = get_league_positions(conn, match)
    home_pos = positions['home']['position']
    away_pos = positions['away']['position']

    matches_played = max(positions['home']['matches_played'], 
                        positions['away']['matches_played'])
    
    is_cup_match = match['competition_type'] == 'cup'

    
    if is_cup_match:
        importance = BASE_CUP_IMPORTANCE
        
        cup_name = match['competition_name']
        if cup_name in CUP_IMPORTANCE:
            importance = CUP_IMPORTANCE[cup_name]
        
        cup_stage = match['round_display']
        # Try to extract a number from the cup stage first
        try:
            round_num = int(''.join(filter(str.isdigit, str(cup_stage))))
            round_weight = round_num * 0.5
        except ValueError:
            # If no number found, check for specific round names
            stage_lower = str(cup_stage).lower()
            if 'round_of_16' in stage_lower or 'last_16' in stage_lower:
                round_weight = 3.0
            elif 'quarterfinal' in stage_lower or 'quarter' in stage_lower:
                round_weight = 4.0
            elif 'semifinal' in stage_lower or 'semi' in stage_lower:
                round_weight = 5.0
            elif 'final' in stage_lower:
                round_weight = 6.0
            else:
                round_weight = 0.5  # Default weight for unrecognized rounds

        importance += round_weight
        
    else:  # League match
        importance = BASE_LEAGUE_IMPORTANCE
        
        # Check if it's a derby by looking up teams directly
        home_team = match['home_team']
        away_team = match['away_team']
        
        # Check both orderings of teams across all competitions
        is_derby = False
        for derby_list in DERBIES.values():
            for team1, team2 in derby_list:
                if (home_team == team1 and away_team == team2) or \
                   (home_team == team2 and away_team == team1):
                    is_derby = True
                    break
            if is_derby:
                break
        
        if is_derby:
            print(f"Derby match bonus: +{DERBY_BONUS}")
            importance += DERBY_BONUS
        
        # Position proximity bonus (closer positions = more important)
        if home_pos and away_pos and (home_pos <= 6 or away_pos <= 6):
            position_bonus = max(0, POSITION_PROXIMITY_MAX_BONUS * (1 - (max(home_pos, away_pos) / 10)))
            position_bonus *= 1.25  # 25% bonus for matches between top teams
            importance += position_bonus
        
        # Points proximity bonus
        if home_points is not None and away_points is not None:
            points_diff = abs(home_points - away_points)
            points_bonus = max(0, POINTS_PROXIMITY_MAX_BONUS * (1 - (points_diff / 9)))
            importance += points_bonus
        # Check if it's late in the season (more than 70% complete)
        TOTAL_SEASON_MATCHES = 38
        is_late_season = matches_played > (TOTAL_SEASON_MATCHES * 0.7)
        if is_late_season:
            # Title race check (teams near top of table)
            if (home_pos and home_pos <= 3) or (away_pos and away_pos <= 3):
                importance += TITLE_RACE_BONUS
            # Relegation battle check (teams near bottom)
            #TODO: Make this dynamic
            TEAMS_IN_LEAGUE = 20
            relegation_zone = TEAMS_IN_LEAGUE - 3
            if (home_pos and home_pos >= relegation_zone) or (away_pos and away_pos >= relegation_zone):
                importance += RELEGATION_BATTLE_BONUS
        # Late season importance multipliers
        if matches_played > (TOTAL_SEASON_MATCHES * 0.9):  # Final stretch
            TITLE_RACE_BONUS *= 1.5
            RELEGATION_BATTLE_BONUS *= 1.5
        elif matches_played > (TOTAL_SEASON_MATCHES * 0.8):  # Very late
            TITLE_RACE_BONUS *= 1.25
            RELEGATION_BATTLE_BONUS *= 1.25
    
    return (round(importance, 2))