from datetime import datetime
import sqlite3
import pandas as pd
import numpy as np

from match_helpers import get_previous_matches

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

def calculate_stats_averages(conn, before_match_date):
    cursor = conn.cursor()

    home_team_stats = get_previous_matches(conn, before_match_date['home_team'], before_match_date['start_time'])
    away_team_stats = get_previous_matches(conn, before_match_date['away_team'], before_match_date['start_time'])

    # Initialize counters
    goals_scored = goals_conceded = wins = clean_sheets = 0
    passes_successful = passes_total = shots_on_target = shots_total = 0
    chances_created = tackles_successful = tackles_total = 0

    # Get number of available matches
    home_num_matches = len(home_team_stats)  # Will be between 0 and 5
    away_num_matches = len(away_team_stats)  # Will be between 0 and 5

    for stats in home_team_stats:
        goals_scored += stats['goals_scored']
        goals_conceded += stats['goals_conceded']
        wins += stats['wins']
        clean_sheets += stats['clean_sheets']
        passes_successful += stats['passes_successful']
        passes_total += stats['passes_total']
        shots_on_target += stats['shots_on_target']
        shots_total += stats['shots_total']
        chances_created += stats['chances_created']
        tackles_successful += stats['tackles_successful']
        tackles_total += stats['tackles_total']
    
    home_divisor = max(1, min(5, home_num_matches))
    away_divisor = max(1, min(5, away_num_matches))

    
    home_metrics = {
        'average_goals_scored': goals_scored / home_divisor,
        'average_goals_conceded': goals_conceded / home_divisor,
        'average_win_rate': wins / home_divisor,
        'average_clean_sheets': clean_sheets / home_divisor,
        'average_passes_successful': passes_successful / home_divisor,
        'average_passes_total': passes_total / home_divisor,
        'average_shots_on_target': shots_on_target / home_divisor,
        'average_shots_total': shots_total / home_divisor,
        'average_chances_created': chances_created / home_divisor,
        'average_tackles_successful': tackles_successful / home_divisor,
        'average_tackles_total': tackles_total / home_divisor
    }

    for stats in away_team_stats:
        goals_scored += stats['goals_scored']
        goals_conceded += stats['goals_conceded']
        wins += stats['wins']
        clean_sheets += stats['clean_sheets']
        passes_successful += stats['passes_successful']
        passes_total += stats['passes_total']
        shots_on_target += stats['shots_on_target']
        shots_total += stats['shots_total']
        chances_created += stats['chances_created']
        tackles_successful += stats['tackles_successful']
        tackles_total += stats['tackles_total']

    away_average_goals_scored = goals_scored / away_divisor
    away_average_goals_conceded = goals_conceded / away_divisor
    away_average_win_rate = wins / away_divisor
    away_average_clean_sheets = clean_sheets / away_divisor
    away_average_passes_successful = passes_successful / away_divisor
    away_average_passes_total = passes_total / away_divisor
    away_average_shots_on_target = shots_on_target / away_divisor
    away_average_shots_total = shots_total / away_divisor
    away_average_chances_created = chances_created / away_divisor
    away_average_tackles_successful = tackles_successful / away_divisor
    away_average_tackles_total = tackles_total / away_divisor

    #Final Away Metrics
    away_metrics = {'average_goals_scored': away_average_goals_scored, 'average_goals_conceded': away_average_goals_conceded, 'average_win_rate': away_average_win_rate, 'average_clean_sheets': away_average_clean_sheets, 'average_passes_successful': away_average_passes_successful, 'average_passes_total': away_average_passes_total, 'average_shots_on_target': away_average_shots_on_target, 'average_shots_total': away_average_shots_total, 'average_chances_created': away_average_chances_created, 'average_tackles_successful': away_average_tackles_successful, 'average_tackles_total': away_average_tackles_total}

def add_team_stats(conn, match):
    cursor = conn.cursor()
    
    # Debug print
    print(f"\nAttempting to add match: {match.get('fixture_id')}")
    print(f"Match data available: {match.keys()}")
    
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

        # Debug print
        print(f"Home stats prepared: {home_stats}")
        print(f"Away stats prepared: {away_stats}")

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

        conn.commit()
        print("Successfully added match to database")
        
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

        return home_points, away_points

    except Exception as e:
        print(f"Error updating points: {str(e)}")
        conn.rollback()
        raise

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

def calculate_match_importance(conn, match):
    """Calculate the importance of a match based on various factors"""
    
    # TODO: Need to verify these constants are appropriate
    BASE_LEAGUE_IMPORTANCE = 5.0
    BASE_CUP_IMPORTANCE = 5.5
    DERBY_BONUS = 2.0
    TITLE_RACE_BONUS = 2.5
    RELEGATION_BATTLE_BONUS = 2.0
    POSITION_PROXIMITY_MAX_BONUS = 1.5
    POINTS_PROXIMITY_MAX_BONUS = 1.5
    
    importance = 0.0
    
    # Get current league positions and points
    positions = get_league_positions(conn, match)
    home_pos = positions['home']['position']
    away_pos = positions['away']['position']
    home_points = positions['home']['points']
    away_points = positions['away']['points']
    matches_played = positions['home']['matches_played']  # Use home team's matches played as reference
    
    # TODO: Need a way to identify if it's a cup match
    is_cup_match = match['competition_type'] == 'cup'
    print(f"Is cup match: {is_cup_match}")
    
    if is_cup_match:
        importance = BASE_CUP_IMPORTANCE
        
        # TODO: Need to verify how to get cup round information
        cup_round = match.get('cup_round', 0)
        
        # Increase importance based on cup round
        if cup_round > 0:
            importance += (cup_round * 0.5)  # Later rounds are more important
        
        # Final match
        if match.get('is_final', False):
            importance += 3.0
            
    else:  # League match
        importance = BASE_LEAGUE_IMPORTANCE
        
        # Check if it's a derby
        # TODO: Need a function or list to identify derby matches
        if is_derby_match(match['home_team'], match['away_team']):
            importance += DERBY_BONUS
        
        # Position proximity bonus (closer positions = more important)
        if home_pos and away_pos:  # Only if both positions are available
            position_diff = abs(home_pos - away_pos)
            position_bonus = max(0, POSITION_PROXIMITY_MAX_BONUS * (1 - (position_diff / 10)))
            importance += position_bonus
        
        # Points proximity bonus
        if home_points is not None and away_points is not None:
            points_diff = abs(home_points - away_points)
            points_bonus = max(0, POINTS_PROXIMITY_MAX_BONUS * (1 - (points_diff / 9)))
            importance += points_bonus
        
        # Check if it's late in the season (more than 70% complete)
        # TODO: Need to verify total matches in season (usually 38 in top leagues)
        TOTAL_SEASON_MATCHES = 38
        is_late_season = matches_played > (TOTAL_SEASON_MATCHES * 0.7)
        
        if is_late_season:
            # Title race check (teams near top of table)
            if (home_pos and home_pos <= 3) or (away_pos and away_pos <= 3):
                importance += TITLE_RACE_BONUS
            
            # Relegation battle check (teams near bottom)
            # TODO: Need to verify number of teams in league to determine relegation zone
            TEAMS_IN_LEAGUE = 20
            relegation_zone = TEAMS_IN_LEAGUE - 3
            if (home_pos and home_pos >= relegation_zone) or (away_pos and away_pos >= relegation_zone):
                importance += RELEGATION_BATTLE_BONUS
    
    return round(importance, 2)