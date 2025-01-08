from datetime import datetime
import json
from typing import Dict, List, Any
import numpy as np

def initialize_player_database(conn):
    """Create necessary tables for tracking player running statistics"""
    cursor = conn.cursor()
    
    # Drop existing tables to ensure we have the correct schema
    cursor.execute("DROP TABLE IF EXISTS player_running_stats")
    
    # Create player_running_stats table
    cursor.execute('''CREATE TABLE IF NOT EXISTS player_running_stats (
        player_id TEXT,
        player_name TEXT,
        team_id TEXT,
        start_time TEXT,
        match_id TEXT,
        
        -- The important scores
        match_importance_score REAL DEFAULT 0,  -- Score for this specific match
        overall_importance_score REAL DEFAULT 0, -- Average over 20 games
        form_rating REAL DEFAULT 0,             -- Based on last 5 games
        
        -- Tracking
        matches_counted INTEGER DEFAULT 0,      -- How many matches in the average (up to 20)
        form_trend TEXT,                        -- JSON: increasing/decreasing/stable
        
        -- Metadata
        last_updated TEXT,
        
        PRIMARY KEY (player_id, match_id)
    )''')
    
    # Create indices for common queries
    cursor.execute('''CREATE INDEX IF NOT EXISTS idx_player_stats_time 
                     ON player_running_stats(player_id, start_time)''')
    cursor.execute('''CREATE INDEX IF NOT EXISTS idx_player_stats_team 
                     ON player_running_stats(team_id, start_time)''')
    
    # Create simplified player_squad_status table
    cursor.execute('''CREATE TABLE IF NOT EXISTS player_squad_status (
        player_id TEXT,
        team_id TEXT,
        player_name TEXT,
        last_appearance TEXT,
        is_in_squad BOOLEAN DEFAULT FALSE,
        PRIMARY KEY (player_id, team_id)
    )''')
    
    conn.commit()
    print("Player databases initialized successfully")

def calculate_player_match_importance(player_stats):
    """
    Calculate a player's match importance score (0-35) with scaled defensive actions
    """
    # Default all stats to 0 if None
    stats = {k: (v if v is not None else 0) for k, v in player_stats.items()}
    
    score = 0
    minutes_weight = min(1.0, stats.get('minutes_played', 0) / 90)
    
    # Check if player has goalkeeper stats
    is_goalkeeper = (
        stats.get('diving_saves', 0) > 0 or 
        stats.get('penalties_saved', 0) > 0 or
        stats.get('shots_faced_saved', 0) > 0
    )
    
    if is_goalkeeper:
        # Goalkeeper scoring (reduced by ~30%)
        # 1. Shot Stopping (max 25 points, down from 35)
        save_rate = 0
        if stats.get('shots_faced_total', 0) > 0:
            save_rate = (stats.get('shots_faced_saved', 0) / stats.get('shots_faced_total', 1)) * 100
        
        shot_stopping_score = (
            (stats.get('diving_saves', 0) * 2) +           # Reduced from 3
            (save_rate * 0.14) +                           # Reduced from 0.2
            (stats.get('penalties_saved', 0) * 5.5)        # Reduced from 8
        )
        score += min(25, shot_stopping_score)              # Reduced from 35
        
        # 2. Clean Sheet Bonus (max 14 points, down from 20)
        if stats.get('goals_conceded', 0) == 0:
            score += 14                                    # Reduced from 20
        elif stats.get('goals_conceded', 0) == 1:
            score += 7                                     # Reduced from 10
        
        # 3. Distribution (max 14 points, down from 20)
        pass_accuracy = 0
        if stats.get('passes_total', 0) > 0:
            pass_accuracy = (stats.get('passes_successful', 0) / stats.get('passes_total', 1)) * 100
        
        distribution_score = (
            (pass_accuracy * 0.1) +                        # Reduced from 0.15
            (min(stats.get('long_passes_successful', 0) * 0.35, 3.5))  # Reduced from 0.5 and 5
        )
        score += min(14, distribution_score)               # Reduced from 20
        
    else:
        # 1. Goal Contributions (max 25 points)
        goal_contribution_score = (
            (stats.get('goals_scored', 0) * 10) +
            (stats.get('assists', 0) * 8) +
            (stats.get('chances_created', 0) * 3)
        )
        score += min(25, goal_contribution_score)
        
        # 2. Defensive Actions (scaled down by ~25%)
        defensive_score = (
            (stats.get('tackles_successful', 0) * 2.25) +  # Reduced from 3
            (stats.get('clearances', 0) * 1.5) +          # Reduced from 2
            (stats.get('interceptions', 0) * 1.5) +       # Reduced from 2
            (stats.get('defensive_blocks', 0) * 1.5)      # Reduced from 2
        )
        score += min(25, defensive_score)
        
        # 3. Ball Control & Distribution (max 25 points)
        pass_accuracy = 0
        if stats.get('passes_total', 0) > 0:
            pass_accuracy = (stats.get('passes_successful', 0) / stats.get('passes_total', 1)) * 100
        
        possession_score = (
            (pass_accuracy * 0.15) +  # Up to 15 points for pass accuracy
            (stats.get('dribbles_completed', 0) * 2) +
            (min(stats.get('crosses_successful', 0) * 3, 10))  # Max 10 points from crosses
        )
        score += min(25, possession_score)
    
    # 4. Negative Actions (directly subtract points)
    negative_score = (
        (stats.get('yellow_cards', 0) * -5) +      # -5 points per yellow
        (stats.get('red_cards', 0) * -15) +        # -15 points per red
        (stats.get('loss_of_possession', 0) * -1) + # -1 point per loss of possession
        (stats.get('fouls_committed', 0) * -2)      # -2 points per foul
    )
    
    # Additional goalkeeper penalties
    if is_goalkeeper:
        negative_score += (
            (stats.get('goals_conceded', 0) * -3) +     # -3 points per goal conceded
            (stats.get('penalties_faced', 0) * -1)      # -1 point per penalty faced
        )
    
    score += negative_score
    
    # Apply minutes played weight and cap at 35
    final_score = score * minutes_weight
    return min(35, max(0, final_score))

def update_player_running_stats(conn, player_stats):
    """Update running stats for a player with detailed timing"""
    cursor = conn.cursor()
    
    try:
        # Get recent scores
        cursor.execute("""
            SELECT match_importance_score 
            FROM player_running_stats 
            WHERE player_id = ? 
            ORDER BY start_time DESC 
            LIMIT 20
        """, (player_stats['player_id'],))
        recent_scores = [row[0] for row in cursor.fetchall()]

        # Calculate new stats
        match_importance = calculate_player_match_importance(player_stats)
        overall_importance = (sum(recent_scores) / len(recent_scores)) if recent_scores else 0
        form_rating = (sum(recent_scores[:5]) / len(recent_scores[:5])) if len(recent_scores) >= 5 else overall_importance
        trend = calculate_trend(recent_scores)

        # Insert new record
        cursor.execute("""
            INSERT INTO player_running_stats (
                player_id, player_name, team_id, start_time, match_id,
                match_importance_score, overall_importance_score,
                form_rating, matches_counted, form_trend
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            player_stats['player_id'],
            player_stats['player_name'],
            player_stats['team_id'],
            player_stats['start_time'],
            player_stats['match_id'],
            match_importance,
            overall_importance,
            form_rating,
            len(recent_scores),
            trend
        ))

    except Exception as error:
        print(f"Error updating player {player_stats.get('player_name', 'unknown')}: {str(error)}")
        raise

def update_squad_status(conn, player_stats):
    """
    Update player's squad status after each match
    """
    cursor = conn.cursor()
    
    try:
        # Update or insert player status
        cursor.execute("""
            INSERT INTO player_squad_status (
                player_id, team_id, player_name, last_appearance, is_in_squad
            ) VALUES (?, ?, ?, ?, TRUE)
            ON CONFLICT(player_id, team_id) DO UPDATE SET
                last_appearance = ?,
                is_in_squad = TRUE
        """, (
            player_stats['player_id'],
            player_stats['team_id'],
            player_stats['player_name'],
            player_stats['start_time'],
            player_stats['start_time']
        ))
        
        # Update is_in_squad for players who haven't played in 30 days
        cursor.execute("""
            UPDATE player_squad_status
            SET is_in_squad = FALSE
            WHERE team_id = ?
            AND datetime(last_appearance) < datetime(?, '-30 days')
        """, (
            player_stats['team_id'],
            player_stats['start_time']
        ))
        
        conn.commit()
        
    except Exception as e:
        print(f"Error updating squad status: {str(e)}")
        conn.rollback()
        raise

def process_match_stats(conn, fixture_id, home_team_id, away_team_id, start_time):
    """
    Process all players' stats for a match and update their running stats
    Returns processed count and key player information
    """
    # Process all players first
    player_stats_list = get_match_player_stats(conn, fixture_id)
    
    if not player_stats_list:
        raise ValueError(f"No player stats found for match_id: {fixture_id}")
    
    # Update running stats for each player
    processed_count = 0
    errors = []
    
    for i, player_stats in enumerate(player_stats_list):
        try:
            update_player_running_stats(conn, player_stats)
            processed_count += 1
        except Exception as error:
            errors.append(f"Error processing player {player_stats.get('player_name', 'unknown')}: {str(error)}")
    
    
    if processed_count == 0:
        raise ValueError(f"Failed to process any players for match_id: {fixture_id}")
    

    # Get missing key players for both teams using passed parameters
    home_missing = get_missing_key_players(conn, fixture_id, home_team_id, start_time)
    away_missing = get_missing_key_players(conn, fixture_id, away_team_id, start_time)
    
    # Get total key players count
    home_key_players_total = get_key_players_count(conn, home_team_id) #TODO: Could combine into a single query
    away_key_players_total = get_key_players_count(conn, away_team_id)
    
    return {
        'processed_count': processed_count,
        'home_team_id': home_team_id,
        'away_team_id': away_team_id,
        'home_key_players_missing': home_missing,
        'away_key_players_missing': away_missing,
        'home_squad_strength': home_key_players_total - len(home_missing),
        'away_squad_strength': away_key_players_total - len(away_missing)
    }

#Helper functions

def get_missing_key_players(conn, match_id, team_id, start_time):
    """Get key players who didn't play in this match"""
    cursor = conn.cursor()
    
    # Get current squad players with high importance scores
    cursor.execute("""
        WITH current_squad AS (
            SELECT ps.player_id, ps.player_name
            FROM player_squad_status ps
            WHERE ps.team_id = ? 
            AND ps.is_in_squad = TRUE
        ),
        latest_stats AS (
            SELECT 
                prs.player_id,
                prs.player_name,
                prs.overall_importance_score,
                prs.form_rating,
                prs.start_time,
                ROW_NUMBER() OVER (
                    PARTITION BY prs.player_id 
                    ORDER BY prs.start_time DESC
                ) as rn
            FROM player_running_stats prs
            JOIN current_squad cs ON prs.player_id = cs.player_id
            WHERE prs.team_id = ?
            AND prs.overall_importance_score >= 15
            AND prs.form_rating >= 15
            AND datetime(prs.start_time) >= datetime(?, '-30 days')
        )
        SELECT 
            player_id,
            player_name,
            overall_importance_score,
            form_rating
        FROM latest_stats
        WHERE rn = 1
        AND player_id NOT IN (
            SELECT player_id 
            FROM player_running_stats 
            WHERE match_id = ? 
            AND team_id = ?
        )
    """, (team_id, team_id, start_time, match_id, team_id))
    
    return [
        {
            'player_id': row[0],
            'player_name': row[1],
            'importance_score': row[2],
            'form_rating': row[3]
        }
        for row in cursor.fetchall()
    ]

def get_key_players_count(conn, team_id):
    """Get count of key players for a team"""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT COUNT(DISTINCT prs.player_id)
        FROM player_running_stats prs
        JOIN player_squad_status pss ON prs.player_id = pss.player_id
        WHERE prs.team_id = ?
        AND pss.is_in_squad = TRUE
        AND prs.overall_importance_score >= 15
        AND prs.form_rating >= 15
    """, (team_id,))
    return cursor.fetchone()[0]

def calculate_trend(scores):
    """
    Calculate trend based on available scores (up to 5 matches)
    Returns: 'increasing', 'decreasing', or 'stable'
    """
    if len(scores) <= 1:
        return 'stable'
    
    # Use up to 5 most recent scores
    recent_five = scores[:5]
    
    # Calculate differences between consecutive matches
    differences = [recent_five[i] - recent_five[i+1] for i in range(len(recent_five)-1)]
    
    # Count positive and negative differences
    positives = sum(1 for d in differences if d > 0)
    negatives = sum(1 for d in differences if d < 0)
    
    # Calculate trend based on majority direction
    if len(differences) >= 2:  # Need at least 3 matches for meaningful trend
        if positives > len(differences) / 2:
            return 'increasing'
        elif negatives > len(differences) / 2:
            return 'decreasing'
    
    # If no clear trend or not enough matches
    return 'stable'

def get_match_player_stats(conn, match_id):
    """
    Get all player stats for a specific match
    
    Args:
        conn: Database connection
        match_id: ID of the match
    
    Returns:
        list: List of dictionaries containing each player's stats
    """
    cursor = conn.cursor()
    
    try:
        # Get all players' stats for this match
        cursor.execute("""
            SELECT *
            FROM player_stats 
            WHERE match_id = ?
        """, (match_id,))
        
        # Get column names
        columns = [description[0] for description in cursor.description]
        
        # Convert rows to list of dictionaries
        player_stats = []
        for row in cursor.fetchall():
            player_dict = dict(zip(columns, row))
            player_stats.append(player_dict)
            
        return player_stats
        
    except Exception as e:
        print(f"Error getting player stats for match {match_id}: {str(e)}")
        raise
