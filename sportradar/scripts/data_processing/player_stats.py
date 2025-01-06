from datetime import datetime
import json
from typing import Dict, List, Any

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
    
    conn.commit()
    print("Player running stats table initialized successfully")

def calculate_player_match_importance(player_stats):
    """
    Calculate a player's match importance score (0-100) with special consideration for goalkeepers
    when goalkeeper-specific stats are present.
    """
    score = 0
    minutes_weight = min(1.0, player_stats.get('minutes_played', 0) / 90)
    
    # Check if player has goalkeeper stats
    is_goalkeeper = (
        player_stats.get('diving_saves', 0) > 0 or 
        player_stats.get('penalties_saved', 0) > 0 or
        player_stats.get('shots_faced_saved', 0) > 0
    )
    
    #TODO: Make Goalkeeper scoring also take into account the number of goals conceded from match_data because I don't have that data in player_stats
    if is_goalkeeper:
        # Goalkeeper-specific scoring (max 75 points for positive actions)
        
        # 1. Shot Stopping (max 35 points)
        save_rate = 0
        if player_stats.get('shots_faced_total', 0) > 0:
            save_rate = (player_stats.get('shots_faced_saved', 0) / player_stats.get('shots_faced_total', 1)) * 100
        
        shot_stopping_score = (
            (player_stats.get('diving_saves', 0) * 3) +
            (save_rate * 0.2) +  # Up to 20 points for save rate
            (player_stats.get('penalties_saved', 0) * 8)
        )
        score += min(35, shot_stopping_score)
        
        # 2. Clean Sheet Bonus (max 20 points)
        if player_stats.get('goals_conceded', 0) == 0:
            score += 20
        elif player_stats.get('goals_conceded', 0) == 1:
            score += 10
        
        # 3. Distribution (max 20 points)
        pass_accuracy = 0
        if player_stats.get('passes_total', 0) > 0:
            pass_accuracy = (player_stats.get('passes_successful', 0) / player_stats.get('passes_total', 1)) * 100
        
        distribution_score = (
            (pass_accuracy * 0.15) +  # Up to 15 points for pass accuracy
            (min(player_stats.get('long_passes_successful', 0) * 0.5, 5))  # Max 5 points from long passes
        )
        score += min(20, distribution_score)
        
    else:
        # Original outfield player scoring
        # 1. Goal Contributions (max 25 points)
        goal_contribution_score = (
            (player_stats.get('goals_scored', 0) * 10) +
            (player_stats.get('assists', 0) * 8) +
            (player_stats.get('chances_created', 0) * 3)
        )
        score += min(25, goal_contribution_score)
        
        # 2. Defensive Actions (max 25 points)
        defensive_score = (
            (player_stats.get('tackles_successful', 0) * 3) +
            (player_stats.get('clearances', 0) * 2) +
            (player_stats.get('interceptions', 0) * 2) +
            (player_stats.get('defensive_blocks', 0) * 2)
        )
        score += min(25, defensive_score)
        
        # 3. Ball Control & Distribution (max 25 points)
        pass_accuracy = 0
        if player_stats.get('passes_total', 0) > 0:
            pass_accuracy = (player_stats.get('passes_successful', 0) / player_stats.get('passes_total', 1)) * 100
        
        possession_score = (
            (pass_accuracy * 0.15) +  # Up to 15 points for pass accuracy
            (player_stats.get('dribbles_completed', 0) * 2) +
            (min(player_stats.get('crosses_successful', 0) * 3, 10))  # Max 10 points from crosses
        )
        score += min(25, possession_score)
    
    # 4. Negative Actions (directly subtract points) - Applied to both GKs and outfield players
    negative_score = (
        (player_stats.get('yellow_cards', 0) * -5) +      # -5 points per yellow
        (player_stats.get('red_cards', 0) * -15) +        # -15 points per red
        (player_stats.get('loss_of_possession', 0) * -1) + # -1 point per loss of possession
        (player_stats.get('fouls_committed', 0) * -2)      # -2 points per foul
    )

    score += negative_score
    
    # Apply minutes played weight
    final_score = score * minutes_weight
    
    # Ensure score stays between 0 and 100
    return min(100, max(0, final_score))

def update_player_running_stats(conn, player_stats):
    """
    Update player's running stats with new match data.
    Calculates form before and after adding the current match.
    """
    cursor = conn.cursor()
    
    try:
        # Get player's recent scores
        cursor.execute("""
            SELECT match_importance_score, start_time 
            FROM player_running_stats 
            WHERE player_id = ? 
            ORDER BY datetime(start_time) DESC 
            LIMIT 20
        """, (player_stats['player_id'],))
        
        recent_scores = cursor.fetchall()
        
        # Convert to list of scores
        recent_scores = [score[0] for score in recent_scores]
        
        # Calculate pre-match stats (form before this match)
        if recent_scores:
            overall_importance = sum(recent_scores[:20]) / len(recent_scores[:20])
            form_rating = sum(recent_scores[:5]) / len(recent_scores[:5]) if len(recent_scores) >= 5 else sum(recent_scores) / len(recent_scores)
    
            # Calculate trend using new function
            trend = calculate_trend(recent_scores)
        else:
            overall_importance = 0
            form_rating = 0
            trend = 'stable'
        
        # Calculate importance score for current match
        match_importance = calculate_player_match_importance(player_stats)
        
        # Insert new stats
        cursor.execute("""
            INSERT INTO player_running_stats (
                player_id, player_name, team_id, start_time, match_id,
                match_importance_score, overall_importance_score,
                form_rating, matches_counted, form_trend, last_updated
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
        """, (
            player_stats['player_id'],
            player_stats['player_name'],
            player_stats['team_id'],
            player_stats['start_time'],
            player_stats['match_id'],
            match_importance,
            overall_importance, #This is pre-match overall importance
            form_rating,  # This is the pre-match form
            min(len(recent_scores), 20),
            trend
        ))
        
        conn.commit()
        
    except Exception as e:
        print(f"Error updating running stats: {str(e)}")
        conn.rollback()
        raise #If I want to I can return overall_importance, form_rating, trend from this


def process_match_stats(conn, match_id):
    """
    Process all players' stats for a match and update their running stats
    
    Args:
        conn: Database connection
        match_id: ID of the match
        
    Raises:
        ValueError: If match_id doesn't exist or no player stats found
    """
    cursor = conn.cursor()
    
    # First validate match_id exists
    cursor.execute("""
        SELECT COUNT(*) 
        FROM player_stats 
        WHERE match_id = ?
    """, (match_id,))
    
    count = cursor.fetchone()[0]
    if count == 0:
        raise ValueError(f"No data found for match_id: {match_id}")
    
    # Get all player stats for the match
    player_stats_list = get_match_player_stats(conn, match_id)
    
    if not player_stats_list:
        raise ValueError(f"No player stats found for match_id: {match_id}")
    
    # Update running stats for each player
    processed_count = 0
    errors = []
    
    for player_stats in player_stats_list:
        try:
            update_player_running_stats(conn, player_stats)
            processed_count += 1
        except Exception as e:
            errors.append(f"Error processing player {player_stats.get('player_name', 'unknown')}: {str(e)}")
    
    # Report results
    print(f"Processed {processed_count} out of {len(player_stats_list)} players for match {match_id}")
    
    if errors:
        print("\nErrors encountered:")
        for error in errors:
            print(error)
        
    if processed_count == 0:
        raise ValueError(f"Failed to process any players for match_id: {match_id}")
        
    return processed_count


#Helper functions

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


