import sqlite3
from datetime import datetime
import pandas as pd

def get_team_stats(conn, team_id, before_date, limit=5):
    """Get the last 5 matches' stats for a team before a specific date"""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT 
            match_id,
            passes_successful,
            passes_total,
            shots_on_target,
            shots_total,
            tackles_successful,
            tackles_total
        FROM team_stats 
        WHERE team_id = ?
        AND match_id IN (
            SELECT match_id 
            FROM matches 
            WHERE start_time < ?
            ORDER BY start_time DESC
            LIMIT ?
        )
    """, (team_id, before_date, limit))
    
    return cursor.fetchall()

def calculate_advanced_stats(stats_list):
    """Calculate advanced stats from raw stats"""
    totals = {
        'passes_successful': sum(stat[1] or 0 for stat in stats_list),
        'passes_total': sum(stat[2] or 0 for stat in stats_list),
        'shots_on_target': sum(stat[3] or 0 for stat in stats_list),
        'shots_total': sum(stat[4] or 0 for stat in stats_list),
        'tackles_successful': sum(stat[5] or 0 for stat in stats_list),
        'tackles_total': sum(stat[6] or 0 for stat in stats_list)
    }
    
    return {
        'pass_effectiveness': totals['passes_successful'] / max(totals['passes_total'], 1),
        'shot_accuracy': totals['shots_on_target'] / max(totals['shots_total'], 1),
        'defensive_success': totals['tackles_successful'] / max(totals['tackles_total'], 1)
    }

def main():
    # Read the test data
    df = pd.read_csv('sportradar/scripts/tests/advanced_stats_test.csv')
    
    # Connect to database
    conn = sqlite3.connect('football_data.db')
    
    for _, match in df.iterrows():
        print(f"\nAnalyzing match: {match['fixture_id']}")
        print(f"Date: {match['start_time']}")
        print(f"Teams: {match['home_team']} vs {match['away_team']}")
        
        # Get home team stats
        home_stats = get_team_stats(conn, match['home_team_id'], match['start_time'])
        home_calculated = calculate_advanced_stats(home_stats)
        
        print("\nHome Team Advanced Stats:")
        print(f"File values:")
        print(f"Pass effectiveness: {match['home_pass_effectiveness']}")
        print(f"Shot accuracy: {match['home_shot_accuracy']}")
        print(f"Defensive success: {match['home_defensive_success']}")
        
        print(f"\nCalculated values:")
        print(f"Pass effectiveness: {home_calculated['pass_effectiveness']}")
        print(f"Shot accuracy: {home_calculated['shot_accuracy']}")
        print(f"Defensive success: {home_calculated['defensive_success']}")
        
        # Get away team stats
        away_stats = get_team_stats(conn, match['away_team_id'], match['start_time'])
        away_calculated = calculate_advanced_stats(away_stats)
        
        print("\nAway Team Advanced Stats:")
        print(f"File values:")
        print(f"Pass effectiveness: {match['away_pass_effectiveness']}")
        print(f"Shot accuracy: {match['away_shot_accuracy']}")
        print(f"Defensive success: {match['away_defensive_success']}")
        
        print(f"\nCalculated values:")
        print(f"Pass effectiveness: {away_calculated['pass_effectiveness']}")
        print(f"Shot accuracy: {away_calculated['shot_accuracy']}")
        print(f"Defensive success: {away_calculated['defensive_success']}")
        
        print("\n" + "="*50)

if __name__ == "__main__":
    main()