import pandas as pd
from constants import PREVIOUS_MATCHES_QUERY

def get_previous_matches(conn, team_name, before_date, limit):
    """Get previous matches for a team before a specific date"""
    print(f"\nGetting previous matches for {team_name} before {before_date}")
    
    params = [team_name, team_name, team_name, before_date, limit]
    result = pd.read_sql_query(PREVIOUS_MATCHES_QUERY, conn, params=params)
    
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