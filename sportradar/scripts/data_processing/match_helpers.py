import pandas as pd
from constants import PREVIOUS_MATCHES_QUERY

def get_previous_matches(conn, team_name, before_date, limit):
    """Get previous matches for a team before a specific date"""
    params = [team_name, team_name, team_name, before_date, limit]
    result = pd.read_sql_query(PREVIOUS_MATCHES_QUERY, conn, params=params)
    return result