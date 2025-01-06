import sqlite3
import pandas as pd

conn = sqlite3.connect('football_data.db')
query = """
SELECT * FROM player_stats 
WHERE date(start_time) = '2024-10-27'
ORDER BY datetime(start_time)
"""
df = pd.read_sql_query(query, conn)
df.to_csv('october27_games.csv', index=False)
conn.close()