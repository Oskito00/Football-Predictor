import requests
import json
import time
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_team_stats(season_id, team_id, api_key):
    """Fetch statistics for a specific team"""
    url = f'https://api.sportradar.com/soccer/production/v4/en/seasons/sr:season:{season_id}/competitors/{team_id}/statistics'
    params = {'api_key': api_key}
    headers = {
        'Accept': 'application/json',
        'Content-Type': 'application/json'
    }
    
    # Print the full URL for debugging (without API key)
    print(f"Requesting URL: {url}")
    
    try:
        response = requests.get(url, params=params, headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error fetching stats for team {team_id}: {response.status_code}")
            print(f"Response content: {response.text}")  # Add this to see error details
            return None
    except Exception as e:
        print(f"Exception fetching stats for team {team_id}: {str(e)}")
        return None

def format_team_data(raw_data):
    """Format team and player statistics"""
    if not raw_data or 'competitor' not in raw_data:
        return None
    
    team_data = {
        'team_statistics': raw_data['competitor']['statistics'],
        'players': {}
    }
    
    # Process each player's statistics
    for player in raw_data['competitor'].get('players', []):
        player_id = player['id'].split(':')[-1]  # Extract numeric ID
        team_data['players'][player['name']] = {
            'id': player_id,
            'statistics': player['statistics']
        }
    
    return team_data

# Configuration
SEASON_ID = "118689"  # Current season
API_KEY = os.getenv('SPORTRADAR_API_KEY')

if not API_KEY:
    raise ValueError("SPORTRADAR_API_KEY not found in environment variables")

# Read the team IDs from the JSON file
with open('sportradar/team_ids.json', 'r', encoding='utf-8') as f:
    teams_data = json.load(f)

# Process each team
stats_data = {}
for team in teams_data['season_competitors']:
    team_id = team['id']  # Use the full ID instead of splitting it
    print(f"Fetching stats for {team['name']} (ID: {team_id})...")
    
    team_stats = get_team_stats(SEASON_ID, team_id, API_KEY)
    if team_stats:
        formatted_stats = format_team_data(team_stats)
        if formatted_stats:
            stats_data[team['name']] = formatted_stats
    
    # Add delay to avoid rate limiting
    time.sleep(1)

# Save the results
output_file = 'sportradar/team_and_player_stats.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(stats_data, f, indent=4, ensure_ascii=False)

print(f"\nTeam and player statistics saved to {output_file}")

# Print summary
print("\nStats Summary:")
for team_name, data in stats_data.items():
    team_stats = data['team_statistics']
    print(f"\n{team_name}:")
    print(f"  Matches played: {team_stats.get('matches_played', 0)}")
    print(f"  Goals scored: {team_stats.get('goals_scored', 0)}")
    print(f"  Goals conceded: {team_stats.get('goals_conceded', 0)}")
    print(f"  Number of players: {len(data['players'])}")
    
    # Print top 3 scorers
    players = [(name, stats['statistics'].get('goals_scored', 0)) 
              for name, stats in data['players'].items()]
    top_scorers = sorted(players, key=lambda x: x[1], reverse=True)[:3]
    
    print("  Top scorers:")
    for player_name, goals in top_scorers:
        print(f"    - {player_name}: {goals} goals")