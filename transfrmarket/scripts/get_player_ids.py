import requests
import json
import time

def get_team_players(team_id):
    url = f'https://transfermarkt-api.fly.dev/clubs/{team_id}/players'
    headers = {
        'accept': 'application/json'
    }
    
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching team {team_id}: {response.status_code}")
        return None

# Read the team IDs from the JSON file
with open('transfrmarket/team_ids_2024.json', 'r') as f:
    teams_data = json.load(f)

team_players = {}

# Process each team
for club in teams_data['clubs']:
    team_id = club['id']
    team_name = club['name']
    print(f"Fetching players for {team_name}...")
    
    players_data = get_team_players(team_id)
    if players_data:
        # Debug the response structure
        print(f"Response type: {type(players_data)}")
        print(f"Response content: {players_data}")
        team_players[team_name] = players_data

    # Add a delay to avoid hitting rate limits
    time.sleep(1)

# Save the results
with open('transfrmarket/team_player_ids.json', 'w', encoding='utf-8') as f:
    json.dump(team_players, f, indent=4, ensure_ascii=False)

print("Player IDs saved successfully to team_player_ids.json")