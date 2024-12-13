import requests
import json
import time
from datetime import datetime

def get_player_injuries(player_id):
    """Fetch injuries for a specific player"""
    url = f'https://transfermarkt-api.fly.dev/players/{player_id}/injuries'
    params = {'page_number': 1}
    headers = {'accept': 'application/json'}
    
    try:
        response = requests.get(url, params=params, headers=headers)
        if response.status_code == 200:
            return response.json().get('injuries', [])
        else:
            print(f"Error fetching injuries for player {player_id}: {response.status_code}")
            return []
    except Exception as e:
        print(f"Exception fetching injuries for player {player_id}: {str(e)}")
        return []

def format_injury_data(injuries):
    """Format injury data to include all injuries from the 24/25 season"""
    current_season = "24/25"
    formatted_injuries = []
    
    for injury in injuries:
        if injury.get('season') == current_season:
            # Handle all fields with .get() to provide defaults
            formatted_injury = {
                'injury': injury.get('injury', 'Unknown injury'),
                'from': injury.get('from', 'Unknown'),
                'until': injury.get('until', 'Unknown'),
                'days': injury.get('days', 'Unknown')
            }
            
            # Override days if until date is missing or unknown
            if formatted_injury['until'] in ['Unknown', '-', None]:
                formatted_injury['days'] = 'Still injured'
            
            formatted_injuries.append(formatted_injury)
    
    return formatted_injuries

# Read the team and player data
with open('transfrmarket/team_player_ids.json', 'r', encoding='utf-8') as f:
    teams_data = json.load(f)

# Process each team and player
injury_data = {}
for team_name, team_data in teams_data.items():
    print(f"Processing team: {team_name}")
    injury_data[team_name] = []
    
    for player in team_data.get('players', []):
        player_id = player.get('id')
        if not player_id:
            continue
            
        print(f"Fetching injuries for {player.get('name', 'Unknown')} ({player_id})")
        
        injuries = get_player_injuries(player_id)
        season_injuries = format_injury_data(injuries)
        if season_injuries:  # If player had any injuries in 24/25 season
            injury_data[team_name].append({
                'player_id': player_id,
                'name': player.get('name'),
                'injuries': season_injuries
            })
        
        # Add delay to avoid rate limiting
        time.sleep(1)
    
    # Remove teams with no injuries
    if not injury_data[team_name]:
        del injury_data[team_name]

# Save the results
output_file = 'transfrmarket/player_injuries.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(injury_data, f, indent=4, ensure_ascii=False)

print(f"\nInjury data saved to {output_file}")

# Print summary
print("\nSeason 24/25 Injuries Summary:")
for team, players in injury_data.items():
    print(f"\n{team}:")
    for player in players:
        print(f"  {player['name']}:")
        for injury in player['injuries']:
            print(f"    - {injury['injury']} (From: {injury['from']}, Until: {injury['until']})")