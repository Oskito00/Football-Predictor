import requests
from dotenv import load_dotenv
import os
import json


# Load environment variables
load_dotenv()

# Get API key from environment variables
api_key = os.getenv('SPORTRADAR_API_KEY')

# Define the endpoint (note the .json extension)
url = 'https://api.sportradar.com/soccer/production/v4/en/seasons/sr:season:118689/missing_players.json'

# Set up the parameters for the request
params = {
    'api_key': api_key,
}

# Set up the headers
headers = {
    'Accept': 'application/json'
}

#Helper functions
def process_injuries(data):
    teams_injuries = {}
    for team in data['competitors']:
        team_name = team['name']
        teams_injuries[team_name] = []
        for player in team.get('players', []):
            teams_injuries[team_name].append({
                'Player Name': player['name'],
                'Reason': player['reason'],
                'Start Date': player['start_date']
            })
    return teams_injuries

# Make the GET request with headers
response = requests.get(url, params=params, headers=headers)

# Check if the request was successful
if response.status_code == 200:
    data = response.json()
    processed_data = process_injuries(data)
    # Save processed data to JSON file
    with open('sportradar/injuries_suspensions.json', 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=4, ensure_ascii=False)
    
    print("Injuries and suspensions data saved successfully")
    
else:
    print(f"Error {response.status_code}: {response.text}")
    print(f"Full URL: {response.url}")