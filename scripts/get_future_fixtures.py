from config.config import headers
import requests
import json

def load_team_ids():
    """Load team IDs from JSON file"""
    with open('team_data/team_ids.json', 'r') as f:
        return json.load(f)

def get_scores_data(page=0):
    url = 'https://footballapi.pulselive.com/football/fixtures'
    params = {
        'comps': 1,
        'compSeasons': 719,
        'teams': '1,2,127,130,131,4,6,7,34,8,26,10,11,12,23,15,20,21,25,38',
        'page': page,
        'pageSize': 100,
        'sort': 'asc',
        'statuses': 'U,L',
        'altIds': True,
        'fast': False
    }
    response = requests.get(url, headers=headers, params=params)
    return response.json()

def format_fixtures_data(raw_data, team_ids):
    """Extract fixture data and format with team IDs"""
    formatted_fixture = {
        'fixture_id': raw_data['id'],
        'kickoff': {
            'timestamp': raw_data['kickoff']['millis'],
            'date': raw_data['kickoff']['label']
        },
        'home_team': {
            'id': team_ids[raw_data['teams'][0]['team']['name']],
            'name': raw_data['teams'][0]['team']['name']
        },
        'away_team': {
            'id': team_ids[raw_data['teams'][1]['team']['name']],
            'name': raw_data['teams'][1]['team']['name']
        }
    }
    
    return formatted_fixture

def save_fixtures_to_json(fixtures, filename='future_fixtures.json'):
    """Save list of fixtures to a JSON file"""
    with open(filename, 'w') as f:
        json.dump(fixtures, f, indent=4)
    print(f"Fixtures saved to {filename}")

# Load team IDs
team_ids = load_team_ids()

# Get and process the fixtures
response = get_scores_data()
list_of_fixtures = []
for i in response['content']:
    fixture_info = format_fixtures_data(i, team_ids)
    list_of_fixtures.append(fixture_info)

# Save to JSON file
save_fixtures_to_json(list_of_fixtures)