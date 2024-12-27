import json
import os
import requests
from config.config import headers

#Helper function to get match stats
#--------------------------------
def get_match_stats(match_id):
    """Fetch detailed statistics for a specific match"""
    url = f'https://footballapi.pulselive.com/football/stats/match/{int(match_id)}'
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        if not response.content:  # Check if response is empty
            print(f"Empty response for match {match_id}")
            return None
            
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching stats for match {match_id}: {str(e)}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON for match {match_id}: {str(e)}")
        return None

def process_match_stats(stats_data, home_team_id):
    """Process raw stats data into organized home/away stats"""
    if not stats_data or 'data' not in stats_data:
        return None
        
    processed_stats = {'home': {}, 'away': {}}
    
    # Get the team IDs from the data
    team_ids = list(stats_data['data'].keys())
    
    for team_id in team_ids:
        # Determine if this is home or away team
        team_type = 'home' if int(team_id) == home_team_id else 'away'
        team_stats = processed_stats[team_type]
        
        # Process each stat for the team
        for stat in stats_data['data'][team_id]['M']:
            if stat.get('value') is not None:
                team_stats[stat['name']] = stat['value']
    
    return processed_stats

#Helper function to get fixtures
#--------------------------------
def get_scores_data(year, page=0):
    with open('scripts/data_scraping/request_params.json', 'r') as f:
        params_data = json.load(f)
    
    if year not in params_data:
        raise ValueError(f"No request parameters found for year {year}")
    
    params = {
        'comps': 1,
        'compSeasons': params_data[year]['compSeasons'],
        'teams': params_data[year]['teams'],
        'page': page,
        'pageSize': 100,
        'sort': 'desc',
        'statuses': 'A,C',
        'altIds': True,
        'fast': False
    }
    
    url = 'https://footballapi.pulselive.com/football/fixtures'
    response = requests.get(url, headers=headers, params=params)
    return response.json()

def process_fixtures_data(year):
    all_fixtures = []
    page = 0
    
    while True:
        response = get_scores_data(year, page)
        if not response['content']:  # If no more fixtures
            break
            
        for fixture in response['content']:
            # Extract home and away teams
            home_team = fixture['teams'][0]
            away_team = fixture['teams'][1]
            
            # Process goals
            goals = []
            for goal in fixture.get('goals', []):
                goals.append({
                    'scorer_id': goal.get('personId'),
                    'assist_id': goal.get('assistId'),
                    'minute': goal.get('clock', {}).get('label'),
                    'phase': goal.get('phase'),
                    'type': goal.get('type')
                })
            
            # Get match stats if the match is completed
            match_stats = None
            if fixture['status'] == 'C':  # Only get stats for completed matches
                match_stats_data = get_match_stats(fixture['id'])
                if match_stats_data:
                    match_stats = process_match_stats(match_stats_data, home_team['team']['id'])
            
            # Create fixture dictionary
            fixture_info = {
                'fixture_id': fixture['id'],
                'gameweek': fixture['gameweek']['gameweek'],
                'season': fixture['gameweek']['compSeason']['label'],
                'kickoff': {
                    'timestamp': fixture['kickoff']['millis'],
                    'date': fixture['kickoff']['label']
                },
                'home_team': {
                    'id': home_team['team']['id'],
                    'name': home_team['team']['name'],
                    'short_name': home_team['team']['shortName'],
                    'score': home_team['score']
                },
                'away_team': {
                    'id': away_team['team']['id'],
                    'name': away_team['team']['name'],
                    'short_name': away_team['team']['shortName'],
                    'score': away_team['score']
                },
                'venue': {
                    'name': fixture.get('ground', {}).get('name'),
                    'city': fixture.get('ground', {}).get('city')
                },
                'status': fixture['status'],
                'outcome': fixture.get('outcome'),
                'attendance': fixture.get('attendance'),
                'goals': goals,
                'stats': {
                    'final_score': f"{home_team['score']}-{away_team['score']}",
                    'winner': 'home' if fixture.get('outcome') == 'H' else 'away' if fixture.get('outcome') == 'A' else 'draw',
                    'match_stats': match_stats  # Add the detailed match stats here
                }
            }
            
            all_fixtures.append(fixture_info)
            print(f"Processed fixture {fixture['id']}")
        
        print(f"Processed page {page}")
        page += 1
    
    # Sort fixtures by gameweek
    all_fixtures.sort(key=lambda x: x['gameweek'])

    # Ensure the directory exists
    output_dir = f'data/{year}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to JSON file
    output_filename = f'data/{year}/premier_league_results.json'
    with open(output_filename, 'w') as f:
        json.dump({'fixtures': all_fixtures}, f, indent=4)
    
    return all_fixtures

if __name__ == "__main__":
    years = ['2024']  # Change this to the desired year

    for year in years:
        fixtures = process_fixtures_data(year)
        print(f"Processed {len(fixtures)} fixtures and saved to {year}_premier_league_fixtures.json")

    

