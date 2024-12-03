import json
import requests
from config.config import headers


def test_h2h_request():
    # Load team IDs
    with open('team_ids.json', 'r') as f:
        teams = json.load(f)
    
    # Get first two teams for testing
    team_names = list(teams.keys())[:2]
    team1_id = teams[team_names[0]]
    team2_id = teams[team_names[1]]
    
    # Construct URL
    base_url = "https://footballapi.pulselive.com/football/stats/headtohead"
    params = {
        'teams': f"{team1_id},{team2_id}",
        'altIds': 'true',
        'comps': '1'
    }
    
    # Headers for the request

    
    print(f"Testing H2H request for:")
    print(f"{team_names[0]} (ID: {team1_id}) vs {team_names[1]} (ID: {team2_id})")
    
    # Make request
    try:
        response = requests.get(base_url, headers=headers, params=params)
        response.raise_for_status()
        
        # Save response to file for inspection
        with open('test_h2h_response.json', 'w') as f:
            json.dump(response.json(), f, indent=4)
            
        print("\nResponse saved to test_h2h_response.json")
        print("First few keys in response:", list(response.json().keys())[:5])
        
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")

if __name__ == "__main__":
    test_h2h_request() 