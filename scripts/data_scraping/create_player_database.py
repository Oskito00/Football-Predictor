import json
from config.config import headers
import requests


#Helper functions
#--------------------------------

def get_clubs():
    """
    Fetches all Premier League clubs from the football API
    
    Returns:
        dict: JSON response containing club information
    """
    url = "https://footballapi.pulselive.com/football/teams"
    params = {
        'pageSize': 100,
        'comps': 1,
        'compSeasons': 719,  # Added season filter
        'altIds': True,
        'page': 0
    }
    response = requests.get(url, headers=headers, params=params)
    return response.json()

def get_club_players(club_id, page=1, page_size=10):  # Note: page starts at 0
    """
    Fetches players from a specific club using the staff endpoint
    
    Args:
        club_id (int): ID of the club
        page (int): Page number (default: 0)
        page_size (int): Number of players per page (default: 30)
    """
    url = f"https://footballapi.pulselive.com/football/teams/{club_id}/compseasons/719/staff"
    params = {
        'pageSize': page_size,
        'compSeasons': 719,
        'altIds': True,
        'page': page,
        'type': 'player'
    }
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        players_data = response.json()
        return players_data
        
    except requests.RequestException as e:
        print(f"Error fetching players data: {e}")
        return None
    
#Main functions
#--------------------------------

def create_club_player_database():
    database = {}
    clubs = get_clubs()
    
    for club in clubs['content']:
        club_id = int(club['club']['id'])  # Convert to integer
        club_name = club['club']['name']
        club_short_name = club['club'].get('shortName', '')
        
        # Initialize club entry
        database[club_id] = {
            'name': club_name,
            'shortName': club_short_name,
            'players': {}
        }
        
        # Get all players for the club
        players_data = get_club_players(club_id, page=0)  # Start at page 0
        
        if players_data and 'players' in players_data:
            for player in players_data['players']:
                player_id = player['id']
                database[club_id]['players'][player_id] = {
                    'name': player['name']['display'],
                    'position': player.get('info', {}).get('position', ''),
                    'shirtNum': player.get('info', {}).get('shirtNum', ''),
                    'nationality': player.get('nationalTeam', {}).get('country', ''),
                    'dateOfBirth': player.get('info', {}).get('dateOfBirth', ''),
                }
        
        print(f"Processed {club_name}")  # Progress indicator
        
    return database


if __name__ == "__main__":
    database = create_club_player_database()

    with open('player_database.json', 'w') as json_file:
        json.dump(database, json_file, indent=4)
    
    # Example of how to access and display the data
    
    for club_id, club_data in database.items():
        print(f"\nClub: {club_data['name']}")
        print(f"Number of players: {len(club_data['players'])}")
        print("Players:")
        for player_id, player_data in club_data['players'].items():
            print(f"- {player_data['name']} ({player_data['position']}) #{player_data['shirtNum']}")

