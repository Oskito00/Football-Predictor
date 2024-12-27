import json
import requests
import time
import os
from itertools import combinations
from config.config import headers


def get_h2h_stats(team1_id, team2_id, headers):
    """Get head-to-head stats for two teams"""
    base_url = "https://footballapi.pulselive.com/football/stats/headtohead"
    params = {
        'teams': f"{team1_id},{team2_id}",
        'altIds': 'true',
        'comps': '1'
    }
    
    try:
        response = requests.get(base_url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Extract relevant stats for both teams
        stats = {}
        for team_id, team_stats in data['stats'].items():
            team_data = {
                'wins': 0,
                'draws': 0,
                'losses': 0,
                'goals': 0,
                'clean_sheets': 0
            }
            
            for stat in team_stats:
                if stat['name'] == 'wins':
                    team_data['wins'] = int(stat['value'])
                elif stat['name'] == 'draws':
                    team_data['draws'] = int(stat['value'])
                elif stat['name'] == 'losses':
                    team_data['losses'] = int(stat['value'])
                elif stat['name'] == 'goals':
                    team_data['goals'] = int(stat['value'])
                elif stat['name'] == 'clean_sheet':
                    team_data['clean_sheets'] = int(stat['value'])
            
            stats[team_id] = team_data
        
        return stats
    
    except requests.exceptions.RequestException as e:
        print(f"Error getting H2H stats for teams {team1_id} vs {team2_id}: {e}")
        return None

def get_all_h2h_stats(year):
    # Load team IDs for current year
    input_file = f'data/{year}/team_ids.json'
    output_file = f'data/{year}/all_h2h_stats.json'
    
    # Try to load previous year's H2H stats if they exist
    prev_year = str(int(year) + 1)
    prev_year_file = f'data/{prev_year}/all_h2h_stats.json'
    previous_h2h_stats = {}
    
    if os.path.exists(prev_year_file):
        with open(prev_year_file, 'r') as f:
            previous_h2h_stats = json.load(f)
    
    with open(input_file, 'r') as f:
        teams = json.load(f)
    
    # Store all H2H stats
    all_h2h_stats = {}
    
    # Get all team combinations
    team_pairs = list(combinations(teams.items(), 2))
    total_pairs = len(team_pairs)
    
    print(f"Year {year}: Getting H2H stats for {total_pairs} team combinations...")
    
    for i, ((team1_name, team1_id), (team2_name, team2_id)) in enumerate(team_pairs, 1):
        match_key = f"{team1_name}_vs_{team2_name}"
        
        # Check if this H2H exists in previous year's data
        if match_key in previous_h2h_stats:
            print(f"Found existing H2H data for {match_key} from {prev_year}")
            all_h2h_stats[match_key] = previous_h2h_stats[match_key]
            continue
        
        print(f"Processing {i}/{total_pairs}: {team1_name} vs {team2_name}")
        
        # Get H2H stats
        stats = get_h2h_stats(team1_id, team2_id, headers)
        
        if stats:
            # Store stats with team names as keys
            all_h2h_stats[match_key] = {
                team1_name: stats.get(str(team1_id), {}),
                team2_name: stats.get(str(team2_id), {})
            }
        
        # Add delay to avoid rate limiting
        time.sleep(1)
    
    # Save all stats to file
    with open(output_file, 'w') as f:
        json.dump(all_h2h_stats, f, indent=4)
    
    print(f"Year {year}: All H2H stats saved to {output_file}")

def process_all_years(start_year=2022, end_year=2014):
    for year in range(start_year, end_year-1, -1):
        try:
            get_all_h2h_stats(str(year))
        except FileNotFoundError:
            print(f"No team IDs file found for year {year}, skipping...")
        except Exception as e:
            print(f"Error processing year {year}: {str(e)}")

if __name__ == "__main__":
    process_all_years() 