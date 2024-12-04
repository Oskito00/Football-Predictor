import json
import os

def extract_team_ids(year):
    input_file = f'data/{year}/premier_league_results.json'
    output_dir = f'data/{year}'
    output_file = f'{output_dir}/team_ids.json'
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Read input file
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    teams = {}
    
    for match in data['fixtures']:  # Changed from data['data'] to data['fixtures']
        home_team = match['home_team']
        away_team = match['away_team']
        
        # Add home team (using name as key and ID as value)
        if home_team['name'] not in teams:
            teams[home_team['name']] = int(home_team['id'])
        
        # Add away team (using name as key and ID as value)
        if away_team['name'] not in teams:
            teams[away_team['name']] = int(away_team['id'])
    
    # Sort teams by name
    teams = dict(sorted(teams.items()))
    
    # Save to JSON
    with open(output_file, 'w') as f:
        json.dump(teams, f, indent=4)
    
    print(f"Year {year}: Extracted {len(teams)} teams and saved to {output_file}")

def process_all_years(start_year=2024, end_year=2014):
    for year in range(start_year, end_year-1, -1):
        try:
            extract_team_ids(str(year))
        except FileNotFoundError:
            print(f"No data file found for year {year}, skipping...")
        except Exception as e:
            print(f"Error processing year {year}: {str(e)}")

if __name__ == "__main__":
    process_all_years()
