import json

def extract_team_ids(input_file, output_file):
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    teams = {}
    
    for match in data['data']:
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
    
    print(f"Extracted {len(teams)} teams and saved to {output_file}")

# Usage
extract_team_ids('processed_training_data.json', 'team_ids.json')
