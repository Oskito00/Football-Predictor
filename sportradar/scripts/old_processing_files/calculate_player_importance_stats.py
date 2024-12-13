import json
import datetime
import unicodedata

def load_json_file(filepath):
    """Load and return JSON data from a file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def normalize_name(name):
    """Convert all name formats to a standardized format for comparison"""
    # Convert to lowercase and strip spaces
    name = name.lower().strip()
    
    # Remove accents
    name = ''.join(c for c in unicodedata.normalize('NFKD', name)
                  if not unicodedata.combining(c))
    
    # Handle "Last, First" format
    if ',' in name:
        last_name, first_name = name.split(',')
        name = f"{first_name.strip()} {last_name.strip()}"
    
    # Remove any special characters and extra spaces
    name = ' '.join(name.split())
    name = ''.join(c for c in name if c.isalnum() or c.isspace())
    
    # Handle special cases where players are known by single names
    single_name_players = {
        'ederson', 'alisson', 'rodri', 'casemiro', 'antony', 'fred', 'fabinho',
        'jorginho', 'richarlison', 'fernandinho', 'firmino', 'neto', 'beto',
        'andre', 'igor', 'chiquinho', 'buonanotte', 'danilo', 'podence'
    }
    
    # Handle reversed names (where first name is actually last name)
    reversed_names = {
        'alcaraz carlos': 'carlos alcaraz',
        'borges carlos': 'carlos borges',
        'norgaard christian': 'christian norgaard',
        'jebbison daniel': 'daniel jebbison',
        'oliveira danilo': 'danilo oliveira',
        'maghoma edmondparis': 'edmondparis maghoma',
        'gilmour billy': 'billy gilmour',
        'irving andy': 'andy irving'
    }
    
    # Handle hyphenated/compound names
    name = name.replace('-', ' ')
    name = name.replace('  ', ' ')
    
    # Check for reversed names first
    if name in reversed_names:
        name = reversed_names[name]
    
    # Then check for single names
    if name.split()[0] in single_name_players:
        return name.split()[0]
    
    return name

def format_name(name):
    """Convert name to 'FirstName LastName' format"""
    # Remove any commas and extra spaces
    name = name.replace(",", "").strip()
    parts = name.split()
    if len(parts) >= 2:
        # Capitalize each part of the name
        return ' '.join(part.capitalize() for part in parts)
    return name.capitalize()

def calculate_defender_score(stats):
    """Calculate importance score for defenders"""
    return (3 * stats.get('shots_blocked', 0) +
            1 * stats.get('matches_played', 0) +
            5 * stats.get('goals_scored', 0) +
            4 * stats.get('assists', 0) -
            1 * stats.get('goals_conceded', 0) -
            3 * stats.get('yellow_cards', 0) -
            5 * stats.get('red_cards', 0))

def calculate_forward_score(stats):
    """Calculate importance score for forwards"""
    return (10 * stats.get('goals_scored', 0) +
            7 * stats.get('assists', 0) +
            2 * stats.get('shots_on_target', 0) -
            1 * stats.get('shots_off_target', 0) -
            0.5 * stats.get('offsides', 0) -
            3 * stats.get('yellow_cards', 0) -
            5 * stats.get('red_cards', 0))

def calculate_midfielder_score(stats):
    """Calculate importance score for midfielders"""
    return (8 * stats.get('assists', 0) +
            1.5 * stats.get('shots_on_target', 0) +
            2 * stats.get('shots_blocked', 0) +
            0.5 * stats.get('matches_played', 0) -
            0.5 * stats.get('goals_conceded', 0) -
            1 * stats.get('shots_off_target', 0) -
            0.5 * stats.get('offsides', 0) -
            3 * stats.get('yellow_cards', 0) -
            5 * stats.get('red_cards', 0))

def calculate_goalkeeper_score(stats):
    """Calculate importance score for goalkeepers"""
    return (3 * stats.get('shots_blocked', 0) +
            1 * stats.get('matches_played', 0) -
            2 * stats.get('goals_conceded', 0) -
            3 * stats.get('yellow_cards', 0) -
            5 * stats.get('red_cards', 0))

def calculate_importance_score(position, stats):
    """Calculate importance score based on position"""
    if position.startswith('D'):
        return calculate_defender_score(stats)
    elif position.startswith('F'):
        return calculate_forward_score(stats)
    elif position.startswith('M'):
        return calculate_midfielder_score(stats)
    elif position.startswith('G'):
        return calculate_goalkeeper_score(stats)
    return 0

def check_player_counts(team_stats, player_database, player_positions):
    """Compare player counts between files and check matching success"""
    print("\nPLAYER COUNT ANALYSIS")
    print("=" * 50)
    
    # Count players in team_stats (players with actual game stats)
    stats_total = 0
    stats_players = set()
    for team_name, team_data in team_stats.items():
        team_count = len(team_data['players'])
        stats_total += team_count
        for player_name in team_data['players'].keys():
            normalized_name = normalize_name(player_name)
            stats_players.add(normalized_name)
    
    # Count players in player_database (all registered players)
    db_total = 0
    db_players = set()
    for team_id, team_data in player_database.items():
        if isinstance(team_data, dict) and 'players' in team_data:
            team_count = len(team_data['players'])
            db_total += team_count
            for player_data in team_data['players'].values():
                if 'name' in player_data:
                    db_players.add(normalize_name(player_data['name']))
    
    # Calculate matches
    matched_players = stats_players.intersection(set(player_positions.keys()))
    
    print(f"\nRegistered players (database): {db_total}")
    print(f"Players with stats: {stats_total}")
    print(f"Successfully matched: {len(matched_players)}")
    print(f"Match rate of players with stats: {(len(matched_players)/stats_total)*100:.1f}%")
    
    # Show unmatched players who have stats
    unmatched = stats_players - set(player_positions.keys())
    if unmatched:
        print("\nPlayers with stats but no position data:")
        for name in sorted(list(unmatched))[:10]:
            print(f"- {name}")

def main():
    # Load the JSON files
    team_stats = load_json_file('sportradar/team_and_player_stats.json')
    player_database = load_json_file('data/player_database.json')
    
    # Create a dictionary of player positions
    player_positions = {}
    for team_id, team_data in player_database.items():
        if isinstance(team_data, dict) and 'players' in team_data:
            for player_id, player_data in team_data['players'].items():
                if isinstance(player_data, dict) and 'name' in player_data and 'position' in player_data:
                    full_name = player_data['name']
                    normalized_name = normalize_name(full_name)
                    player_positions[normalized_name] = player_data['position']
    
    # Check player counts before processing
    check_player_counts(team_stats, player_database, player_positions)
    
    # Process each team and their players
    results_by_team = {}
    processed_players = set()
    
    for team_name, team_data in team_stats.items():
        print(f"\nProcessing {team_name}...")
        team_results = []
        
        for player_name, player_data in team_data['players'].items():
            normalized_name = normalize_name(player_name)
            print(f"Looking for: {normalized_name}")
            position = player_positions.get(normalized_name)
            
            if position:
                if normalized_name not in processed_players:
                    score = calculate_importance_score(position, player_data['statistics'])
                    team_results.append({
                        'name': format_name(player_name),
                        'position': position,
                        'score': score,
                        'stats': player_data['statistics']
                    })
                    processed_players.add(normalized_name)
                else:
                    print(f"Skipping duplicate player: {player_name}")
            else:
                print(f"Position not found for player: {player_name}")
        
        # Sort team's players by score
        team_results.sort(key=lambda x: x['score'], reverse=True)
        results_by_team[team_name] = team_results
    
    # Sort teams alphabetically
    sorted_teams = dict(sorted(results_by_team.items()))
    
    # Write results to file
    output_file = 'sportradar/player_importance_stats.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.datetime.now().isoformat(),
            'total_players_analyzed': len(processed_players),
            'teams': sorted_teams
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\nProcessed {len(processed_players)} unique players")
    print(f"Written results for {len(sorted_teams)} teams to {output_file}")
    
    # Print top 3 players from each team
    print("\nTop 3 Players by Team:")
    print("=" * 80)
    for team_name, team_results in sorted_teams.items():
        print(f"\n{team_name}:")
        for player in team_results[:3]:
            print(f"  {player['name']} ({player['position']}): {player['score']:.1f}")

if __name__ == "__main__":
    main()