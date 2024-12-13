import json
import statistics

def convert_market_value_to_float(value):
    """Convert market value string to float in millions"""
    if not value or value == "-":
        return 0.0
    
    # Remove the '€' symbol and 'm' or 'k' suffix
    value = value.replace('€', '').strip()
    
    if value.endswith('m'):
        return float(value[:-1])
    elif value.endswith('k'):
        return float(value[:-1]) / 1000
    else:
        return float(value)

def calculate_team_importance(players):
    """Calculate importance scores for players based on market value"""
    # Convert market values to floats
    market_values = []
    for player in players:
        value = convert_market_value_to_float(player.get('marketValue', '0'))
        market_values.append(value)
        player['market_value_float'] = value
    
    # Calculate median market value
    median_value = statistics.median(market_values)
    
    # Calculate importance scores (difference from median)
    for player in players:
        player['importance_score'] = player['market_value_float'] - median_value
    
    # Sort players by importance score
    sorted_players = sorted(players, key=lambda x: x['importance_score'], reverse=True)
    
    return sorted_players

def format_player_output(player):
    """Format player data for output"""
    return {
        'name': player.get('name', 'Unknown'),
        'position': player.get('position', 'Unknown'),
        'market_value': player.get('marketValue', '€0'),
        'importance_score': round(player.get('importance_score', 0), 2)
    }

# Read the team player data
with open('transfrmarket/team_player_ids.json', 'r', encoding='utf-8') as f:
    teams_data = json.load(f)

# Process each team
importance_data = {}
for team_name, team_data in teams_data.items():
    players = team_data.get('players', [])
    sorted_players = calculate_team_importance(players)
    
    # Format output for each team
    importance_data[team_name] = [format_player_output(player) for player in sorted_players]

# Save results
with open('transfrmarket/player_importance.json', 'w', encoding='utf-8') as f:
    json.dump(importance_data, f, indent=4, ensure_ascii=False)

# Print some example results
for team_name, players in importance_data.items():
    print(f"\n{team_name}:")
    print("Top 3 most important players:")
    for player in players[:3]:
        print(f"- {player['name']}: {player['importance_score']}M € above median")
    print("Bottom 3 least important players:")
    for player in players[-3:]:
        print(f"- {player['name']}: {player['importance_score']}M € below median")
