import json

def load_json_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json_file(data, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def normalize_name(name):
    """Standardize name format for comparison"""
    name = name.lower().strip()
    # Remove common prefixes/suffixes
    name = name.replace(" jr", "").replace(" sr", "")
    # Handle special characters
    name = name.replace("é", "e").replace("í", "i").replace("á", "a").replace("ó", "o").replace("ú", "u")
    name = name.replace("ñ", "n").replace("ç", "c")
    # Remove punctuation
    name = name.replace("-", " ").replace(".", " ")
    # Handle spaces and commas
    name = name.replace(", ", " ").replace(",", " ")
    # Remove multiple spaces
    return ' '.join(name.split())

def find_best_match(player_name, players_list):
    """Find the best matching player in the list"""
    normalized_name = normalize_name(player_name)
    best_match = None
    best_score = 0
    
    # Try exact match first
    for player in players_list:
        if normalize_name(player['name']) == normalized_name:
            return player
    
    # If no exact match, try partial matches
    for player in players_list:
        player_normalized = normalize_name(player['name'])
        
        # Check if names share words
        name_parts1 = set(normalized_name.split())
        name_parts2 = set(player_normalized.split())
        common_parts = name_parts1.intersection(name_parts2)
        
        if len(common_parts) >= 1:  # If they share at least one word
            score = len(common_parts) / max(len(name_parts1), len(name_parts2))
            if score > best_score:
                best_score = score
                best_match = player
    
    # Return match only if it's a good match
    if best_score >= 0.5:  # At least 50% of words match
        return best_match
    return None

def combine_player_scores():
    print("Loading market data...")
    market_data = load_json_file('transfrmarket/player_importance.json')
    print(f"Loaded {len(market_data)} teams from market data")
    
    print("\nLoading stats data...")
    stats_raw = load_json_file('sportradar/player_importance_stats.json')
    stats_data = stats_raw['teams']
    print(f"Loaded {len(stats_data)} teams from stats data")

    results = {}
    total_matches = 0
    
    for team_name, market_players in market_data.items():
        print(f"\nProcessing {team_name}:")
        print(f"Market players: {len(market_players)}")
        
        if team_name not in stats_data:
            print(f"No stats data found for {team_name}")
            continue
            
        stats_players = stats_data[team_name]
        print(f"Stats players: {len(stats_players)}")
        matched_players = []
        
        for market_player in market_players:
            best_match = find_best_match(market_player['name'], stats_players)
            if best_match:
                matched_players.append({
                    'name': market_player['name'],
                    'position': market_player['position'],
                    'market_importance': market_player['importance_score'],
                    'stats_importance': best_match['score'],
                    'combined_importance': (market_player['importance_score'] + best_match['score']) / 2
                })
                print(f"Matched: {market_player['name']} -> {best_match['name']}")
            else:
                print(f"No match found for: {market_player['name']}")
        
        if matched_players:
            matched_players.sort(key=lambda x: x['combined_importance'], reverse=True)
            results[team_name] = matched_players
            total_matches += len(matched_players)
    
    print(f"\nTotal players matched: {total_matches}")
    
    # Save results to JSON file
    output_path = 'data/combined_player_importance.json'
    save_json_file(results, output_path)
    print(f"Saved combined results to {output_path}")

if __name__ == "__main__":
    combine_player_scores()