import json
from collections import defaultdict
from datetime import datetime

class TeamAnalyzer:
    def __init__(self):
        self.fixtures = self._load_fixtures()
        self.players = self._load_players()
        
    def _load_fixtures(self):
        with open('premier_league_fixtures.json', 'r') as f:
            return json.load(f)['fixtures']
            
    def _load_players(self):
        with open('player_database.json', 'r') as f:
            return json.load(f)
    
    def analyze_team(self, team_id, current_gameweek):
        """Analyze team performance up to a specific gameweek"""
        team_stats = {
            'form': [],  # Last 5 results
            'goals_scored': 0,
            'goals_conceded': 0,
            'clean_sheets': 0,
            'top_scorers': defaultdict(int),
            'top_assisters': defaultdict(int),
            'results': [],
            'points': 0
        }
        
        # Analyze each fixture up to current gameweek
        for fixture in self.fixtures:
            if fixture['gameweek'] >= current_gameweek:
                continue
                
            home_team = fixture['home_team']
            away_team = fixture['away_team']
            
            if home_team['id'] == team_id or away_team['id'] == team_id:
                is_home = home_team['id'] == team_id
                team_score = home_team['score'] if is_home else away_team['score']
                opponent_score = away_team['score'] if is_home else home_team['score']
                
                # Update goals
                team_stats['goals_scored'] += team_score
                team_stats['goals_conceded'] += opponent_score
                
                # Update clean sheets
                if opponent_score == 0:
                    team_stats['clean_sheets'] += 1
                
                # Update form and points
                if fixture['outcome'] == 'D':
                    result = 'D'
                    team_stats['points'] += 1
                elif (fixture['outcome'] == 'H' and is_home) or (fixture['outcome'] == 'A' and not is_home):
                    result = 'W'
                    team_stats['points'] += 3
                else:
                    result = 'L'
                
                team_stats['form'].append(result)
                team_stats['results'].append({
                    'gameweek': fixture['gameweek'],
                    'opponent': away_team['name'] if is_home else home_team['name'],
                    'score': f"{team_score}-{opponent_score}",
                    'result': result
                })
                
                # Track scorers and assisters
                for goal in fixture['goals']:
                    if goal['scorer_id']:
                        scorer_id = str(int(goal['scorer_id']))
                        if self._is_player_in_team(scorer_id, team_id):
                            team_stats['top_scorers'][scorer_id] += 1
                    
                    if goal['assist_id']:
                        assister_id = str(int(goal['assist_id']))
                        if self._is_player_in_team(assister_id, team_id):
                            team_stats['top_assisters'][assister_id] += 1
        
        # Keep only last 5 for form
        team_stats['form'] = team_stats['form'][-5:]
        
        # Convert player IDs to names
        team_stats['top_scorers'] = self._convert_ids_to_names(team_stats['top_scorers'])
        team_stats['top_assisters'] = self._convert_ids_to_names(team_stats['top_assisters'])
        
        return team_stats
    
    def get_recent_form(self, team_id, gameweek, num_games=5):
        """Get team's form for X games before specified gameweek"""
        form_data = {
            'results': [],
            'goals_scored': 0,
            'goals_conceded': 0,
            'points': 0,
            'clean_sheets': 0
    }
    
        # Get relevant fixtures before this gameweek
        relevant_fixtures = []
        for fixture in self.fixtures:
            if fixture['gameweek'] < gameweek:
                if fixture['home_team']['id'] == team_id or fixture['away_team']['id'] == team_id:
                    relevant_fixtures.append(fixture)
                
        # Sort by gameweek descending and take last num_games
        relevant_fixtures.sort(key=lambda x: x['gameweek'], reverse=True)
        recent_fixtures = relevant_fixtures[:num_games]

        print(f"Found {len(recent_fixtures)} recent fixtures")  # Debug print

        for fixture in recent_fixtures:
            is_home = fixture['home_team']['id'] == team_id
            team_score = fixture['home_team']['score'] if is_home else fixture['away_team']['score']
            opponent_score = fixture['away_team']['score'] if is_home else fixture['home_team']['score']
            opponent_name = fixture['away_team']['name'] if is_home else fixture['home_team']['name']
            
            # Determine result
            if fixture['outcome'] == 'D':
                result = 'D'
                points = 1
            elif (fixture['outcome'] == 'H' and is_home) or (fixture['outcome'] == 'A' and not is_home):
                result = 'W'
                points = 3
            else:
                result = 'L'
                points = 0
            
            form_data['results'].append({
                'gameweek': fixture['gameweek'],
                'opponent': opponent_name,
                'score': f"{team_score}-{opponent_score}",
                'result': result,
                'points': points,
                'venue': 'H' if is_home else 'A'  # Add home/away indicator
        })
        
        form_data['goals_scored'] += team_score
        form_data['goals_conceded'] += opponent_score
        form_data['points'] += points
        if opponent_score == 0:
            form_data['clean_sheets'] += 1
    
        return form_data
    
    def _is_player_in_team(self, player_id, team_id):
        """Check if player belongs to team"""
        team_str = str(int(team_id))
        if team_str in self.players:
            return player_id in self.players[team_str]['players']
        return False
    
    def _convert_ids_to_names(self, id_dict):
        """Convert player IDs to names"""
        name_dict = {}
        for team in self.players.values():
            for player_id, player_data in team['players'].items():
                if player_id in id_dict:
                    name_dict[player_data['name']] = id_dict[player_id]
        return dict(sorted(name_dict.items(), key=lambda x: x[1], reverse=True))

# Usage example
if __name__ == "__main__":
    analyzer = TeamAnalyzer()
    
    # Get Arsenal's form for 5 games before gameweek 11
    arsenal_form = analyzer.get_recent_form(1, 12, 5)
    
    print("Arsenal's Last 5 Games:")
    for game in arsenal_form['results']:
        print(f"GW{game['gameweek']}: {game['venue']} vs {game['opponent']} - {game['score']} ({game['result']})")
    
    print(f"\nStats from last {len(arsenal_form['results'])} games:")
    print(f"Points: {arsenal_form['points']}")
    print(f"Goals Scored: {arsenal_form['goals_scored']}")
    print(f"Goals Conceded: {arsenal_form['goals_conceded']}")
    print(f"Clean Sheets: {arsenal_form['clean_sheets']}")
    print(f"Form: {' '.join([game['result'] for game in arsenal_form['results']])}")