# Competition tiers (1-6, 1 being highest)
COMPETITION_TIERS = {
    # Tier 1 - Elite European Competition
    'UEFA Champions League': 1,
    
    # Tier 2 - Top 5 Domestic Leagues
    'Premier League': 2,
    'LaLiga': 2,
    'Bundesliga': 2,
    'Serie A': 2,
    'Ligue 1': 2,
    
    # Tier 3 - Secondary European Competitions
    'UEFA Europa League': 3,
    'FIFA Club World Cup': 3,
    
    # Tier 4 - Major Domestic Cups & Strong Leagues
    'FA Cup': 4,
    'Copa del Rey': 4,
    'DFB-Pokal': 4,
    'Coppa Italia': 4,
    'Coupe de France': 4,
    'Eredivisie': 4,
    'UEFA Europa Conference League': 4,
    
    # Tier 5 - Secondary Domestic Cups & Other Leagues
    'EFL Cup': 5,
    'Swiss Super League': 5,
    'Austrian Bundesliga': 5,
    'Danish Superliga': 5,
    'Norwegian Eliteserien': 5,
    'Swedish Allsvenskan': 5,
    
    # Tier 6 - Super Cups
    'UEFA Super Cup': 6,
    'Community Shield': 6,
    'Supercopa': 6,
    'Supercoppa Italiana': 6
}

# Derby matches by competition
DERBIES= {
    'Premier League': [
        ('Arsenal', 'Tottenham Hotspur'),  # North London Derby
        ('Liverpool', 'Everton'),  # Merseyside Derby
        ('Manchester United', 'Manchester City'),  # Manchester Derby
        ('Chelsea', 'Tottenham Hotspur'),  # London Derby
        ('Arsenal', 'Chelsea'),  # London Derby
    ],
    'LaLiga': [
        ('Real Madrid', 'Barcelona'),  # El Clásico
        ('Atletico Madrid', 'Real Madrid'),  # Madrid Derby
        ('Sevilla', 'Real Betis'),  # Seville Derby
        ('Athletic Bilbao', 'Real Sociedad'),  # Basque Derby
        ('Valencia', 'Villarreal'),  # Valencian Community Derby
    ],
    'Bundesliga': [
        ('Borussia Dortmund', 'Schalke 04'),  # Revierderby
        ('Bayern Munich', 'Borussia Dortmund'),  # Der Klassiker
        ('Hamburger SV', 'Werder Bremen'),  # Nordderby
        ('Bayern Munich', '1860 Munich'),  # Munich Derby (historical)
    ],
    'Serie A': [
        ('Inter Milan', 'AC Milan'),  # Derby della Madonnina
        ('Roma', 'Lazio'),  # Derby della Capitale
        ('Juventus', 'Torino'),  # Derby della Mole
        ('Napoli', 'Roma'),  # Derby del Sole
        ('Genoa', 'Sampdoria'),  # Derby della Lanterna
    ],
    'Ligue 1': [
        ('Paris Saint-Germain', 'Marseille'),  # Le Classique
        ('Lyon', 'Saint-Etienne'),  # Derby Rhône-Alpes
        ('Nice', 'Monaco'),  # Côte d'Azur Derby
    ],
    'Eredivisie': [
        ('Ajax', 'Feyenoord'),  # De Klassieker
        ('PSV Eindhoven', 'Ajax'),  # Dutch Derby
        ('Feyenoord', 'Sparta Rotterdam'),  # Rotterdam Derby
    ],
    'Swiss Super League': [
        ('FC Basel', 'FC Zürich'),  # Swiss Classic
        ('FC Zürich', 'Grasshopper Club Zürich'),  # Zurich Derby
        ('Young Boys', 'FC Basel'),  # Key Rivalry
    ],
    'Austrian Bundesliga': [
        ('Rapid Wien', 'Austria Wien'),  # Vienna Derby
        ('RB Salzburg', 'Rapid Wien'),  # Top Clash
    ],
    'Danish Superliga': [
        ('FC Copenhagen', 'Brøndby IF'),  # Copenhagen Derby
    ],
    'Norwegian Eliteserien': [
        ('Rosenborg', 'Molde'),  # Norwegian Classic
    ],
    'Swedish Allsvenskan': [
        ('AIK', 'Djurgården'),  # Stockholm Derby
        ('Malmö FF', 'Helsingborg'),  # Skåne Derby
    ],
    'UEFA Champions League': [],
    'UEFA Europa League': [],
    'UEFA Conference League': [],
    'UEFA Super Cup': [],
    'FIFA Club World Cup': [],
    'FA Cup': [],
    'EFL Cup': [],
    'Community Shield': [],
    'Copa del Rey': [],
    'Supercopa': [],
    'Coppa Italia': [],
    'Supercoppa Italiana': [],
    'Coupe de France': [],
}

# Default values
DEFAULT_METRICS = {
    'matches_played': 0,
    'average_goals_scored': 0,
    'average_goals_conceded': 0,
    'average_win_rate': 0,
    'average_clean_sheets': 0,
    'pass_effectiveness': 0,
    'shot_accuracy': 0,
    'conversion_rate': 0,
    'defensive_success': 0,
    'has_advanced_stats': False
}
# Previous matches query with team stats
PREVIOUS_MATCHES_QUERY = """
    WITH TeamMatches AS (
        SELECT 
            m.match_id,
            m.start_time,
            m.competition_name,
            m.home_team_name,
            m.away_team_name,
            m.home_score,
            m.away_score,
            CASE 
                WHEN m.home_team_name = ? THEN 'home'
                ELSE 'away'
            END as team_position
        FROM matches m
        WHERE (m.home_team_name = ? OR m.away_team_name = ?)
            AND m.start_time < ?
            AND m.match_status = 'ended'
        ORDER BY m.start_time DESC
        LIMIT ?
    )
    SELECT 
        tm.*,
        ts.ball_possession,
        ts.passes_successful,
        ts.passes_total,
        ts.shots_total,
        ts.shots_on_target,
        ts.chances_created,
        ts.tackles_successful,
        ts.tackles_total,
        ts.shots_saved
    FROM TeamMatches tm
    LEFT JOIN team_stats ts ON tm.match_id = ts.match_id 
        AND ((tm.team_position = 'home' AND ts.qualifier = 'home')
         OR (tm.team_position = 'away' AND ts.qualifier = 'away'))
    """

# Debug and normal matches queries
DEBUG_ENDED_MATCHES_QUERY = """
WITH numbered_matches AS (
    SELECT 
        match_id as fixture_id,
        start_time,
        competition_name,
        home_team_name as home_team,
        away_team_name as away_team,
        home_score as home_goals,
        away_score as away_goals,
        ROW_NUMBER() OVER (ORDER BY start_time) as row_num
    FROM matches
    WHERE match_status = 'ended'
)
SELECT 
    fixture_id,
    start_time,
    competition_name,
    home_team,
    away_team,
    home_goals,
    away_goals
FROM numbered_matches
WHERE row_num % 100 = 0
ORDER BY start_time
"""

ENDED_MATCHES_QUERY = """
SELECT 
    match_id as fixture_id,
    start_time,
    competition_name,
    home_team_name as home_team,
    away_team_name as away_team,
    home_score as home_goals,
    away_score as away_goals
FROM matches
WHERE match_status = 'ended'
ORDER BY start_time
"""

ALL_MATCHES_INFO_QUERY = """
SELECT 
    match_id as fixture_id,
    start_time,
    competition_name,
    home_team_name as home_team,
    away_team_name as away_team,
    home_score as home_goals,
    away_score as away_goals
FROM matches
ORDER BY start_time
"""

# Query to check team stats table summary
STATS_CHECK_QUERY = """
SELECT COUNT(*) as count, 
       COUNT(DISTINCT match_id) as unique_matches,
       COUNT(DISTINCT team_name) as unique_teams
FROM team_stats
"""


