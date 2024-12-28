Football Match Predictor using Sportradar API

New implementation is in the folder 'sportradar'

#How to run the scripts

1.
Extract top competitions from Sportradar API and save to a file.
This is done by the script 'competition_extraction/extract_competitions.py'

2.
Then the seasons for each top competition are extracted and saved to a file.
This is done by the script 'competition_extraction/extract_seasons.py'

3.
Once we have all the season Ids from all the top competitions we can scrape all the matches from these seasons and their relevant stats. (Note: This will scrape all matches for a season, if it is a current season future matches will also be scraped but they will be missing stats (because they have not been played yet))
This is done by the script 'data_scraping/scrape_matches_for_season.py'

4.
Once we have all the matches and their stats we can process the data and save it to an SQLite database.
This is done by the script 'data_migration/process_match_data.py'
Note: Create an SQLLite database and use the name of this as the input to the above script.

5.
Once we have the data in the database we can process it (extract recent form, match importance and other useful information about the teams in the match and save them to a CSV file. This is the training data you will use in your model)
This is done by the script 'data_processing/create_training_data.py'


#Requirements

Sportradar API key
Python 3.10 or higher
SQLite

#Contributors
Oscar Alberigo

