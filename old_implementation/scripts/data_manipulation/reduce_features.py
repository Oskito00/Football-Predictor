import pandas as pd

# Load the CSV file
df = pd.read_csv('LinearRegression/data/2014/training_data.csv', sep=',', skiprows=1)

# Combine home performance stats
df['home_performance_stats'] = df['home_avg_possession'] + df['home_avg_shots_on_target'] + df['home_avg_pass_accuracy']

# Combine away performance stats
df['away_performance_stats'] = df['away_avg_possession'] + df['away_avg_shots_on_target'] + df['away_avg_pass_accuracy']

# Combine home defense stats
df['home_defense_stats'] = df['home_avg_tackles'] + df['home_avg_saves']

# Combine away defense stats
df['away_defense_stats'] = df['away_avg_tackles'] + df['away_avg_saves']

# Calculate home clinical score
df['home_clinical_score'] = df['home_avg_big_chances'] - df['home_goals_scored']

# Calculate away clinical score
df['away_clinical_score'] = df['away_avg_big_chances'] - df['away_goals_scored']

# Drop the original columns that were combined
df.drop(columns=[
    'home_avg_possession', 'home_avg_shots_on_target', 'home_avg_pass_accuracy',
    'away_avg_possession', 'away_avg_shots_on_target', 'away_avg_pass_accuracy',
    'home_avg_tackles', 'home_avg_saves', 'away_avg_tackles', 'away_avg_saves',
    'home_avg_big_chances', 'away_avg_big_chances'
], inplace=True)

# Save the modified DataFrame to a new CSV file
df.to_csv('LinearRegression/data/2014/reduced_training_data.csv', index=False)