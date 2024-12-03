# Football Match Predictor

A machine learning system that predicts Premier League football match outcomes using historical data and team statistics.

## Overview

This project uses Ridge Regression (L2 regularization) to predict football match scores based on various features including:
- Team performance metrics
- Head-to-head statistics
- Recent form and momentum
- Match-specific statistics

## Features

### Data Collection
- Fetches fixture data from the Football API
- Collects detailed match statistics
- Tracks team performance metrics
- Records head-to-head statistics

### Data Processing
- Calculates team momentum based on recent form
- Processes historical match data
- Generates feature sets for training
- Handles missing data and edge cases

### Prediction System
- Predicts match scores using Ridge Regression
- Evaluates predictions for:
  - Match results (Win/Draw/Loss)
  - Exact scores
  - Over/Under 2.5 goals
- Provides detailed prediction analysis

## Usage

1. Set up API credentials in `config/config.py`
2. Collect fixture data:

```python scripts/get_fixtures.py```

3. Prepare training data:

```python scripts/prepare_training_data.py```

