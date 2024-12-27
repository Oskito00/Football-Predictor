import pandas as pd
import numpy as np
import os

def load_weighted_training_data(start_year=2014, end_year=2023, base_weight=0.5):
    """
    Load and combine training data from multiple years with exponential weighting
    Args:
        start_year: First year to include
        end_year: Last year to include
        base_weight: Base weight for exponential decay (older years get less weight)
    """
    all_data = []
    
    for year in range(start_year, end_year + 1):
        file_path = f'LinearRegression/data/{year}/training_data.csv'
        
        if os.path.exists(file_path):
            print(f"Loading data for year {year}")
            df = pd.read_csv(file_path)
            
            # Calculate weight based on recency
            years_from_present = end_year - year
            weight = base_weight ** years_from_present
            
            # Add weight column
            df['sample_weight'] = weight
            all_data.append(df)
        else:
            print(f"No data found for year {year}")
    
    if not all_data:
        raise ValueError("No training data found")
        
    # Combine all years
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"Combined {len(all_data)} years of data, total samples: {len(combined_df)}")
    
    return combined_df
