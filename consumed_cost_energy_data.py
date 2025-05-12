"""
Energy Consumption Data Processor

This script loads electricity consumption data, processes it, and creates weekly data with realistic variations.
The processed data is saved to the data/preprocessed directory.
"""

import pandas as pd
import numpy as np
import os
from datetime import timedelta

def load_consumption_data(file_path):
    """
    Load energy consumption data from CSV file and prepare initial transformations.
    
    Args:
        file_path (str): Path to the CSV file containing energy consumption data
        
    Returns:
        pd.DataFrame: Processed dataframe with consumption data
    """
    # Load data from CSV
    df_consumed = pd.read_csv(file_path)
    
    # Select relevant columns
    df_consumed = df_consumed[['Revenue Month', 'Current Charges', 'Consumption (KWH)']]
    
    # Rename columns
    df_consumed.rename(columns={'Revenue Month': 'Datetime', 'Current Charges': 'Cost ($)'}, inplace=True)
    
    # Set datetime as index
    df_consumed.set_index('Datetime', inplace=True)
    df_consumed.index = pd.to_datetime(df_consumed.index)
    
    # Group by index to handle any duplicate dates
    df_consumed_grouped = df_consumed.groupby(df_consumed.index).sum()
    
    return df_consumed_grouped

def create_weekly_data_with_variation(monthly_data, start_date='2010-01-01', variation=0.15):
    """
    Create weekly data from monthly data with realistic variations.
    
    Args:
        monthly_data (pd.Series): Monthly data values
        start_date (str): Starting date for the weekly series
        variation (float): Amount of random variation to apply (0-1)
        
    Returns:
        pd.Series: Weekly data with variation
    """
    np.random.seed(42)  # For reproducibility
    weekly_data = []
    date_indices = []
    current_date = pd.to_datetime(start_date)
    
    for monthly_value in monthly_data:
        # Determine number of weeks in month (4 or 5)
        weeks = 4 + np.random.choice([0, 1])
        
        # Generate variation factors
        variations = np.random.normal(loc=1.0, scale=variation, size=weeks)
        variations = variations / variations.sum()  # Normalize
        
        # Apply variations to get weekly values
        weekly_values = monthly_value * variations
        
        # Add weekly data points
        for value in weekly_values:
            weekly_data.append(value)
            date_indices.append(current_date)
            current_date += timedelta(days=7)
    
    return pd.Series(weekly_data, index=date_indices)


def main():
    """Main execution function"""
    import os
    
    # Define file path
    file_path = r"C:\Users\Genji\Desktop\solar_forecasting_project\data\Electric_Consumption_And_Cost__2010_-_Feb_2025__20250311.csv"
    
    # Process data
    df_consumed_grouped = load_consumption_data(file_path)
    
    # Create weekly data
    weekly_energy_consumed = create_weekly_data_with_variation(df_consumed_grouped['Consumption (KWH)'])
    weekly_energy_cost = create_weekly_data_with_variation(df_consumed_grouped['Cost ($)'])
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join("data", "preprocessed")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the series to CSV files
    weekly_energy_consumed.to_frame('Consumption (KWH)').to_csv(os.path.join(output_dir, "energy_consumed.csv"))
    weekly_energy_cost.to_frame('Cost ($)').to_csv(os.path.join(output_dir, "energy_cost.csv"))
    
    print(f"Data processing complete.")
    print(f"Created {len(weekly_energy_consumed)} weekly consumption data points.")
    print(f"Created {len(weekly_energy_cost)} weekly cost data points.")
    print(f"Data saved to {output_dir} folder.")
    
    return weekly_energy_consumed, weekly_energy_cost


if __name__ == "__main__":
    # Execute main function when script is run directly
    weekly_energy_consumed, weekly_energy_cost = main()
    
    # Display sample of processed data
    print("\nSample of weekly consumption data:")
    print(weekly_energy_consumed.head())
    
    print("\nSample of weekly cost data:")
    print(weekly_energy_cost.head())