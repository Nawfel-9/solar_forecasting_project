import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any
from datetime import datetime, timedelta
import joblib
from models.lstm_model import LSTM

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_energy_data():
    """Load and preprocess energy data"""
    cost = pd.read_csv("data/preprocessed/energy_cost.csv", index_col=0, parse_dates=True).squeeze()
    consumed = pd.read_csv("data/preprocessed/energy_consumed.csv", index_col=0, parse_dates=True).squeeze()
    generated = pd.read_csv("data/preprocessed/energy_generated.csv", index_col=0, parse_dates=True).squeeze()
    return cost, consumed, generated

def load_data():
    """
    Load historical data for visualization.
    
    Returns:
        Dictionary containing DataFrames with historical data.
    """
    try:
        # In a real implementation, you would load your actual data files
        # This is a placeholder that creates synthetic data for demonstration
        
        # Create date ranges
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        # Hourly data for generation
        hourly_range = pd.date_range(start=start_date, end=end_date, freq='h')
        # Weekly data for consumption and cost
        weekly_range = pd.date_range(start=start_date, end=end_date, freq='W')
        
        # Create synthetic data
        # Solar generation with daily pattern (higher during day, zero at night)
        generation = []
        for ts in hourly_range:
            hour = ts.hour
            if 6 <= hour <= 18:  # Daylight hours
                # Simple sine pattern with random noise
                hour_factor = np.sin(np.pi * (hour - 6) / 12)
                day_factor = 0.5 + 0.5 * np.sin(np.pi * ts.dayofyear / 365)  # Seasonal variation
                gen = 5 * hour_factor * day_factor + 0.5 * np.random.random()
                gen = max(0, gen)  # Ensure non-negative
            else:
                gen = 0  # No generation at night
            generation.append(gen)
        
        # Weekly consumption with some randomness
        consumption = [
            30 + 5 * np.sin(np.pi * i / 26) + 3 * np.random.random() 
            for i, _ in enumerate(weekly_range)
        ]
        
        # Weekly electricity price with upward trend
        cost = [
            0.12 + 0.01 * (i / len(weekly_range)) + 0.005 * np.random.random() 
            for i, _ in enumerate(weekly_range)
        ]
        
        # Create pandas Series
        data = {
            'generated': pd.Series(generation, index=hourly_range),
            'consumed': pd.Series(consumption, index=weekly_range),
            'cost': pd.Series(cost, index=weekly_range)
        }
        
        return data
        
    except Exception as e:
        print(f"Error loading data: {e}")
        # Return empty data in case of error
        return {
            'generated': pd.Series(dtype=float),
            'consumed': pd.Series(dtype=float),
            'cost': pd.Series(dtype=float)
        }
def load_data_aligned():
    """
    Load energy data from prepared datasets.
    This is your existing function that loads the actual data.
    """
    # Placeholder - your implementation would load actual data files
    # Return a dictionary with 'generated', 'consumed', and 'cost' keys
    # For now, create synthetic data for demonstration
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    # Hourly data for generation
    hourly_range = pd.date_range(start=start_date, end=end_date, freq='h')
    # Weekly data for consumption and cost
    weekly_range = pd.date_range(start=start_date, end=end_date, freq='W')
    
    # Create synthetic data with realistic patterns
    generation = []
    for ts in hourly_range:
        hour = ts.hour
        if 6 <= hour <= 18:  # Daylight hours
            hour_factor = np.sin(np.pi * (hour - 6) / 12)
            day_factor = 0.5 + 0.5 * np.sin(np.pi * ts.dayofyear / 365)  # Seasonal variation
            gen = 5 * hour_factor * day_factor + 0.5 * np.random.random()
            gen = max(0, gen)  # Ensure non-negative
        else:
            gen = 0  # No generation at night
        generation.append(gen)
    
    # Weekly consumption with some randomness
    consumption = [
        30 + 5 * np.sin(np.pi * i / 26) + 3 * np.random.random() 
        for i, _ in enumerate(weekly_range)
    ]
    
    # Weekly electricity price with upward trend
    cost = [
        0.12 + 0.01 * (i / len(weekly_range)) + 0.005 * np.random.random() 
        for i, _ in enumerate(weekly_range)
    ]
    
    # Create pandas Series
    data = {
        'generated': pd.Series(generation, index=hourly_range),
        'consumed': pd.Series(consumption, index=weekly_range),
        'cost': pd.Series(cost, index=weekly_range)
    }
    
    return data


def prepare_train_test(series, test_size=0.2):
    """Split into train/test sets using percentage"""
    test_length = int(len(series) * test_size)
    return series[:-test_length], series[-test_length:]

def create_sequences(data, time_steps=24):
    X, Y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i+time_steps])
        Y.append(data[i+time_steps])
    
    X = np.array(X)
    Y = np.array(Y)

    # Add feature dimension for PyTorch: (batch, time_steps, 1)
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
    Y = torch.tensor(Y, dtype=torch.float32)

    return X, Y

def create_lstm_sequences(data: np.ndarray, time_steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Create input-output sequences for LSTM"""
    X, y = [], []
    for i in range(len(data)-time_steps):
        X.append(data[i:(i+time_steps)])
        y.append(data[i+time_steps])
    return np.array(X), np.array(y)

def fit_scaler(data: pd.Series) -> StandardScaler:
    """Fit and return a scaler"""
    scaler = StandardScaler()
    scaler.fit(data.values.reshape(-1, 1))
    return scaler

def scale_data(data: pd.Series, scaler: StandardScaler) -> np.ndarray:
    """Scale data using pre-fitted scaler"""
    return scaler.transform(data.values.reshape(-1, 1)).squeeze()

def inverse_scale(data: np.ndarray, scaler: StandardScaler) -> pd.Series:
    """Inverse transform scaled data"""
    return scaler.inverse_transform(data.reshape(-1, 1)).squeeze()

def align_frequencies(generated_hourly, cost_weekly, consumed_weekly):
    """Align all series to hourly frequency"""
    # Resample weekly data to hourly (forward fill)
    cost_hourly = cost_weekly.resample('h').ffill()
    consumed_hourly = consumed_weekly.resample('h').ffill()
    
    # Align all indexes
    start_date = max(generated_hourly.index.min(), 
                    cost_hourly.index.min(), 
                    consumed_hourly.index.min())
    end_date = min(generated_hourly.index.max(), 
                  cost_hourly.index.max(), 
                  consumed_hourly.index.max())
    
    return (
        generated_hourly.loc[start_date:end_date],
        cost_hourly.loc[start_date:end_date],
        consumed_hourly.loc[start_date:end_date]
    )

def align_forecast_data(forecasts, horizon_days):
    """
    Aligns forecasts of different granularities (hourly, weekly) to a common time frame.
    
    Args:
        forecasts: Dictionary containing forecast data
        horizon_days: Number of days for forecast horizon
        
    Returns:
        DataFrame with aligned forecast data
    """
    # Start with the hourly generation data as the base timeframe
    if not forecasts['generated']['dates'].empty:
        base_timestamps = forecasts['generated']['dates']
        start_date = base_timestamps.min()
        end_date = start_date + timedelta(days=horizon_days)
        
        # Create a continuous datetime range at hourly frequency
        full_range = pd.date_range(start=start_date, end=end_date, freq='h')
        
        # Create the base dataframe with timestamps
        aligned_df = pd.DataFrame({'timestamp': full_range})
        aligned_df['generation_kw'] = np.nan
        
        # Map generation values to their timestamps
        gen_df = pd.DataFrame({
            'timestamp': forecasts['generated']['dates'],
            'generation_kw': forecasts['generated']['values']
        })
        aligned_df = pd.merge(aligned_df, gen_df, on='timestamp', how='left', suffixes=('', '_y'))
        aligned_df['generation_kw'] = aligned_df['generation_kw_y'].fillna(aligned_df['generation_kw'])
        aligned_df = aligned_df.drop(columns=['generation_kw_y'])
        
        # For weekly data (cost and consumption), expand to hourly by repeating values
        # First, create a mapping function that assigns the appropriate weekly value to each hour
        if 'cost' in forecasts and 'dates' in forecasts['cost']:
            weekly_timestamps = forecasts['cost']['dates']
            costs = forecasts['cost']['values']
            
            # Function to find the closest week's value for any given timestamp
            def get_weekly_value(ts, weekly_ts, values):
                # Find the closest week start date that's not after the timestamp
                valid_weeks = weekly_ts[weekly_ts <= ts]
                if len(valid_weeks) == 0:
                    return np.nan
                closest_week = valid_weeks.max()
                week_idx = weekly_ts.get_loc(closest_week)
                if week_idx < len(values):
                    return values[week_idx]
                return np.nan
            
            # Apply the mapping function to each timestamp
            aligned_df['cost_per_kwh'] = aligned_df['timestamp'].apply(
                lambda ts: get_weekly_value(ts, weekly_timestamps, costs)
            )
            
            # Do the same for consumption data
            if 'consumed' in forecasts and 'dates' in forecasts['consumed']:
                consumed_timestamps = forecasts['consumed']['dates']
                consumption = forecasts['consumed']['values']
                
                aligned_df['consumption_kwh'] = aligned_df['timestamp'].apply(
                    lambda ts: get_weekly_value(ts, consumed_timestamps, consumption)
                )
        
        return aligned_df
    
    return pd.DataFrame()

# Function to parse forecast horizon selection to days
def parse_horizon(horizon_text):
    if horizon_text == "1 Week":
        return 7
    elif horizon_text == "2 Weeks":
        return 14
    elif horizon_text == "1 Month":
        return 30
    return 7  # Default

def load_models():
    """Cache-loaded models to avoid redundant disk reads"""
    device = get_device()
    
    # LSTM Model
    lstm = LSTM().to(device)
    lstm.load_state_dict(torch.load("models/best_model_energy_generated.pth", 
                                  map_location=device, weights_only=False)["model_state_dict"])
    
    # SARIMA Models
    sarima_cost = joblib.load("models/sarima_cost_model.pkl")
    sarima_consumed = joblib.load("models/sarima_consumed_model.pkl")
    
    return {
        'lstm': lstm,
        'sarima_cost': sarima_cost,
        'sarima_consumed': sarima_consumed
    }
