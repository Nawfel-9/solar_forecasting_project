# solar_forecasting_project/utils/preprocessing.py
import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, Tuple, List, Optional
from datetime import datetime, timedelta
import joblib
from pathlib import Path
import streamlit as st # For @st.cache_resource


# This structure allows the script/module to be run/imported from different locations
import sys
if str(Path(__file__).parent.parent) not in sys.path: # Add project root if not already there
    sys.path.append(str(Path(__file__).parent.parent))
from models.lstm_model import LSTMAttention as LSTM  # Import LSTM model class

# Centralized configuration loading
try:
    from utils.config_loader import load_config, get_preprocessed_data_path, get_model_path
except ImportError:
    import sys
    if str(Path(__file__).parent.parent) not in sys.path:
        sys.path.append(str(Path(__file__).parent.parent))
    from utils.config_loader import load_config, get_preprocessed_data_path, get_model_path

CONFIG = load_config()

def get_device() -> torch.device:
    """Gets the appropriate PyTorch device (CUDA or CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_energy_data_from_config(log_errors: bool = True) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
    """
    Loads energy cost (weekly), consumption (weekly), and generation (hourly, multivariate) data.

    Returns:
        Tuple[pd.Series, pd.Series, pd.DataFrame]:
            - cost series (weekly)
            - consumed series (weekly)
            - generated dataframe (hourly, with target and features)
    """
    data_cfg = CONFIG['data_paths']
    cost_path = Path(data_cfg['preprocessed_dir']) / data_cfg['energy_cost_csv']
    consumed_path = Path(data_cfg['preprocessed_dir']) / data_cfg['energy_consumed_csv']
    generated_path = Path(data_cfg['preprocessed_dir']) / data_cfg['energy_generated_csv']
    
    # --- Load Cost Data ---
    try:
        cost_df = pd.read_csv(cost_path, index_col=0, parse_dates=True)
        # Assuming cost data is a single column or the first column is the target
        cost_series = cost_df.iloc[:, 0]
    except (FileNotFoundError, pd.errors.EmptyDataError, Exception) as e:
        if log_errors: st.error(f"Error loading Cost data from {cost_path}: {e}")
        cost_series = pd.Series(dtype=float)

    # --- Load Consumption Data ---
    try:
        consumed_df = pd.read_csv(consumed_path, index_col=0, parse_dates=True)
        # Assuming consumption data is a single column or the first column is the target
        consumed_series = consumed_df.iloc[:, 0]
    except (FileNotFoundError, pd.errors.EmptyDataError, Exception) as e:
        if log_errors: st.error(f"Error loading Consumption data from {consumed_path}: {e}")
        consumed_series = pd.Series(dtype=float)

    # --- Load Generation Data (Multivariate) ---
    try:
        # Load the entire DataFrame because we need all feature columns
        generated_df = pd.read_csv(generated_path, index_col=0, parse_dates=True)
        
        # Check if the required target column exists
        target_col_name = data_cfg.get('lstm_target_column_name')
        if not target_col_name or target_col_name not in generated_df.columns:
            if log_errors:
                st.error(f"The specified target column '{target_col_name}' was not found in {generated_path}. "
                         "Please check `lstm_target_column_name` in your config.yaml.")
            # Return an empty DataFrame to signal a critical error
            generated_df = pd.DataFrame()
            
    except (FileNotFoundError, pd.errors.EmptyDataError, Exception) as e:
        if log_errors: st.error(f"Error loading Generation data from {generated_path}: {e}")
        generated_df = pd.DataFrame() # Return empty DataFrame on any error
            
    return cost_series, consumed_series, generated_df

def prepare_train_test(series: pd.Series, test_size: Optional[float] = None) -> Tuple[pd.Series, pd.Series]:
    """Splits a time series into training and testing sets."""
    actual_test_size = test_size if test_size is not None else CONFIG['preprocessing']['test_size']
    
    if series.empty:
        return pd.Series(dtype=series.dtype), pd.Series(dtype=series.dtype)

    n = len(series)
    test_length = int(n * actual_test_size)
    
    if test_length == 0 and n > 1: test_length = 1 # Min 1 for test if possible
    if n - test_length == 0 and n > 1: test_length = n - 1 # Min 1 for train if possible
    if test_length < 0: test_length = 0
    
    split_point = n - test_length
    train_series = series.iloc[:split_point]
    test_series = series.iloc[split_point:]
    
    return train_series, test_series

def create_lstm_sequences(data: np.ndarray, time_steps: int, output_chunk_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates input-output sequences for a MULTIVARIATE Direct Multi-Step model.
    Assumes the target variable is the FIRST column in the data array.
    """
    X, y = [], []
    # data shape: (num_timesteps, num_features)
    num_features = data.shape[1]
    
    for i in range(len(data) - time_steps - output_chunk_size + 1):
        # Input sequence X will contain all features
        X.append(data[i:(i + time_steps), :]) # Shape: (time_steps, num_features)
        
        # Output sequence y will contain ONLY the target variable (the first column)
        y.append(data[(i + time_steps):(i + time_steps + output_chunk_size), 0]) # Shape: (output_chunk_size,)
        
    return np.array(X), np.array(y)

def fit_scaler_and_save(data_for_fitting: pd.Series, scaler_filename_key: str) -> StandardScaler:
    """Fits a StandardScaler and saves it using a key from CONFIG for the filename."""
    scaler = StandardScaler()
    if data_for_fitting.empty:
        st.warning("Data for fitting scaler is empty. Returning unfitted scaler.")
        return scaler # Unfitted scaler
        
    scaler.fit(data_for_fitting.values.reshape(-1, 1))
    
    scaler_save_path = get_model_path(scaler_filename_key) # Uses data_paths.models_dir
    Path(scaler_save_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_save_path)
    print(f"Scaler fitted and saved to {scaler_save_path}")
    return scaler

def load_scaler(scaler_filename_key: str) -> Optional[StandardScaler]:
    """Loads a pre-fitted StandardScaler using a key from CONFIG for the filename."""
    scaler_path = get_model_path(scaler_filename_key)
    try:
        scaler = joblib.load(scaler_path)
        print(f"Scaler loaded from {scaler_path}")
        return scaler
    except FileNotFoundError:
        st.error(f"Scaler file not found: {scaler_path}. Train LSTM model to generate it.")
    except Exception as e:
        st.error(f"Error loading scaler from {scaler_path}: {e}")
    return None # Return None if loading fails

def scale_data(data: pd.Series, scaler: Optional[StandardScaler]) -> np.ndarray:
    """Scales Series data using a pre-fitted scaler. Returns empty array on issues."""
    if scaler is None or not hasattr(scaler, 'mean_'):
        st.warning("Scaler is not available or not fitted. Returning unscaled data as numpy array.")
        return data.values if isinstance(data, pd.Series) else np.array([])
    if data.empty: return np.array([])
    return scaler.transform(data.values.reshape(-1, 1)).squeeze()

def inverse_scale(scaled_data: np.ndarray, scaler: Optional[StandardScaler]) -> np.ndarray:
    """Inverse transforms scaled numpy array data using a pre-fitted scaler."""
    if scaler is None or not hasattr(scaler, 'mean_'):
        st.warning("Scaler is not available or not fitted. Returning original scaled data.")
        return scaled_data
    if scaled_data.size == 0: return np.array([])

    data_reshaped = scaled_data.reshape(-1, 1) if scaled_data.ndim == 1 or scaled_data.ndim == 0 else scaled_data
    return scaler.inverse_transform(data_reshaped).squeeze()

def align_frequencies(
    generated_hourly: pd.Series, 
    cost_series: pd.Series, 
    consumed_series: pd.Series,
    target_freq: str = 'h' # Target frequency, typically hourly
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Aligns multiple time series to a common target frequency and date range.
    Assumes generated_hourly is already at target_freq if target_freq='h'.
    Other series are resampled (e.g., weekly cost/consumption to hourly).
    """
    all_valid_series = []
    if not generated_hourly.empty and generated_hourly.index.is_monotonic_increasing:
        all_valid_series.append(generated_hourly)
    
    # Resample cost and consumed if their frequency is different from target_freq
    # This simplified version assumes cost/consumed might be daily/weekly and need ffill to hourly
    # A more robust version would check actual frequencies.
    resampled_cost = cost_series.resample(target_freq).ffill() if not cost_series.empty else pd.Series(dtype=float)
    if not resampled_cost.empty and resampled_cost.index.is_monotonic_increasing:
        all_valid_series.append(resampled_cost)

    resampled_consumed = consumed_series.resample(target_freq).ffill() if not consumed_series.empty else pd.Series(dtype=float)
    if not resampled_consumed.empty and resampled_consumed.index.is_monotonic_increasing:
        all_valid_series.append(resampled_consumed)

    if not all_valid_series:
        empty_idx = pd.DatetimeIndex([], freq=target_freq)
        return pd.Series(dtype=float, index=empty_idx), pd.Series(dtype=float, index=empty_idx), pd.Series(dtype=float, index=empty_idx)

    # Determine common date range (intersection)
    common_start = max(s.index.min() for s in all_valid_series)
    common_end = min(s.index.max() for s in all_valid_series)

    if common_start > common_end: # No overlapping period
        empty_idx = pd.DatetimeIndex([], freq=target_freq, start=common_start, end=common_end) # Keep freq for consistency
        return pd.Series(dtype=float, index=empty_idx), pd.Series(dtype=float, index=empty_idx), pd.Series(dtype=float, index=empty_idx)
        
    # Create a master index for the common period at the target frequency
    master_index = pd.date_range(start=common_start, end=common_end, freq=target_freq)

    # Reindex all series to this master index
    gen_aligned = generated_hourly.reindex(master_index, method='nearest', limit=1) if not generated_hourly.empty else pd.Series(dtype=float, index=master_index)
    cost_aligned = resampled_cost.reindex(master_index, method='ffill') if not resampled_cost.empty else pd.Series(dtype=float, index=master_index)
    cons_aligned = resampled_consumed.reindex(master_index, method='ffill') if not resampled_consumed.empty else pd.Series(dtype=float, index=master_index)
    
    return gen_aligned, cost_aligned, cons_aligned


def align_forecast_data(
    forecasts_dict: Dict[str, Dict[str, Any]],
    master_hourly_index: pd.DatetimeIndex
    ) -> pd.DataFrame:
    """
    Aligns forecasts of different granularities to a common master hourly DatetimeIndex.
    - 'generation' forecast is assumed to be alignable to hourly.
    - 'cost' (weekly total $) and 'consumed' (weekly total kWh) are used to derive
      a weekly $/kWh rate, which is then forward-filled to hourly for 'cost_per_kwh'.
    - 'consumed' (weekly total kWh) is averaged to an hourly rate for 'consumption_kwh'.

    Args:
        forecasts_dict (Dict): Forecasts for 'generated', 'cost', 'consumed'.
                               Each item must have 'dates' (pd.DatetimeIndex or similar)
                               and 'values' (np.array or list).
                               'cost' and 'consumed' are expected to have aligned weekly dates and values.
        master_hourly_index (pd.DatetimeIndex): The target hourly index for alignment.

    Returns:
        pd.DataFrame: With 'timestamp' (from master_hourly_index) and aligned forecast columns:
                      'generation_kw', 'cost_per_kwh', 'consumption_kwh'.
    """
    if not isinstance(master_hourly_index, pd.DatetimeIndex):
        raise ValueError("master_hourly_index must be a pandas DatetimeIndex.")
    if master_hourly_index.empty:
        print("Warning: master_hourly_index is empty in align_forecast_data. Returning empty DataFrame.")
        return pd.DataFrame(columns=['timestamp', 'generation_kw', 'cost_per_kwh', 'consumption_kwh'])

    aligned_df = pd.DataFrame(index=master_hourly_index)
    aligned_df.index.name = 'timestamp'

    # --- Align Generation Forecast (Assumed hourly or near-hourly) ---
    gen_fc = forecasts_dict.get('generated', {})
    gen_dates = gen_fc.get('dates')
    gen_values = gen_fc.get('values')
    if gen_dates is not None and gen_values is not None and len(gen_values) > 0:
        try:
            gen_dates_idx = pd.DatetimeIndex(gen_dates)
            if len(gen_dates_idx) == len(gen_values):
                gen_series = pd.Series(gen_values, index=gen_dates_idx, name='generation_kw')
                aligned_df['generation_kw'] = gen_series.reindex(master_hourly_index, method='nearest', tolerance=pd.Timedelta('30minutes'))
            else:
                print("Warning (align_forecast_data): Length mismatch for generation dates/values. Generation data set to NaN.")
                aligned_df['generation_kw'] = np.nan
        except Exception as e:
            print(f"Warning (align_forecast_data): Error processing generation forecast: {e}. Generation data set to NaN.")
            aligned_df['generation_kw'] = np.nan
    else:
        aligned_df['generation_kw'] = np.nan

    # --- Derive Hourly Cost ($/kWh) and Consumption (kWh) from Weekly Totals ---
    cost_fc = forecasts_dict.get('cost', {})
    cons_fc = forecasts_dict.get('consumed', {})

    cost_dates = cost_fc.get('dates')
    cost_values_weekly_total = cost_fc.get('values') # Weekly total cost $
    cons_dates = cons_fc.get('dates') # Assuming same dates as cost for weekly alignment
    cons_values_weekly_total = cons_fc.get('values') # Weekly total consumption kWh

    # Initialize columns to NaN
    aligned_df['cost_per_kwh'] = np.nan
    aligned_df['consumption_kwh'] = np.nan

    if (cost_dates is not None and cost_values_weekly_total is not None and len(cost_values_weekly_total) > 0 and
        cons_dates is not None and cons_values_weekly_total is not None and len(cons_values_weekly_total) > 0):
        try:
            cost_dates_idx = pd.DatetimeIndex(cost_dates)
            cons_dates_idx = pd.DatetimeIndex(cons_dates)

            if len(cost_dates_idx) == len(cost_values_weekly_total) and \
               len(cons_dates_idx) == len(cons_values_weekly_total) and \
               cost_dates_idx.equals(cons_dates_idx): # Critical: weekly dates must align for direct division

                cost_series_weekly_total = pd.Series(cost_values_weekly_total, index=cost_dates_idx)
                cons_series_weekly_total = pd.Series(cons_values_weekly_total, index=cons_dates_idx)

                # Calculate derived weekly price per kWh ($/kWh)
                # Handle division by zero: if weekly consumption is 0, price is undefined (NaN) or could be set to 0 if cost is also 0.
                derived_price_per_kwh_weekly = cost_series_weekly_total / cons_series_weekly_total.replace(0, np.nan) # Avoid division by zero, result in NaN

                # Resample derived weekly $/kWh to hourly, forward-filling values
                price_per_kwh_hourly_ffilled = derived_price_per_kwh_weekly.resample('h').ffill()
                aligned_df['cost_per_kwh'] = price_per_kwh_hourly_ffilled.reindex(master_hourly_index, method='ffill')

                # Disaggregate weekly total consumption to average hourly consumption
                cons_total_weekly_ffilled_hourly = cons_series_weekly_total.resample('h').ffill()
                average_hourly_consumption = cons_total_weekly_ffilled_hourly / (7.0 * 24.0)
                aligned_df['consumption_kwh'] = average_hourly_consumption.reindex(master_hourly_index, method='ffill')
            else:
                print("Warning (align_forecast_data): Mismatch in dates or lengths for weekly cost and consumption forecasts. Cannot derive $/kWh. Cost/Consumption data set to NaN.")
        except Exception as e:
            print(f"Warning (align_forecast_data): Error processing cost/consumption forecasts: {e}. Cost/Consumption data set to NaN.")
    else:
        print("Warning (align_forecast_data): Weekly cost or consumption forecast data is missing/empty. Cost/Consumption data set to NaN.")
        
    return aligned_df.reset_index()


def get_steps_from_config(horizon_key: str, config: dict = CONFIG) -> dict:
    """
    Retrieves 'hourly' and 'weekly' step counts from config for a given horizon_key.
    """
    try:
        return config['forecast_horizons']['steps_map'][horizon_key]
    except KeyError:
        default_key = config['forecast_horizons']['default_app_horizon']
        st.warning(f"Horizon key '{horizon_key}' not in config. Falling back to default: '{default_key}'.")
        return config['forecast_horizons']['steps_map'][default_key]

# --- REPLACE your old load_all_models_and_scaler with this one ---

@st.cache_resource(ttl=CONFIG.get('caching', {}).get('model_ttl_seconds', 3600))
def load_all_models_and_scaler() -> Dict[str, Any]:
    """
    Loads all trained models (LSTM, SARIMA) and the LSTM scaler using paths and
    parameters from the global CONFIG. Caches results.
    This version passes all necessary parameters to the new LSTM model constructor.
    """
    device = get_device()
    loaded_objects: Dict[str, Any] = {
        'lstm': None, 'lstm_scaler': None, 'sarima_cost': None, 'sarima_consumed': None
    }
    models_cfg = CONFIG['data_paths']
    lstm_params_cfg = CONFIG['lstm_params']

    # Load LSTM Model
    try:
        lstm_model_instance = LSTM(
            input_size=lstm_params_cfg['input_size'],
            hidden_size=lstm_params_cfg['hidden_size'],
            num_layers=lstm_params_cfg['num_layers'],
            dropout=lstm_params_cfg['dropout'],
            # NEW: Pass bidirectional and output_chunk_size from config
            bidirectional=lstm_params_cfg.get('bidirectional', False), # Default to False if not in config
            output_chunk_size=lstm_params_cfg.get('output_chunk_size', 1) # Default to 1 (single-step) if not in config
        ).to(device)
        
        lstm_model_path = get_model_path('lstm_model_name')
        # Explicitly set weights_only=False as you are loading a dictionary checkpoint
        checkpoint = torch.load(lstm_model_path, map_location=device, weights_only=False)
        
        # Handle both direct state_dict and checkpoint dict containing 'model_state_dict'
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            lstm_model_instance.load_state_dict(checkpoint['model_state_dict'])
        else:
            lstm_model_instance.load_state_dict(checkpoint)
        
        loaded_objects['lstm'] = lstm_model_instance
        print(f"LSTM model loaded successfully from {lstm_model_path}.")
    except FileNotFoundError:
        st.error(f"LSTM model file not found: {get_model_path('lstm_model_name')}. Please train it first.")
    except KeyError as e:
        st.error(f"Error loading LSTM model: Missing key {e} in lstm_params config. "
                 "Ensure 'bidirectional' and 'output_chunk_size' are defined.")
    except Exception as e:
        st.error(f"Error loading LSTM model: {e}")

    # Load LSTM Scaler (no changes needed here)
    loaded_objects['lstm_scaler'] = load_scaler('lstm_scaler_name')

    # Load SARIMA Models (no changes needed here)
    for model_key_suffix in ['cost', 'consumed']:
        sarima_model_key = f'sarima_{model_key_suffix}_model_name'
        sarima_path = get_model_path(sarima_model_key)
        try:
            loaded_objects[f'sarima_{model_key_suffix}'] = joblib.load(sarima_path)
            print(f"SARIMA {model_key_suffix} model loaded successfully from {sarima_path}.")
        except FileNotFoundError:
            st.error(f"SARIMA {model_key_suffix} model not found: {sarima_path}. Please train it first.")
        except Exception as e:
            st.error(f"Error loading SARIMA {model_key_suffix} model: {e}")
            
    return loaded_objects

def add_cyclical_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds cyclical time-based features to a dataframe with a DatetimeIndex."""
    df_out = df.copy()
    # Ensure index is datetime
    df_out.index = pd.to_datetime(df_out.index)

    # Hour of day (0-23)
    df_out['hour_sin'] = np.sin(2 * np.pi * df_out.index.hour / 24.0)
    df_out['hour_cos'] = np.cos(2 * np.pi * df_out.index.hour / 24.0)
    # Day of year (1-366)
    df_out['day_of_year_sin'] = np.sin(2 * np.pi * df_out.index.dayofyear / 365.25)
    df_out['day_of_year_cos'] = np.cos(2 * np.pi * df_out.index.dayofyear / 365.25)
    return df_out