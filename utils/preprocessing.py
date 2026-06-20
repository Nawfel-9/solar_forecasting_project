# utils/preprocessing.py
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, Tuple, Optional
import joblib
import streamlit as st

# Allow project-root imports when this module is executed from sub-directories
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.append(str(_PROJECT_ROOT))

from models.lstm_model import LSTMAttention as LSTM
from utils.config_loader import load_config, get_model_path

CONFIG = load_config()


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Return the best available PyTorch device (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_energy_data_from_config(log_errors: bool = True) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
    """Load energy cost (weekly), consumption (weekly), and generation (hourly, multivariate) data.

    Paths are resolved from the global CONFIG (data_paths section).

    Args:
        log_errors: When True, surface file/parse errors via ``st.error``.

    Returns:
        Tuple of:
        - cost series (weekly, $)
        - consumed series (weekly, kWh)
        - generated DataFrame (hourly, target + cyclical time features)
          Returns an empty DataFrame on critical errors.
    """
    data_cfg = CONFIG["data_paths"]
    cost_path = Path(data_cfg["preprocessed_dir"]) / data_cfg["energy_cost_csv"]
    consumed_path = Path(data_cfg["preprocessed_dir"]) / data_cfg["energy_consumed_csv"]
    generated_path = Path(data_cfg["preprocessed_dir"]) / data_cfg["energy_generated_csv"]

    # Cost
    try:
        cost_series = pd.read_csv(cost_path, index_col=0, parse_dates=True).iloc[:, 0]
    except Exception as e:
        if log_errors:
            st.error(f"Error loading cost data from {cost_path}: {e}")
        cost_series = pd.Series(dtype=float)

    # Consumption
    try:
        consumed_series = pd.read_csv(consumed_path, index_col=0, parse_dates=True).iloc[:, 0]
    except Exception as e:
        if log_errors:
            st.error(f"Error loading consumption data from {consumed_path}: {e}")
        consumed_series = pd.Series(dtype=float)

    # Generation (multivariate — all feature columns required)
    try:
        generated_df = pd.read_csv(generated_path, index_col=0, parse_dates=True)
        target_col = data_cfg.get("lstm_target_column_name")
        if not target_col or target_col not in generated_df.columns:
            if log_errors:
                st.error(
                    f"Target column '{target_col}' not found in {generated_path}. "
                    "Check lstm_target_column_name in config.yaml."
                )
            generated_df = pd.DataFrame()
    except Exception as e:
        if log_errors:
            st.error(f"Error loading generation data from {generated_path}: {e}")
        generated_df = pd.DataFrame()

    return cost_series, consumed_series, generated_df


# ---------------------------------------------------------------------------
# Train / test splitting
# ---------------------------------------------------------------------------

def prepare_train_test(series: pd.Series, test_size: Optional[float] = None) -> Tuple[pd.Series, pd.Series]:
    """Split a time series into train and test sets (no shuffle).

    Args:
        series: Input time series.
        test_size: Fraction of the series to use as the test set.
            Defaults to preprocessing.test_size in config.

    Returns:
        Tuple of (train_series, test_series).
    """
    actual_test_size = test_size if test_size is not None else CONFIG["preprocessing"]["test_size"]

    if series.empty:
        return pd.Series(dtype=series.dtype), pd.Series(dtype=series.dtype)

    n = len(series)
    test_length = int(n * actual_test_size)
    if test_length == 0 and n > 1:
        test_length = 1
    if n - test_length == 0 and n > 1:
        test_length = n - 1
    test_length = max(test_length, 0)

    split = n - test_length
    return series.iloc[:split], series.iloc[split:]


# ---------------------------------------------------------------------------
# Sequence creation
# ---------------------------------------------------------------------------

def create_lstm_sequences(data: np.ndarray, time_steps: int, output_chunk_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Create sliding-window (X, y) pairs for a direct multi-step LSTM.

    Assumes the target variable occupies the **first column** of ``data``.

    Args:
        data: Array of shape ``(num_timesteps, num_features)``.
        time_steps: Length of the input sequence (look-back window).
        output_chunk_size: Number of future steps to predict at once.

    Returns:
        Tuple ``(X, y)`` where:
        - X has shape ``(n_samples, time_steps, num_features)``
        - y has shape ``(n_samples, output_chunk_size)`` (target column only)
    """
    X, y = [], []
    for i in range(len(data) - time_steps - output_chunk_size + 1):
        X.append(data[i : i + time_steps, :])
        y.append(data[i + time_steps : i + time_steps + output_chunk_size, 0])
    return np.array(X), np.array(y)


# ---------------------------------------------------------------------------
# Scaler helpers
# ---------------------------------------------------------------------------

def fit_scaler_and_save(data_for_fitting: pd.Series, scaler_filename_key: str) -> StandardScaler:
    """Fit a StandardScaler on the provided series and persist it to disk.

    Args:
        data_for_fitting: 1-D series of values to fit on (training set only).
        scaler_filename_key: Key in data_paths config pointing to the scaler filename.

    Returns:
        The fitted StandardScaler instance.
    """
    scaler = StandardScaler()
    if data_for_fitting.empty:
        st.warning("Data for fitting scaler is empty. Returning unfitted scaler.")
        return scaler

    scaler.fit(data_for_fitting.values.reshape(-1, 1))

    scaler_save_path = get_model_path(scaler_filename_key)
    Path(scaler_save_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_save_path)
    print(f"Scaler fitted and saved to {scaler_save_path}")
    return scaler


def load_scaler(scaler_filename_key: str) -> Optional[StandardScaler]:
    """Load a pre-fitted StandardScaler from disk.

    Args:
        scaler_filename_key: Key in data_paths config pointing to the scaler filename.

    Returns:
        Loaded StandardScaler, or None on failure.
    """
    scaler_path = get_model_path(scaler_filename_key)
    try:
        scaler = joblib.load(scaler_path)
        print(f"Scaler loaded from {scaler_path}")
        return scaler
    except FileNotFoundError:
        st.error(f"Scaler file not found: {scaler_path}. Train the LSTM model to generate it.")
    except Exception as e:
        st.error(f"Error loading scaler from {scaler_path}: {e}")
    return None


def scale_data(data: pd.Series, scaler: Optional[StandardScaler]) -> np.ndarray:
    """Apply a fitted StandardScaler to a pandas Series.

    Args:
        data: Values to transform.
        scaler: Fitted scaler (must have ``mean_`` attribute).

    Returns:
        Scaled numpy array, or the raw values if the scaler is unavailable.
    """
    if scaler is None or not hasattr(scaler, "mean_"):
        st.warning("Scaler is not available or not fitted. Returning unscaled data.")
        return data.values if isinstance(data, pd.Series) else np.array([])
    if data.empty:
        return np.array([])
    return scaler.transform(data.values.reshape(-1, 1)).squeeze()


def inverse_scale(scaled_data: np.ndarray, scaler: Optional[StandardScaler]) -> np.ndarray:
    """Reverse a StandardScaler transformation.

    Args:
        scaled_data: Scaled values to invert.
        scaler: The same fitted scaler used for the forward transform.

    Returns:
        Unscaled numpy array, or the original scaled data if the scaler is unavailable.
    """
    if scaler is None or not hasattr(scaler, "mean_"):
        st.warning("Scaler is not available or not fitted. Returning original scaled data.")
        return scaled_data
    if scaled_data.size == 0:
        return np.array([])
    reshaped = scaled_data.reshape(-1, 1) if scaled_data.ndim <= 1 else scaled_data
    return scaler.inverse_transform(reshaped).squeeze()


# ---------------------------------------------------------------------------
# Forecast alignment
# ---------------------------------------------------------------------------

def align_forecast_data(
    forecasts_dict: Dict[str, Dict[str, Any]],
    master_hourly_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Align forecasts of different granularities to a common hourly DatetimeIndex.

    - ``'generated'`` forecast is assumed to be hourly and is reindexed directly.
    - ``'cost'`` (weekly total $) and ``'consumed'`` (weekly total kWh) are used
      to derive a $/kWh rate and an average hourly consumption, both forward-filled
      to the hourly index.

    Args:
        forecasts_dict: Keys ``'generated'``, ``'cost'``, ``'consumed'``.
            Each value must contain ``'dates'`` and ``'values'``.
        master_hourly_index: Target hourly DatetimeIndex.

    Returns:
        DataFrame indexed by ``master_hourly_index`` (column name ``'timestamp'``)
        with columns ``'generation_kw'``, ``'cost_per_kwh'``, ``'consumption_kwh'``.
    """
    if not isinstance(master_hourly_index, pd.DatetimeIndex):
        raise ValueError("master_hourly_index must be a pandas DatetimeIndex.")
    if master_hourly_index.empty:
        print("Warning: master_hourly_index is empty in align_forecast_data. Returning empty DataFrame.")
        return pd.DataFrame(columns=["timestamp", "generation_kw", "cost_per_kwh", "consumption_kwh"])

    aligned_df = pd.DataFrame(index=master_hourly_index)
    aligned_df.index.name = "timestamp"

    # --- Generation (hourly) ---
    gen_fc = forecasts_dict.get("generated", {})
    gen_dates = gen_fc.get("dates")
    gen_values = gen_fc.get("values")
    if gen_dates is not None and gen_values is not None and len(gen_values) > 0:
        try:
            gen_idx = pd.DatetimeIndex(gen_dates)
            if len(gen_idx) == len(gen_values):
                gen_series = pd.Series(gen_values, index=gen_idx, name="generation_kw")
                aligned_df["generation_kw"] = gen_series.reindex(
                    master_hourly_index, method="nearest", tolerance=pd.Timedelta("30min")
                )
            else:
                print("Warning (align_forecast_data): Length mismatch for generation dates/values.")
                aligned_df["generation_kw"] = np.nan
        except Exception as e:
            print(f"Warning (align_forecast_data): Error processing generation forecast: {e}")
            aligned_df["generation_kw"] = np.nan
    else:
        aligned_df["generation_kw"] = np.nan

    # --- Cost and consumption (weekly → hourly) ---
    cost_fc = forecasts_dict.get("cost", {})
    cons_fc = forecasts_dict.get("consumed", {})
    aligned_df["cost_per_kwh"] = np.nan
    aligned_df["consumption_kwh"] = np.nan

    cost_dates = cost_fc.get("dates")
    cost_vals = cost_fc.get("values")
    cons_dates = cons_fc.get("dates")
    cons_vals = cons_fc.get("values")

    if all(v is not None and len(v) > 0 for v in [cost_dates, cost_vals, cons_dates, cons_vals]):
        try:
            cost_idx = pd.DatetimeIndex(cost_dates)
            cons_idx = pd.DatetimeIndex(cons_dates)

            if (len(cost_idx) == len(cost_vals)
                    and len(cons_idx) == len(cons_vals)
                    and cost_idx.equals(cons_idx)):
                cost_s = pd.Series(cost_vals, index=cost_idx)
                cons_s = pd.Series(cons_vals, index=cons_idx)

                # Derived $/kWh (NaN where weekly consumption is zero)
                price_per_kwh = cost_s / cons_s.replace(0, np.nan)
                aligned_df["cost_per_kwh"] = (
                    price_per_kwh.resample("h").ffill()
                    .reindex(master_hourly_index, method="ffill")
                )

                # Average hourly consumption from weekly totals
                avg_hourly_cons = cons_s.resample("h").ffill() / (7.0 * 24.0)
                aligned_df["consumption_kwh"] = avg_hourly_cons.reindex(master_hourly_index, method="ffill")
            else:
                print("Warning (align_forecast_data): Date mismatch between weekly cost and consumption forecasts.")
        except Exception as e:
            print(f"Warning (align_forecast_data): Error processing cost/consumption forecasts: {e}")
    else:
        print("Warning (align_forecast_data): Weekly cost or consumption data is missing or empty.")

    return aligned_df.reset_index()


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def get_steps_from_config(horizon_key: str, config: dict = CONFIG) -> dict:
    """Retrieve hourly and weekly step counts for a named forecast horizon.

    Args:
        horizon_key: Key in ``forecast_horizons.steps_map`` (e.g. ``'1 Month'``).
        config: Config dictionary (defaults to global CONFIG).

    Returns:
        Dict with keys ``'hourly'`` and ``'weekly'``.
    """
    try:
        return config["forecast_horizons"]["steps_map"][horizon_key]
    except KeyError:
        default_key = config["forecast_horizons"]["default_app_horizon"]
        st.warning(f"Horizon key '{horizon_key}' not in config. Falling back to '{default_key}'.")
        return config["forecast_horizons"]["steps_map"][default_key]


# ---------------------------------------------------------------------------
# Model loading (cached for the Streamlit session)
# ---------------------------------------------------------------------------

@st.cache_resource(ttl=CONFIG.get("caching", {}).get("model_ttl_seconds", 3600))
def load_all_models_and_scaler() -> Dict[str, Any]:
    """Load all trained models and the LSTM scaler, with Streamlit session caching.

    Reads model paths and architecture parameters from the global CONFIG.

    Returns:
        Dict with keys ``'lstm'``, ``'lstm_scaler'``, ``'sarima_cost'``,
        ``'sarima_consumed'``. Any component that fails to load is set to None.
    """
    device = get_device()
    loaded: Dict[str, Any] = {
        "lstm": None,
        "lstm_scaler": None,
        "sarima_cost": None,
        "sarima_consumed": None,
    }
    lstm_cfg = CONFIG["lstm_params"]

    # LSTM
    try:
        model = LSTM(
            input_size=lstm_cfg["input_size"],
            hidden_size=lstm_cfg["hidden_size"],
            num_layers=lstm_cfg["num_layers"],
            dropout=lstm_cfg["dropout"],
            bidirectional=lstm_cfg.get("bidirectional", False),
            output_chunk_size=lstm_cfg.get("output_chunk_size", 1),
        ).to(device)

        lstm_path = get_model_path("lstm_model_name")
        checkpoint = torch.load(lstm_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        loaded["lstm"] = model
        print(f"LSTM model loaded from {lstm_path}.")
    except FileNotFoundError:
        st.error(f"LSTM model file not found: {get_model_path('lstm_model_name')}. Train it first.")
    except KeyError as e:
        st.error(f"Error loading LSTM model: missing config key {e}.")
    except Exception as e:
        st.error(f"Error loading LSTM model: {e}")

    # Scaler
    loaded["lstm_scaler"] = load_scaler("lstm_scaler_name")

    # SARIMA models
    for key in ["cost", "consumed"]:
        model_key = f"sarima_{key}_model_name"
        sarima_path = get_model_path(model_key)
        try:
            loaded[f"sarima_{key}"] = joblib.load(sarima_path)
            print(f"SARIMA {key} model loaded from {sarima_path}.")
        except FileNotFoundError:
            st.error(f"SARIMA {key} model not found: {sarima_path}. Train it first.")
        except Exception as e:
            st.error(f"Error loading SARIMA {key} model: {e}")

    return loaded


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def add_cyclical_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add sine/cosine encodings of hour-of-day and day-of-year to a DataFrame.

    Args:
        df: DataFrame with a DatetimeIndex (or convertible index).

    Returns:
        Copy of ``df`` with four additional columns:
        ``hour_sin``, ``hour_cos``, ``day_of_year_sin``, ``day_of_year_cos``.
    """
    df_out = df.copy()
    df_out.index = pd.to_datetime(df_out.index)
    df_out["hour_sin"] = np.sin(2 * np.pi * df_out.index.hour / 24.0)
    df_out["hour_cos"] = np.cos(2 * np.pi * df_out.index.hour / 24.0)
    df_out["day_of_year_sin"] = np.sin(2 * np.pi * df_out.index.dayofyear / 365.25)
    df_out["day_of_year_cos"] = np.cos(2 * np.pi * df_out.index.dayofyear / 365.25)
    return df_out
