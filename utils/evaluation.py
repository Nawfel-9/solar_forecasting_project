# solar_forecasting_project/utils/evaluation.py
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Any,Dict, Union

def evaluate_forecast(true_values: Union[np.ndarray, pd.Series], 
                      predicted_values: Union[np.ndarray, pd.Series]) -> Dict[str, float]:
    """
    Calculates multiple common evaluation metrics for time series forecasts.

    Args:
        true_values (Union[np.ndarray, pd.Series]): The actual observed values.
        predicted_values (Union[np.ndarray, pd.Series]): The values predicted by the model.

    Returns:
        Dict[str, float]: A dictionary containing MAE, RMSE, MAPE, and R2 score.
                          MAPE is returned as a percentage.
                          Returns empty dict if inputs are invalid.
    """
    if len(true_values) != len(predicted_values):
        print("Error: True values and predicted values have different lengths. Cannot evaluate.")
        return {}
    if len(true_values) == 0:
        print("Error: Input arrays are empty. Cannot evaluate.")
        return {}

    # Ensure numpy arrays for calculations
    true_values_np = np.asarray(true_values)
    predicted_values_np = np.asarray(predicted_values)

    mae = mean_absolute_error(true_values_np, predicted_values_np)
    rmse = np.sqrt(mean_squared_error(true_values_np, predicted_values_np))
    r2 = r2_score(true_values_np, predicted_values_np)

    # Calculate MAPE, handling potential division by zero if true_values contain zero
    # Replace true zeros with a very small number to avoid division by zero,
    # or only calculate MAPE for non-zero true values.
    # Here, we'll mask zeros.
    mask = true_values_np != 0
    if np.any(mask): # Proceed if there's at least one non-zero true value
        mape = np.mean(np.abs((true_values_np[mask] - predicted_values_np[mask]) / true_values_np[mask])) * 100
    else: # All true values are zero
        # If all predictions are also zero, MAPE is arguably 0. Otherwise, it's infinite or undefined.
        mape = 0.0 if np.all(predicted_values_np == 0) else np.nan


    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE (%)': mape,
        'R2 Score': r2
    }

def print_metrics(metrics: Dict[str, float], model_name: str = "Model") -> None:
    """
    Prints the evaluation metrics in a formatted way.

    Args:
        metrics (Dict[str, float]): A dictionary of metric names and their values.
        model_name (str): Name of the model or forecast being reported (for context in printout).
    """
    if not metrics:
        print(f"No metrics to display for {model_name}.")
        return

    print(f"\n--- {model_name} Evaluation Metrics ---")
    for metric_name, value in metrics.items():
        if isinstance(value, float):
            print(f"{metric_name:<10}: {value:.3f}")
        else:
            print(f"{metric_name:<10}: {value}") # For non-float values like NaN
    print("-------------------------------------")

def calculate_financial_summary(
    aligned_forecast_df_hourly: pd.DataFrame, # Expects HOURLY data
    # User's desired system parameters
    user_num_panels: int,
    user_panel_capacity_kw: float,
    # Reference system parameters
    reference_num_panels: int,
    reference_panel_capacity_w: float,
    # Other parameters
    panel_efficiency: float = 0.9,
    export_tariff_rate: float = 0.0,
    import_tariff_col: str = 'cost_per_kwh'
    ) -> Dict[str, Union[float, Any]]:
    """
    Calculates financial summary metrics based on HOURLY aligned forecast data.
    It scales the generation forecast from a reference system to the user-defined system size.
    """
    # Initialize results dictionary with clear keys for user's system
    results: Dict[str, Union[float, Any]] = {
        "total_hours_forecasted": 0,
        "total_solar_capacity_kwp_user_system": 0,
        "total_energy_generated_kwh_user_system": 0,
        "total_energy_consumed_kwh": 0,
        "energy_self_consumed_kwh": 0,
        "energy_exported_kwh": 0,
        "energy_imported_kwh": 0,
        "cost_without_solar_usd": 0,
        "cost_with_solar_usd": 0,
        "revenue_from_export_usd": 0,
        "net_savings_usd_comprehensive": 0,
        "avg_generation_kw_hourly_user_system": 0,
        "peak_generation_kw_hourly_user_system": 0,
        "avg_consumption_kwh_hourly": 0,
        "average_import_price_forecasted": np.nan,
        "net_savings_usd_simplified": np.nan
    }

    ref_gen_col = 'generation_kw'
    cons_col = 'consumption_kwh'
    required_cols = [ref_gen_col, cons_col, import_tariff_col]
    if aligned_forecast_df_hourly.empty or not all(col in aligned_forecast_df_hourly.columns for col in required_cols):
        print("Warning: Input DataFrame is empty or missing required columns.")
        return results

    df = aligned_forecast_df_hourly.copy()
    results["total_hours_forecasted"] = len(df)

    # Ensure key columns are numeric, filling NaNs with 0
    df[ref_gen_col] = pd.to_numeric(df[ref_gen_col], errors='coerce').fillna(0)
    for col in [cons_col, import_tariff_col]:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # 1. Calculate Total System Capacities
    reference_panel_cap_kw = float(reference_panel_capacity_w) / 1000.0
    reference_total_capacity_kwp = float(reference_num_panels) * reference_panel_cap_kw
    
    user_total_capacity_kwp = float(user_num_panels) * float(user_panel_capacity_kw)
    results["total_solar_capacity_kwp_user_system"] = user_total_capacity_kwp

    # 2. Scale HOURLY Generation Forecast
    scaling_factor = 0.0
    if reference_total_capacity_kwp > 0:
        scaling_factor = user_total_capacity_kwp / reference_total_capacity_kwp
    elif user_total_capacity_kwp > 0:
        scaling_factor = user_total_capacity_kwp
        print("Warning: Reference system capacity is 0. Scaling generation as if per 1kWp.")
    
    # Apply scaling to the reference system's hourly power forecast (kW)
    df['user_system_generation_kw'] = df[ref_gen_col] * scaling_factor
    
    # Apply efficiency to get usable AC energy (kWh) for the user's system for that hour
    df['actual_generation_kwh_user_system'] = df['user_system_generation_kw'] * panel_efficiency

    # 3. HOURLY Energy Balance Calculations
    df['energy_balance_kwh'] = df['actual_generation_kwh_user_system'] - df[cons_col]
    df['non_negative_generation_kwh'] = np.maximum(0, df['actual_generation_kwh_user_system'])
    df['self_consumption_kwh'] = np.minimum(df['non_negative_generation_kwh'], df[cons_col])
    df['exported_kwh'] = np.maximum(0, df['non_negative_generation_kwh'] - df['self_consumption_kwh'])
    df['imported_kwh'] = np.maximum(0, df[cons_col] - df['non_negative_generation_kwh'])

    # 4. Aggregate Energy Values
    results["total_energy_generated_kwh_user_system"] = df['actual_generation_kwh_user_system'].sum()
    results["total_energy_consumed_kwh"] = df[cons_col].sum()
    results["energy_self_consumed_kwh"] = df['self_consumption_kwh'].sum()
    results["energy_exported_kwh"] = df['exported_kwh'].sum()
    results["energy_imported_kwh"] = df['imported_kwh'].sum()

    if not df.empty:
        results["avg_generation_kw_hourly_user_system"] = df['actual_generation_kwh_user_system'].mean()
        results["peak_generation_kw_hourly_user_system"] = df['actual_generation_kwh_user_system'].max()
        results["avg_consumption_kwh_hourly"] = df[cons_col].mean()

    # 5. Financial Calculations
    results["cost_without_solar_usd"] = (df[cons_col] * df[import_tariff_col]).sum()
    results["revenue_from_export_usd"] = results["energy_exported_kwh"] * export_tariff_rate
    cost_of_imported_energy_usd = (df['imported_kwh'] * df[import_tariff_col]).sum()
    results["cost_with_solar_usd"] = cost_of_imported_energy_usd - results["revenue_from_export_usd"]
    results["net_savings_usd_comprehensive"] = results["cost_without_solar_usd"] - results["cost_with_solar_usd"]

    if df[import_tariff_col].notna().any() and df[import_tariff_col].nunique(dropna=True) > 0 :
        results["average_import_price_forecasted"] = np.nanmean(df[import_tariff_col][df[import_tariff_col].notna()].astype(float))
    else:
        results["average_import_price_forecasted"] = 0.0
    
    total_non_negative_generated_kwh = df['non_negative_generation_kwh'].sum()
    if pd.notna(results["average_import_price_forecasted"]):
        results["net_savings_usd_simplified"] = total_non_negative_generated_kwh * results["average_import_price_forecasted"]
    else:
        results["net_savings_usd_simplified"] = 0.0

    # Final cleanup of NaNs for display
    for key in results:
        if isinstance(results[key], float) and pd.isna(results[key]):
            results[key] = 0.0
            
    return results