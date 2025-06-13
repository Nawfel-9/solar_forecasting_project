# solar_forecasting_project/generated_energy_estimation.py
import numpy as np
import pandas as pd
import pvlib # Ensure pvlib is installed: pip install pvlib
from pathlib import Path
from utils.preprocessing import add_cyclical_time_features

# --- Keep all your helper functions as they were ---
# compute_solar_position, compute_poa_irradiance, 
# compute_cell_temperature, compute_dc_power, 
# estimate_daily_generated_energy (which actually estimates hourly power)
# ... (for brevity, assuming they are here) ...
# Let's rename the main estimation function for clarity
def estimate_hourly_dc_power(df_input_data, ambient_temp_col, ghi_col, dni_col, dhi_col, 
                            solar_zenith_col, latitude, longitude, surface_tilt, 
                            albedo, noct_module_temp, system_pdc0_watts, temp_coeff_pdc) -> pd.Series:
    # This function contains the exact logic from your previous 'estimate_generated_energy'
    # that results in the hourly DC power series.
    # ... (the full logic from your script to calculate and return hourly_dc_power_watts) ...
    # For now, I will just paste the core logic for this function here to be complete.
    df = df_input_data.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        if {'Year', 'Month', 'Day', 'Hour'}.issubset(df.columns):
            df['datetime_temp'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']]) if 'Minute' in df.columns else pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour']])
            df = df.set_index('datetime_temp', drop=True)
            df.index.name = 'datetime'
        else:
            df.index = pd.to_datetime(df.index)
    rename_map = {ambient_temp_col: 'Ambient Temperature', ghi_col: 'GHI', dni_col: 'DNI', dhi_col: 'DHI', solar_zenith_col: 'Solar Zenith Angle'}
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    solpos = pvlib.solarposition.get_solarposition(df.index, latitude, longitude)
    df['Solar Elevation'] = solpos['elevation']
    df['Solar Azimuth'] = solpos['azimuth']
    poa_irradiance = pvlib.irradiance.get_total_irradiance(surface_tilt=surface_tilt, surface_azimuth=180, solar_zenith=solpos['zenith'], solar_azimuth=solpos['azimuth'], dni=df['DNI'], ghi=df['GHI'], dhi=df['DHI'], albedo=albedo).fillna(0)
    df['cell_temperature'] = pvlib.temperature.sapm_cell(poa_irradiance['poa_global'], df['Ambient Temperature'], wind_speed=1, a=-3.56, b=-0.075, deltaT=3)
    power_dc = pvlib.pvsystem.pvwatts_dc(g_poa_effective=poa_irradiance['poa_global'], temp_cell=df['cell_temperature'], pdc0=system_pdc0_watts, gamma_pdc=temp_coeff_pdc).fillna(0)
    return np.maximum(0, power_dc)


if __name__ == "__main__":
    print("Starting solar energy estimation process with time features...")
    
    # --- Configuration for New York City ---
    raw_data_input_file = 'data/generated_2009_2023.csv'
    output_preprocessed_file = 'data/preprocessed/energy_generated.csv' 
    output_target_column_name = "Generated Energy (W)" 

    # --- Parameters (as before) ---
    cfg_ambient_temp_col = 'Temperature'; cfg_ghi_col = 'GHI'; cfg_dni_col = 'DNI'; cfg_dhi_col = 'DHI'
    cfg_solar_zenith_col = 'Solar Zenith Angle'
    cfg_latitude = 40.71; cfg_longitude = -74.00; cfg_surface_tilt = 35.0; cfg_albedo = 0.18
    cfg_noct_module_temp = 45.0; cfg_system_pdc0_watts = 750.0 * 10000; cfg_temp_coeff_pdc = -0.0037
    
    try:
        print(f"Loading raw hourly data from: {raw_data_input_file}")
        df_raw_input = pd.read_csv(raw_data_input_file)
        
        print("Estimating hourly DC power...")
        # I'm assuming your original logic is encapsulated in this function call
        power_series = estimate_hourly_dc_power(
            df_input_data=df_raw_input,
            ambient_temp_col=cfg_ambient_temp_col, ghi_col=cfg_ghi_col, dni_col=cfg_dni_col,
            dhi_col=cfg_dhi_col, solar_zenith_col=cfg_solar_zenith_col,
            latitude=cfg_latitude, longitude=cfg_longitude, surface_tilt=cfg_surface_tilt,
            albedo=cfg_albedo, noct_module_temp=cfg_noct_module_temp,
            system_pdc0_watts=cfg_system_pdc0_watts, temp_coeff_pdc=cfg_temp_coeff_pdc
        )

        # 1. Create a DataFrame from the calculated power series
        final_df = pd.DataFrame({output_target_column_name: power_series})
        
        # 2. Add the new time features
        final_df = add_cyclical_time_features(final_df)
        
        # 3. Reorder columns: target variable first, then features
        feature_cols = ['hour_sin', 'hour_cos', 'day_of_year_sin', 'day_of_year_cos']
        final_df = final_df[[output_target_column_name] + feature_cols]
        
        # 4. Save the new multi-feature DataFrame to CSV
        Path(output_preprocessed_file).parent.mkdir(parents=True, exist_ok=True)
        final_df.to_csv(output_preprocessed_file)
        print(f"\nGenerated energy WITH TIME FEATURES saved to: {output_preprocessed_file}")

        print("\nSample of the new data with features:")
        print(final_df.head())

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        print(traceback.format_exc())