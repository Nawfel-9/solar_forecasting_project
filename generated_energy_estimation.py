import numpy as np
import pandas as pd
import pvlib


def compute_solar_position(index: pd.DatetimeIndex,
                           latitude: float = 40.73,
                           longitude: float = -74.02) -> pd.DataFrame:
    """
    Compute solar elevation and azimuth for each timestamp in the index.
    """
    solar_positions = pvlib.solarposition.get_solarposition(index, latitude, longitude)
    df_pos = pd.DataFrame(
        {
            'Solar Azimuth': solar_positions['azimuth'],
            'Solar Zenith Angle': solar_positions['zenith'],
        },
        index=index
    )
    df_pos['Solar Elevation'] = 90 - df_pos['Solar Zenith Angle']
    return df_pos


def compute_poa_irradiance(df: pd.DataFrame,
                           beta: float = 5.0,
                           rho: float = 0.2) -> pd.Series:
    """
    Compute Plane-of-Array irradiance (Gpoa) using DNI, DHI, GHI,
    solar elevation and azimuth angles, tilt (beta), and albedo (rho).
    """
    elevation_rad = np.radians(df['Solar Elevation'])
    azimuth_rad = np.radians(df['Solar Azimuth'])
    ghi = df['GHI']
    dni = df['DNI']
    dhi = df['DHI']

    cos_aoi = (
        np.cos(np.radians(beta)) * np.sin(elevation_rad)
        + np.sin(np.radians(beta)) * np.cos(elevation_rad) * np.cos(azimuth_rad)
    )
    cos_aoi = np.clip(cos_aoi, 0, None)

    beam = dni * cos_aoi
    sky_diffuse = dhi * (1 + np.cos(np.radians(beta))) / 2
    ground_reflected = ghi * rho * (1 - np.cos(np.radians(beta))) / 2
    gpoa = beam + sky_diffuse + ground_reflected
    return gpoa


def compute_cell_temperature(gpoa: pd.Series,
                             ambient_temp: pd.Series,
                             noct: float = 45.0) -> pd.Series:
    """
    Estimate cell temperature using the NOCT model.
    """
    delta = (noct - 20) / 800.0
    return ambient_temp + delta * gpoa


def compute_dc_power(gpoa: pd.Series,
                     cell_temp: pd.Series,
                     pdc0: float,
                     gamma: float = -0.0037) -> pd.Series:
    """
    Compute DC power output (W) using PVWatts.
    """
    g_prime = gpoa / 1000.0
    t_prime = cell_temp - 25.0
    return pdc0 * g_prime * (1 + gamma * t_prime)


def estimate_generated_energy(df_generated: pd.DataFrame,
                              ambient_temp_col: str = 'Temperature',  # Changed from 'Ambient Temperature'
                              ghi_col: str = 'GHI',
                              dni_col: str = 'DNI',
                              dhi_col: str = 'DHI',
                              solar_zenith_col: str = 'Solar Zenith Angle',  # Added parameter
                              latitude: float = 40.73,
                              longitude: float = -74.02,
                              beta: float = 5.0,
                              rho: float = 0.2,
                              noct: float = 45.0,
                              pdc0: float = 750 * 10000,
                              gamma: float = -0.0037) -> pd.Series:
    """
    Full pipeline:
    1. Ensure index is datetime
    2. Rename irradiance/temp columns to standard names
    3. Compute solar position (elevation & azimuth)
    4. Compute Gpoa, cell temp, and DC power
    """
    df = df_generated.copy()
    
    # Combine separate date columns into datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        if {'Year', 'Month', 'Day', 'Hour'}.issubset(df.columns):
            # Check if Minute column exists and use it
            if 'Minute' in df.columns:
                df['datetime'] = pd.to_datetime(
                    df[['Year', 'Month', 'Day', 'Hour', 'Minute']].assign(
                        Hour=lambda x: x['Hour'] + x['Minute']/60
                    )[['Year', 'Month', 'Day', 'Hour']]
                )
            else:
                df['datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour']])
            df.set_index('datetime', inplace=True)
    
    df.index = pd.to_datetime(df.index)

    # Rename columns to standard names
    rename_dict = {}
    if ambient_temp_col in df.columns:
        rename_dict[ambient_temp_col] = 'Ambient Temperature'
    if ghi_col in df.columns:
        rename_dict[ghi_col] = 'GHI'
    if dni_col in df.columns:
        rename_dict[dni_col] = 'DNI'
    if dhi_col in df.columns:
        rename_dict[dhi_col] = 'DHI'
    
    df = df.rename(columns=rename_dict)

    # Check if we need to compute solar positions or if they're already provided
    if solar_zenith_col in df.columns:
        # Use provided solar zenith angles
        df['Solar Zenith Angle'] = df[solar_zenith_col]
        df['Solar Elevation'] = 90 - df['Solar Zenith Angle']
        
        # We need to estimate Solar Azimuth since it's not provided
        # For a simple estimate, use the pvlib calculation
        solar_positions = pvlib.solarposition.get_solarposition(df.index, latitude, longitude)
        df['Solar Azimuth'] = solar_positions['azimuth']
    else:
        # Compute solar angles
        df_pos = compute_solar_position(df.index, latitude, longitude)
        df = df.join(df_pos)

    # Make sure all required columns exist
    required_cols = ['GHI', 'DNI', 'DHI', 'Ambient Temperature', 'Solar Elevation', 'Solar Azimuth']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column {col} not found in dataframe")

    gpoa = compute_poa_irradiance(df, beta, rho)
    cell_temp = compute_cell_temperature(gpoa, df['Ambient Temperature'], noct)
    dc_power = compute_dc_power(gpoa, cell_temp, pdc0, gamma)
    return dc_power


if __name__ == "__main__":
    
    df_gen = pd.read_csv('data/generated_2009_2023.csv', parse_dates=True)
    # For actual file usage:
    # df_gen = pd.read_csv('data/generated_2009_2023.csv')
    
    power_series = estimate_generated_energy(
        df_gen, 
        ambient_temp_col='Temperature',
        solar_zenith_col='Solar Zenith Angle'
    )
    
    print("First few power values:")
    print(power_series.head())
    
    # Create a DataFrame with results for better inspection
    results_df = df_gen.copy()
    results_df['DC Power (W)'] = power_series.values
    
    print("\nFull results with power:")
    print(results_df.head())

    # Export the generated energy series to a CSV file
    output_file = 'data/preprocessed/energy_generated.csv'

    # Create a DataFrame with datetime index and power values
    power_df = pd.DataFrame({
        'Generated Energy (W)': power_series
    })

    # Export to CSV
    power_df.to_csv(output_file)
    print(f"\nGenerated energy series saved to: {output_file}")