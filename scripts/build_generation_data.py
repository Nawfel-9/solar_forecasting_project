"""
build_generation_data.py
========================
Estimates hourly DC solar power output from raw meteorological/irradiance
data using PVLib, enriches it with cyclical time features, and writes the
result to:

    data/preprocessed/energy_generated.csv

All physical constants (location, panel specs, column names) are read from
config.yaml → generation_estimation.  The raw input CSV path comes from
config.yaml → raw_data.generation_csv.

Run from the project root:

    python scripts/build_generation_data.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pvlib

# Allow project-root imports (utils, models, …) regardless of CWD
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utils.config_loader import load_config
from utils.preprocessing import add_cyclical_time_features

CONFIG = load_config()


# ---------------------------------------------------------------------------
# Core physics
# ---------------------------------------------------------------------------

def estimate_hourly_dc_power(
    df_input_data: pd.DataFrame,
    ambient_temp_col: str,
    ghi_col: str,
    dni_col: str,
    dhi_col: str,
    solar_zenith_col: str,
    latitude: float,
    longitude: float,
    surface_tilt: float,
    albedo: float,
    noct_module_temp: float,
    system_pdc0_watts: float,
    temp_coeff_pdc: float,
) -> pd.Series:
    """Compute hourly DC power output (W) for a PV system using PVLib.

    Uses the PVWatts DC model with SAPM cell temperature and the Perez
    diffuse irradiance decomposition.

    Args:
        df_input_data: DataFrame with a DatetimeIndex (or Year/Month/Day/Hour
            columns from which one can be constructed).
        ambient_temp_col: Column name for ambient temperature (°C).
        ghi_col: Column name for Global Horizontal Irradiance (W/m²).
        dni_col: Column name for Direct Normal Irradiance (W/m²).
        dhi_col: Column name for Diffuse Horizontal Irradiance (W/m²).
        solar_zenith_col: Column name for solar zenith angle (degrees).
        latitude: Site latitude (decimal degrees, positive = North).
        longitude: Site longitude (decimal degrees, positive = East).
        surface_tilt: Panel tilt from horizontal (degrees).
        albedo: Ground reflectance (dimensionless, 0–1).
        noct_module_temp: Nominal Operating Cell Temperature (°C).
        system_pdc0_watts: System DC peak power at STC (W).
        temp_coeff_pdc: Power temperature coefficient (%/°C, typically negative).

    Returns:
        Series of non-negative hourly DC power values (W), indexed as the input.
    """
    df = df_input_data.copy()

    # Normalise the index to DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        if {"Year", "Month", "Day", "Hour"}.issubset(df.columns):
            date_cols = ["Year", "Month", "Day", "Hour"]
            if "Minute" in df.columns:
                date_cols.append("Minute")
            df.index = pd.to_datetime(df[date_cols])
            df.index.name = "datetime"
        else:
            df.index = pd.to_datetime(df.index)

    # Rename raw columns to standard names for clarity
    rename_map = {
        ambient_temp_col: "Ambient Temperature",
        ghi_col: "GHI",
        dni_col: "DNI",
        dhi_col: "DHI",
        solar_zenith_col: "Solar Zenith Angle",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Solar position
    solpos = pvlib.solarposition.get_solarposition(df.index, latitude, longitude)

    # Plane-of-array irradiance (surface facing true south, azimuth=180°)
    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=surface_tilt,
        surface_azimuth=180,
        solar_zenith=solpos["zenith"],
        solar_azimuth=solpos["azimuth"],
        dni=df["DNI"],
        ghi=df["GHI"],
        dhi=df["DHI"],
        albedo=albedo,
    ).fillna(0)

    # Cell temperature via SAPM model (wind speed assumed constant at 1 m/s)
    cell_temp = pvlib.temperature.sapm_cell(
        poa["poa_global"],
        df["Ambient Temperature"],
        wind_speed=1,
        a=-3.56,
        b=-0.075,
        deltaT=3,
    )

    # DC power via PVWatts
    power_dc = pvlib.pvsystem.pvwatts_dc(
        g_poa_effective=poa["poa_global"],
        temp_cell=cell_temp,
        pdc0=system_pdc0_watts,
        gamma_pdc=temp_coeff_pdc,
    ).fillna(0)

    return np.maximum(0, power_dc)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Run the full generation-estimation pipeline and save the output CSV."""
    cfg = CONFIG["generation_estimation"]
    raw_path = CONFIG["raw_data"]["generation_csv"]
    output_path = (
        Path(CONFIG["data_paths"]["preprocessed_dir"])
        / CONFIG["data_paths"]["energy_generated_csv"]
    )
    target_col = cfg["output_target_column_name"]

    print(f"Loading raw irradiance data from: {raw_path}")
    df_raw = pd.read_csv(raw_path)

    print("Estimating hourly DC power output…")
    power_series = estimate_hourly_dc_power(
        df_input_data=df_raw,
        ambient_temp_col=cfg["ambient_temp_col"],
        ghi_col=cfg["ghi_col"],
        dni_col=cfg["dni_col"],
        dhi_col=cfg["dhi_col"],
        solar_zenith_col=cfg["solar_zenith_col"],
        latitude=cfg["latitude"],
        longitude=cfg["longitude"],
        surface_tilt=cfg["surface_tilt"],
        albedo=cfg["albedo"],
        noct_module_temp=cfg["noct_module_temp"],
        system_pdc0_watts=cfg["system_pdc0_watts"],
        temp_coeff_pdc=cfg["temp_coeff_pdc"],
    )

    # Build the multi-feature DataFrame expected by the LSTM
    final_df = pd.DataFrame({target_col: power_series})
    final_df = add_cyclical_time_features(final_df)

    # Column order: target first, then time features
    feature_cols = ["hour_sin", "hour_cos", "day_of_year_sin", "day_of_year_cos"]
    final_df = final_df[[target_col] + feature_cols]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_path)
    print(f"Saved generation data ({len(final_df)} rows, {final_df.shape[1]} columns) → {output_path}")

    print("\nSample output (head):")
    print(final_df.head())


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        import traceback
        print(f"Error: {exc}")
        traceback.print_exc()
