"""
build_consumption_data.py
=========================
Reads the raw electricity consumption/cost CSV (NYC open-data format),
disaggregates the monthly totals into synthetic weekly data with realistic
random variation, and writes two preprocessed CSVs:

    data/preprocessed/energy_consumed.csv
    data/preprocessed/energy_cost.csv

Raw input path is read from config.yaml → raw_data.consumption_csv.
Run from the project root:

    python scripts/build_consumption_data.py
"""

import sys
from pathlib import Path
import os
import pandas as pd
import numpy as np
from datetime import timedelta

# Allow project-root imports (utils, models, …) regardless of CWD
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utils.config_loader import load_config

CONFIG = load_config()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_consumption_data(file_path: str) -> pd.DataFrame:
    """Load the raw monthly electricity CSV and return a grouped DataFrame.

    Expects columns: 'Revenue Month', 'Current Charges', 'Consumption (KWH)'.
    Rows are summed by date to collapse any duplicates from multiple accounts.

    Args:
        file_path: Path to the raw CSV file.

    Returns:
        DataFrame indexed by datetime with columns 'Cost ($)' and
        'Consumption (KWH)', aggregated to monthly frequency.
    """
    df = pd.read_csv(file_path)
    df = df[["Revenue Month", "Current Charges", "Consumption (KWH)"]]
    df = df.rename(columns={"Revenue Month": "Datetime", "Current Charges": "Cost ($)"})
    df = df.set_index("Datetime")
    df.index = pd.to_datetime(df.index)
    return df.groupby(df.index).sum()


# ---------------------------------------------------------------------------
# Weekly disaggregation
# ---------------------------------------------------------------------------

def create_weekly_series(monthly_data: pd.Series, start_date: str = "2010-01-01", variation: float = 0.15) -> pd.Series:
    """Disaggregate monthly totals into synthetic weekly values.

    Each month is split into 4 or 5 weeks. Weekly values are drawn by
    applying normally-distributed variation factors that sum to 1, so the
    weekly values preserve the monthly total in expectation.

    Args:
        monthly_data: Monthly totals as a Series.
        start_date: ISO date string for the first weekly data point.
        variation: Standard deviation of the per-week variation multiplier.
                   0.15 produces ±~15 % week-to-week fluctuation.

    Returns:
        Weekly Series starting at start_date with 7-day frequency.
    """
    np.random.seed(42)  # reproducibility
    weekly_data = []
    date_indices = []
    current_date = pd.to_datetime(start_date)

    for monthly_value in monthly_data:
        weeks = 4 + np.random.choice([0, 1])  # 4 or 5 weeks per month
        factors = np.random.normal(loc=1.0, scale=variation, size=weeks)
        factors = factors / factors.sum()  # normalise so they sum to 1
        for value in monthly_value * factors:
            weekly_data.append(value)
            date_indices.append(current_date)
            current_date += timedelta(days=7)

    return pd.Series(weekly_data, index=date_indices)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Run the full preprocessing pipeline and save the output CSVs."""
    raw_path = CONFIG["raw_data"]["consumption_csv"]
    output_dir = Path(CONFIG["data_paths"]["preprocessed_dir"])

    print(f"Loading raw consumption data from: {raw_path}")
    df = load_consumption_data(raw_path)

    weekly_consumed = create_weekly_series(df["Consumption (KWH)"])
    weekly_cost = create_weekly_series(df["Cost ($)"])

    os.makedirs(output_dir, exist_ok=True)
    consumed_path = output_dir / CONFIG["data_paths"]["energy_consumed_csv"]
    cost_path = output_dir / CONFIG["data_paths"]["energy_cost_csv"]

    weekly_consumed.to_frame("Consumption (KWH)").to_csv(consumed_path)
    weekly_cost.to_frame("Cost ($)").to_csv(cost_path)

    print(f"Created {len(weekly_consumed)} weekly data points.")
    print(f"Saved consumption data  → {consumed_path}")
    print(f"Saved cost data         → {cost_path}")
    return weekly_consumed, weekly_cost


if __name__ == "__main__":
    consumed, cost = main()
    print("\nSample — weekly consumption (head):")
    print(consumed.head())
    print("\nSample — weekly cost (head):")
    print(cost.head())
