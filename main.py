import torch
import joblib
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import utils.preprocessing as get_device

# Local imports
from utils.preprocessing import (
    create_lstm_sequences,
    fit_scaler,
    scale_data,
    inverse_scale,
    align_frequencies
)

from utils.visualization import plot_multiple_forecasts
from utils.evaluation import evaluate_forecast, print_metrics
from models.lstm_model import LSTM

def main():
    device = get_device()
    print(f"Using device: {device}")

    # Configuration
    hourly_steps = 24 * 30 * 1  # 1 month of hourly forecasts
    weekly_steps = 4  # 4 weeks forecast
    model_paths = {
        'generated': Path("models/best_model_energy_generated.pth"),
        'cost': Path("models/sarima_cost_model.pkl"),
        'consumed': Path("models/sarima_consumed_model.pkl")
    }
    print("\n[1/6] Loading data...")
    # Load raw data with different frequencies
    generated_hourly = pd.read_csv("data/preprocessed/energy_generated.csv", 
                                  index_col=0, parse_dates=True).squeeze()
    cost_weekly = pd.read_csv("data/preprocessed/energy_cost.csv", 
                             index_col=0, parse_dates=True).squeeze()
    consumed_weekly = pd.read_csv("data/preprocessed/energy_consumed.csv", 
                                 index_col=0, parse_dates=True).squeeze()
    
    print(f"Loaded data shapes - Generated: {generated_hourly.shape}, Cost: {cost_weekly.shape}, Consumed: {consumed_weekly.shape}")
    # Align frequencies

    print("\n[2/6] Aligning frequencies...")
    generated, cost, consumed = align_frequencies(generated_hourly, cost_weekly, consumed_weekly)
    
    print(f"Aligned shapes - Generated: {generated.shape}, Cost: {cost.shape}, Consumed: {consumed.shape}")
    # Generate forecasts
    forecasts = {
        'generated': {'freq': 'H', 'values': None, 'dates': None},
        'cost': {'freq': 'W', 'values': None, 'dates': None},
        'consumed': {'freq': 'W', 'values': None, 'dates': None}
    }
    
    # 1. Energy Generated (LSTM - Hourly)
    scaler = fit_scaler(generated)
    scaled_data = scale_data(generated, scaler)
    X, _ = create_lstm_sequences(scaled_data, time_steps=24*30*1)
    
    print("\n[3/6] Training/loading LSTM model...")
    model = LSTM().to(device)
    model.load_state_dict(torch.load(model_paths['generated'], map_location=device, weights_only=False)["model_state_dict"])
    last_input = torch.tensor(X[-1], dtype=torch.float32).view(1, -1, 1).to(device)
    scaled_forecast = model.forecast_series(model, last_input, hourly_steps)
    forecasts['generated']['values'] = inverse_scale(np.array(scaled_forecast), scaler)
    forecasts['generated']['dates'] = pd.date_range(start=generated.index[-1] + pd.Timedelta(hours=1), 
                                                   periods=hourly_steps, freq='h')
    
    print("\n[4/6] Generating forecasts...")
    # 2. Energy Cost (SARIMA - Weekly)
    sarima_cost = joblib.load(model_paths['cost'])
    cost_forecast = sarima_cost.get_forecast(steps=weekly_steps).predicted_mean
    forecasts['cost']['values'] = cost_forecast.values
    forecasts['cost']['dates'] = pd.date_range(start=cost_weekly.index[-1] + pd.Timedelta(weeks=1), 
                                             periods=weekly_steps, freq='W')
    
    # 3. Energy Consumed (SARIMA - Weekly)
    sarima_consumed = joblib.load(model_paths['consumed'])
    consumed_forecast = sarima_consumed.get_forecast(steps=weekly_steps).predicted_mean
    forecasts['consumed']['values'] = consumed_forecast.values
    forecasts['consumed']['dates'] = pd.date_range(start=consumed_weekly.index[-1] + pd.Timedelta(weeks=1), 
                                                 periods=weekly_steps, freq='W')
    
    print("\n[5/6] Preparing visualization...")
    # Visualization
    plot_multiple_forecasts(
        historical={
            'generated': generated,
            'cost': cost_weekly,
            'consumed': consumed_weekly
        },
        forecasts=forecasts
    )
    print("\n[6/6] Showing results...")
    # Replace plt.show() with:
    plt.savefig("results/forecast_results.png")
    print("Plot saved to forecast_results.png")

if __name__ == "__main__":
    main()