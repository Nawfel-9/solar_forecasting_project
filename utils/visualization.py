import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import joblib
import pandas as pd
import torch
from typing import Dict, Any

from utils.preprocessing import create_lstm_sequences, fit_scaler, scale_data, inverse_scale, get_device, load_data
from models.lstm_model import LSTM


def reset_losses(train_losses, val_losses, test_losses):
    train_losses.clear()
    val_losses.clear()
    test_losses.clear()

def forecast_series(model, input_seq, forecast_steps):
    model.eval()
    predictions = []
    current_seq = input_seq.clone()

    for _ in range(forecast_steps):
        with torch.no_grad():
            output = model(current_seq)
            next_val = output  # ✅ already 2D: (batch_size, 1)
            predictions.append(next_val.item())

            # Append the new value and drop the first step
            current_seq = torch.cat((current_seq[:, 1:, :], next_val.unsqueeze(1)), dim=1)

    return predictions

def plot_multiple_forecasts(historical, forecasts):
    """Plot mixed-frequency forecasts"""
    fig, axs = plt.subplots(3, 1, figsize=(15, 12))
    
    # Energy Generated (Hourly)
    axs[0].plot(historical['generated'].index, historical['generated'], label='Historical')
    axs[0].plot(forecasts['generated']['dates'], forecasts['generated']['values'], 'r-', label='Hourly Forecast')
    axs[0].set_title('Energy Generation (Hourly)')
    
    # Energy Cost (Weekly)
    axs[1].plot(historical['cost'].index, historical['cost'], 'g-', label='Historical')
    axs[1].plot(forecasts['cost']['dates'], forecasts['cost']['values'], 'r-', label='Weekly Forecast')
    axs[1].set_title('Energy Cost (Weekly)')
    
    # Energy Consumed (Weekly)
    axs[2].plot(historical['consumed'].index, historical['consumed'], 'b-', label='Historical')
    axs[2].plot(forecasts['consumed']['dates'], forecasts['consumed']['values'], 'r-', label='Weekly Forecast')
    axs[2].set_title('Energy Consumption (Weekly)')
    
    for ax in axs:
        ax.grid(True)
        ax.legend()
        ax.set_xlabel('Date')
    
    plt.tight_layout()
    return fig

def generate_forecasts(
    forecast_horizon: str = "1 Week",
    models: dict = None
) -> Dict[str, Dict[str, Any]]:
    """Generate all forecasts with proper confidence interval handling"""
    if models is None:
        models = load_models()
        
    device = get_device()
    
    # Convert horizon to steps
    horizon_map = {
        "1 Week": {"hourly": 24*7, "weekly": 1},
        "2 Weeks": {"hourly": 24*14, "weekly": 2},
        "1 Month": {"hourly": 24*30, "weekly": 4}
    }
    steps = horizon_map[forecast_horizon]
    
    data = load_data()
    forecasts = {}
    
    # 1. Energy Generated (LSTM)
    scaler = fit_scaler(data['generated'])
    scaled_data = scale_data(data['generated'], scaler)
    X, _ = create_lstm_sequences(scaled_data, time_steps=24*30)
    
    last_input = torch.tensor(X[-1], dtype=torch.float32).view(1, -1, 1).to(device)
    scaled_forecast = models['lstm'].forecast_series(models['lstm'], last_input, steps["hourly"])
    
    forecasts['generated'] = {
        'values': inverse_scale(np.array(scaled_forecast), scaler),
        'dates': pd.date_range(start=data['generated'].index[-1] + pd.Timedelta(hours=1), 
                             periods=steps["hourly"], freq='H')
    }
    
    # 2. Energy Cost (SARIMA) - Modified to remove update()
    cost_forecast = models['sarima_cost'].get_forecast(steps=steps["weekly"])

    forecasts['cost'] = {
        'values': cost_forecast.predicted_mean.values,
        'dates': cost_forecast.predicted_mean.index,
        'conf_int': cost_forecast.conf_int()  # Maintain as DataFrame
    }

    # 3. Energy Consumed (SARIMA) - Modified to remove update()
    consumed_forecast = models['sarima_consumed'].get_forecast(steps=steps["weekly"])

    forecasts['consumed'] = {
        'values': consumed_forecast.predicted_mean.values,
        'dates': consumed_forecast.predicted_mean.index,
        'conf_int': consumed_forecast.conf_int()  # Maintain as DataFrame
    }

    return forecasts