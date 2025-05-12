import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from models.sarima_model import SARIMAModel
from utils.preprocessing import load_energy_data, prepare_train_test
from utils.evaluation import evaluate_forecast

import joblib

def train_and_evaluate(forecast_steps=52):
    # Load data
    cost, consumed = load_energy_data()
    
    # Split data (20% test by default)
    train_cost, test_cost = prepare_train_test(cost)
    train_consumed, test_consumed = prepare_train_test(consumed)
    
    # Initialize model
    sarima = SARIMAModel(order=(1,1,1), seasonal_order=(1,1,0,52))
    
    # Process Cost Data
    cost_model = sarima.train(train_cost)
    joblib.dump(cost_model, "models/sarima_cost_model.pkl")
    cost_forecast, cost_conf_int = sarima.forecast(cost_model, steps=forecast_steps)
    
    # Process Consumption Data
    consumed_model = sarima.train(train_consumed)
    joblib.dump(consumed_model, "models/sarima_consumed_model.pkl")
    consumed_forecast, consumed_conf_int = sarima.forecast(consumed_model, steps=forecast_steps)
    
    # Evaluate (only if test set is long enough)
    if len(test_cost) >= forecast_steps:
        print("\nEnergy Cost Metrics:")
        print(evaluate_forecast(test_cost[:forecast_steps], cost_forecast))
    
    if len(test_consumed) >= forecast_steps:
        print("\nEnergy Consumption Metrics:")
        print(evaluate_forecast(test_consumed[:forecast_steps], consumed_forecast))

    print("\nTraining completed.")

if __name__ == "__main__":
    train_and_evaluate(forecast_steps=52)  # Change this value to control forecast length