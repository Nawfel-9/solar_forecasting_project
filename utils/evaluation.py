from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def evaluate_forecast(true: np.ndarray, pred: np.ndarray) -> dict:
    """Calculate multiple evaluation metrics"""
    return {
        'MAE': mean_absolute_error(true, pred),
        'RMSE': np.sqrt(mean_squared_error(true, pred)),
        'MAPE': np.mean(np.abs((true - pred) / true)) * 100,
        "R2": r2_score(true, pred)
    }

def print_metrics(metrics: dict, name: str):
    """Print formatted metrics"""
    print(f'\n{name} Forecast Metrics:')
    for metric, value in metrics.items():
        print(f'{metric}: {value:.2f}')

def calculate_summary_metrics(aligned_data):
    """Calculate summary metrics for dashboard"""
    if aligned_data.empty:
        return {
            "peak_generation": 0,
            "avg_generation": 0,
            "total_consumption": 0,
            "avg_cost": 0,
            "savings_estimate": 0
        }
    
    metrics = {
        "peak_generation": aligned_data['generation_kw'].max(),
        "avg_generation": aligned_data['generation_kw'].mean(),
        "total_consumption": aligned_data['consumption_kwh'].sum(),
        "avg_cost": aligned_data['cost_per_kwh'].mean()
    }
    
    # Calculate potential savings (simple estimate)
    total_generated = aligned_data['generation_kw'].sum()
    avg_cost = metrics["avg_cost"]
    metrics["savings_estimate"] = total_generated * avg_cost
    
    return metrics