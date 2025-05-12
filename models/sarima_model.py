from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings("ignore")

class SARIMAModel:
    def __init__(self, order=(1,1,0), seasonal_order=(1,1,0,52)):
        self.order = order
        self.seasonal_order = seasonal_order
    
    def train(self, series):
        """Train SARIMA model on time series data"""
        model = SARIMAX(series,
                       order=self.order,
                       seasonal_order=self.seasonal_order,
                       enforce_stationarity=False)
        return model.fit(disp=False)
    
    def forecast(self, model, steps=52):
        """Generate forecasts with configurable steps"""
        print(f"Generating {steps}-step forecast")  # Debug output
        forecast = model.get_forecast(steps=steps)
        return forecast.predicted_mean, forecast.conf_int()