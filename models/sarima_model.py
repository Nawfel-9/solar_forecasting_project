# solar_forecasting_project/models/sarima_model.py
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults
import pandas as pd # For type hinting
from typing import Tuple, Optional # Added Optional

class SARIMAModel:
    """
    A wrapper for SARIMA (Seasonal AutoRegressive Integrated Moving Average) model
    from statsmodels.
    """
    def __init__(self, order: Tuple[int, int, int], seasonal_order: Tuple[int, int, int, int]):
        """
        Args:
            order (Tuple[int, int, int]): The (p, d, q) order of the model.
            seasonal_order (Tuple[int, int, int, int]): The (P, D, Q, s) seasonal order of the model.
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.model_fit: Optional[SARIMAXResults] = None # To store the fitted model

    def train(self, series: pd.Series, **kwargs) -> SARIMAXResults:
        """
        Trains the SARIMA model on the provided time series data.

        Args:
            series (pd.Series): The time series data (univariate) with a DatetimeIndex.
            **kwargs: Additional arguments to pass to SARIMAX.fit()
                      (e.g., disp=False, maxiter).

        Returns:
            SARIMAXResults: The fitted model object.
        """
        # Default fitting arguments, can be overridden by kwargs
        fit_kwargs = {'disp': False} 
        fit_kwargs.update(kwargs)

        model = SARIMAX(
            series,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False, # Often handled by differencing (d, D)
            enforce_invertibility=False # Can help convergence
        )
        self.model_fit = model.fit(**fit_kwargs)
        return self.model_fit
    
    def forecast(self, steps: int, fitted_model: Optional[SARIMAXResults] = None) -> Tuple[pd.Series, Optional[pd.DataFrame]]:
        """
        Generates forecasts for a specified number of future steps.

        Args:
            steps (int): The number of steps to forecast into the future.
            fitted_model (Optional[SARIMAXResults]): An already fitted SARIMAXResults object.
                                                     If None, uses the model_fit attribute from a previous train call.

        Returns:
            Tuple[pd.Series, Optional[pd.DataFrame]]:
                - pd.Series: The forecasted mean values, with a DatetimeIndex.
                - Optional[pd.DataFrame]: The confidence interval for the forecasts.
                                          Returns None if confidence intervals cannot be produced.
        """
        model_to_use = fitted_model if fitted_model is not None else self.model_fit

        if model_to_use is None:
            raise ValueError("Model has not been trained or a fitted model was not provided.")

        forecast_results = model_to_use.get_forecast(steps=steps)
        
        predicted_mean = forecast_results.predicted_mean
        conf_int = None
        try:
            conf_int = forecast_results.conf_int()
        except Exception as e:
            print(f"Could not retrieve confidence intervals: {e}")
            # This can happen for some models or edge cases, especially with exog.

        return predicted_mean, conf_int