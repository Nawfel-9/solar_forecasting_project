# solar_forecasting_project/utils/visualisation.py
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np # For type hinting
from typing import Dict, Any, Optional, List
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Configuration loading
try:
    from utils.config_loader import load_config
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from utils.config_loader import load_config

CONFIG = load_config()

def _get_plotting_config(key: str, default: Any) -> Any:
    """Helper to safely get plotting configurations."""
    return CONFIG.get('plotting_config', {}).get(key, default)

def _get_color(config_key: str, default_color: str) -> str:
    """Helper to get colors from config, with a fallback default."""
    return CONFIG.get('plotting_colors', {}).get(config_key, default_color)

def plot_forecasts_matplotlib(
    historical_data: Dict[str, pd.Series],
    forecast_data: Dict[str, Dict[str, Any]],
    aligned_hourly_overlay: Optional[pd.DataFrame] = None 
    ) -> plt.Figure:
    """
    Plots historical data and forecasts for generation, cost, and consumption using Matplotlib.

    Args:
        historical_data: {'generated': Series, 'cost': Series, 'consumed': Series}
        forecast_data: {'generated': {'dates': DatetimeIndex, 'values': np.array, 'conf_int': DataFrame (opt)}, ...}
        aligned_hourly_overlay: Optional DataFrame with 'timestamp' and 'consumption_kwh' (hourly aligned)
                                 to overlay on the generation plot for demand comparison.
    Returns:
        plt.Figure: The Matplotlib figure object.
    """
    fig, axs = plt.subplots(3, 1, 
                            figsize=_get_plotting_config('figure_size_mpl', (14, 11)), 
                            sharex=False) # Initially False, can be True if all x-axes are identical
    fig.tight_layout(pad=_get_plotting_config('tight_layout_pad_mpl', 4.5))

    lw_hist = _get_plotting_config('linewidth_hist_mpl', 1.5)
    lw_fcst = _get_plotting_config('linewidth_fcst_mpl', 1.8)
    ls_fcst = _get_plotting_config('linestyle_fcst_mpl', '--')
    
    date_formatter = mdates.DateFormatter(_get_plotting_config('date_format_mpl', '%Y-%m-%d %Hh')) # More detailed for hourly

    # --- 1. Energy Generation Plot ---
    ax_gen = axs[0]
    hist_gen = historical_data.get('generated')
    fc_gen = forecast_data.get('generated', {})
    
    if hist_gen is not None and not hist_gen.empty:
        ax_gen.plot(hist_gen.index, hist_gen.values, label='Historical Generation', 
                    color=_get_color('historical_generated', 'blue'), linewidth=lw_hist)
    if fc_gen.get('dates') is not None and len(fc_gen.get('values', [])) > 0:
        ax_gen.plot(fc_gen['dates'], fc_gen['values'], label='Forecast Generation', 
                    color=_get_color('forecast_generated', 'orange'), linewidth=lw_fcst, linestyle=ls_fcst)
    
    if aligned_hourly_overlay is not None and 'consumption_kwh' in aligned_hourly_overlay.columns:
        if not aligned_hourly_overlay.empty and 'timestamp' in aligned_hourly_overlay.columns:
            ax_gen.plot(aligned_hourly_overlay['timestamp'], aligned_hourly_overlay['consumption_kwh'],
                        label='Forecast Aligned Consumption (Hourly)', 
                        color=_get_color('overlay_consumption', 'grey'), 
                        linewidth=1.0, linestyle=':', alpha=0.7)

    ax_gen.set_title(_get_plotting_config('title_gen_mpl', 'Solar Energy Generation Forecast (kW)'), fontsize=14)
    ax_gen.set_ylabel(_get_plotting_config('ylabel_gen_mpl', 'Power (kW)'), fontsize=12)
    ax_gen.grid(_get_plotting_config('grid_visible_mpl', True), 
                linestyle=_get_plotting_config('grid_style_mpl', ':'), alpha=0.7)
    ax_gen.legend(fontsize=_get_plotting_config('legend_fontsize_mpl', 10))
    ax_gen.xaxis.set_major_formatter(date_formatter)

    # --- 2. Electricity Cost Plot ---
    ax_cost = axs[1]
    hist_cost = historical_data.get('cost')
    fc_cost = forecast_data.get('cost', {})

    if hist_cost is not None and not hist_cost.empty:
        ax_cost.plot(hist_cost.index, hist_cost.values, label='Historical Cost', 
                     color=_get_color('historical_cost', 'green'), linewidth=lw_hist)
    if fc_cost.get('dates') is not None and len(fc_cost.get('values', [])) > 0:
        fc_dates_cost = pd.DatetimeIndex(fc_cost['dates']) # Ensure DatetimeIndex
        ax_cost.plot(fc_dates_cost, fc_cost['values'], label='Forecast Cost', 
                     color=_get_color('forecast_cost', 'red'), linewidth=lw_fcst, linestyle=ls_fcst)
        if fc_cost.get('conf_int') is not None and not fc_cost['conf_int'].empty:
            conf_int_cost = fc_cost['conf_int']
            ax_cost.fill_between(fc_dates_cost, conf_int_cost.iloc[:, 0], conf_int_cost.iloc[:, 1],
                                 color=_get_color('conf_int_cost', 'pink'), alpha=0.4, label='95% CI')
    ax_cost.set_title(_get_plotting_config('title_cost_mpl', 'Electricity Cost Forecast ($/kWh)'), fontsize=14)
    ax_cost.set_ylabel(_get_plotting_config('ylabel_cost_mpl', 'Cost ($/kWh)'), fontsize=12)
    ax_cost.grid(_get_plotting_config('grid_visible_mpl', True), linestyle=':', alpha=0.7)
    ax_cost.legend(fontsize=10)
    ax_cost.xaxis.set_major_formatter(mdates.DateFormatter(_get_plotting_config('date_format_weekly_mpl', '%Y-%m-%d'))) # Simpler for weekly

    # --- 3. Electricity Consumption Plot ---
    ax_cons = axs[2]
    hist_cons = historical_data.get('consumed')
    fc_cons = forecast_data.get('consumed', {})

    if hist_cons is not None and not hist_cons.empty:
        ax_cons.plot(hist_cons.index, hist_cons.values, label='Historical Consumption', 
                     color=_get_color('historical_consumed', 'purple'), linewidth=lw_hist)
    if fc_cons.get('dates') is not None and len(fc_cons.get('values', [])) > 0:
        fc_dates_cons = pd.DatetimeIndex(fc_cons['dates']) # Ensure DatetimeIndex
        ax_cons.plot(fc_dates_cons, fc_cons['values'], label='Forecast Consumption', 
                     color=_get_color('forecast_consumed', 'brown'), linewidth=lw_fcst, linestyle=ls_fcst)
        if fc_cons.get('conf_int') is not None and not fc_cons['conf_int'].empty:
            conf_int_cons = fc_cons['conf_int']
            ax_cons.fill_between(fc_dates_cons, conf_int_cons.iloc[:, 0], conf_int_cons.iloc[:, 1],
                                 color=_get_color('conf_int_consumed', 'tan'), alpha=0.4, label='95% CI')
    ax_cons.set_title(_get_plotting_config('title_cons_mpl', 'Electricity Consumption Forecast (kWh)'), fontsize=14)
    ax_cons.set_ylabel(_get_plotting_config('ylabel_cons_mpl', 'Energy (kWh)'), fontsize=12)
    ax_cons.set_xlabel(_get_plotting_config('xlabel_mpl', 'Date'), fontsize=12)
    ax_cons.grid(_get_plotting_config('grid_visible_mpl', True), linestyle=':', alpha=0.7)
    ax_cons.legend(fontsize=10)
    ax_cons.xaxis.set_major_formatter(mdates.DateFormatter(_get_plotting_config('date_format_weekly_mpl', '%Y-%m-%d')))

    # Rotate x-axis labels for readability
    for ax in axs:
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        
    return fig

def plot_residuals_autocorrelation(residuals: pd.Series, lags: int = 40, model_name: str = "") -> Optional[plt.Figure]:
    """Plots residuals and their autocorrelation (ACF)."""
    if residuals.empty:
        print(f"Residuals series for {model_name} is empty, skipping plot.")
        return None
    
    from statsmodels.graphics.tsaplots import plot_acf # Local import

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=_get_plotting_config('figure_size_residuals_mpl', (10, 8)))
    fig.suptitle(f'{model_name} Residual Analysis'.strip(), fontsize=15)
    fig.tight_layout(pad=4.0, rect=[0, 0, 1, 0.95]) # Adjust rect for suptitle

    residuals.plot(ax=ax1, title="Residuals Over Time", color=_get_color('residuals_line', 'grey'))
    ax1.set_ylabel("Residual Value")
    ax1.grid(True, linestyle=':', alpha=0.7)

    plot_acf(residuals.dropna(), lags=min(lags, len(residuals.dropna())//2 - 1), ax=ax2, 
             title="Autocorrelation of Residuals",
             color=_get_color('acf_bar', 'steelblue'), 
             vlines_kwargs={'colors': [_get_color('acf_vline', 'steelblue')]})
    ax2.set_xlabel("Lag")
    ax2.grid(True, linestyle=':', alpha=0.7)
    
    return fig

def plot_lstm_training_history(history: Dict[str, List[float]], title: str = "LSTM Model Training History") -> Optional[plt.Figure]:
    """Plots LSTM training and (optional) validation loss."""
    if not history or not any(history.values()):
        print("Training history is empty, skipping plot.")
        return None

    fig, ax = plt.subplots(figsize=_get_plotting_config('figure_size_history_mpl', (10, 6)))
    
    if history.get('train_loss'):
        ax.plot(history['train_loss'], label='Training Loss', 
                color=_get_color('train_loss_line', 'blue'), linewidth=1.5)
    if history.get('val_loss'): # If validation loss is tracked
        ax.plot(history['val_loss'], label='Validation Loss', 
                color=_get_color('val_loss_line', 'orange'), linewidth=1.5)
    
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.legend(fontsize=10)
    
    return fig

def _get_color(config_key: str, default_color: str) -> str:
    """Helper to get colors from config for plotting."""
    return CONFIG.get('plotting_colors', {}).get(config_key, default_color)

def create_plotly_forecast_chart(
    historical_data: Dict[str, pd.Series],
    forecast_data: Dict[str, Dict[str, Any]],
    user_system_generation_forecast_kw: Optional[pd.Series] = None, # <<< THIS PARAMETER MUST BE HERE
    aligned_hourly_overlay: Optional[pd.DataFrame] = None
    ) -> go.Figure:
    """Creates an interactive Plotly chart for forecasts."""
    # ---- TEMPORARY DEBUG PRINT ----
    # print("DEBUG: create_plotly_forecast_chart from VISUALISATION.PY is being called!")
    # if user_system_generation_forecast_kw is not None:
    #     print(f"DEBUG (vis.py): Received user_system_generation_forecast_kw head: {user_system_generation_forecast_kw.head()}")
    # else:
    #     print("DEBUG (vis.py): user_system_generation_forecast_kw is None.")
    # ---- END TEMPORARY DEBUG PRINT ----

    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=("Solar Energy Generation (kW) - User's System",
                        "Electricity Cost ($/kWh) - Forecasted",
                        "Energy Consumption (kWh) - Forecasted"),
        vertical_spacing=0.12,
        shared_xaxes=True
    )
    # ... (rest of the function logic as provided previously, plotting hist_gen_user_scaled and user_system_generation_forecast_kw) ...
    # ... (plotting cost and consumption) ...
    # ... (fig.update_layout and return fig) ...
    # (Make sure this is the full, corrected function body from my previous response)

    # Generation Plot
    hist_gen_user_scaled = historical_data.get('generated') 
    if hist_gen_user_scaled is not None and not hist_gen_user_scaled.empty:
        fig.add_trace(go.Scatter(x=hist_gen_user_scaled.index, y=hist_gen_user_scaled.values, mode='lines',
                                name="Hist. Gen. (User's System Scale)",
                                line=dict(color=_get_color('historical_generated_user_system', 'cornflowerblue'))), row=1, col=1)

    if user_system_generation_forecast_kw is not None and not user_system_generation_forecast_kw.empty:
        fig.add_trace(go.Scatter(x=user_system_generation_forecast_kw.index, y=user_system_generation_forecast_kw.values,
                                mode='lines', name="Fcst. Gen. (User's System)",
                                line=dict(color=_get_color('forecast_generated_user_system', 'darkorange'), dash='dash')), row=1, col=1)
    elif forecast_data.get('generated', {}).get('dates') is not None and len(forecast_data.get('generated', {}).get('values', [])) > 0:
        fc_gen_ref = forecast_data.get('generated', {})
        fig.add_trace(go.Scatter(x=fc_gen_ref['dates'], y=fc_gen_ref['values'], mode='lines', name='Fcst. Gen. (Ref System)',
                                line=dict(color=_get_color('forecast_generated', 'orange'), dash='dash')), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name='No Generation Forecast'), row=1, col=1)

    if aligned_hourly_overlay is not None and 'consumption_kwh' in aligned_hourly_overlay.columns:
        if not aligned_hourly_overlay.empty and 'timestamp' in aligned_hourly_overlay.columns:
            fig.add_trace(go.Scatter(x=aligned_hourly_overlay['timestamp'], y=aligned_hourly_overlay['consumption_kwh'],
                                    mode='lines', name='Fcst. Aligned Cons. (Hourly)',
                                    line=dict(color=_get_color('overlay_consumption', 'grey'), dash='dot', width=1.5)), row=1, col=1)
    # Cost
    hist_cost = historical_data.get('cost')
    fc_cost = forecast_data.get('cost', {})
    if hist_cost is not None and not hist_cost.empty:
        fig.add_trace(go.Scatter(x=hist_cost.index, y=hist_cost.values, mode='lines', name='Hist. Cost',
                                line=dict(color=_get_color('historical_cost', 'green'))), row=2, col=1)
    if fc_cost.get('dates') is not None and len(fc_cost.get('values', [])) > 0:
        fc_dates_cost = pd.DatetimeIndex(fc_cost['dates'])
        fig.add_trace(go.Scatter(x=fc_dates_cost, y=fc_cost['values'], mode='lines', name='Fcst. Cost',
                                line=dict(color=_get_color('forecast_cost', 'red'), dash='dash')), row=2, col=1)
        if fc_cost.get('conf_int') is not None and not fc_cost['conf_int'].empty and len(fc_cost['conf_int']) == len(fc_dates_cost):
            conf_int_cost = fc_cost['conf_int']
            fig.add_trace(go.Scatter(x=fc_dates_cost, y=conf_int_cost.iloc[:, 0], mode='lines', line_width=0, showlegend=False), row=2, col=1)
            fig.add_trace(go.Scatter(x=fc_dates_cost, y=conf_int_cost.iloc[:, 1], mode='lines', line_width=0, fill='tonexty',
                                    fillcolor=_get_color('conf_int_cost_fill_plotly', 'rgba(255,0,0,0.1)'), name='Cost 95% CI'), row=2, col=1)
    # Consumption
    hist_cons = historical_data.get('consumed')
    fc_cons = forecast_data.get('consumed', {})
    if hist_cons is not None and not hist_cons.empty:
        fig.add_trace(go.Scatter(x=hist_cons.index, y=hist_cons.values, mode='lines', name='Hist. Cons.',
                                line=dict(color=_get_color('historical_consumed', 'purple'))), row=3, col=1)
    if fc_cons.get('dates') is not None and len(fc_cons.get('values', [])) > 0:
        fc_dates_cons = pd.DatetimeIndex(fc_cons['dates'])
        fig.add_trace(go.Scatter(x=fc_dates_cons, y=fc_cons['values'], mode='lines', name='Fcst. Cons.',
                                line=dict(color=_get_color('forecast_consumed', 'brown'), dash='dash')), row=3, col=1)
        if fc_cons.get('conf_int') is not None and not fc_cons['conf_int'].empty and len(fc_cons['conf_int']) == len(fc_dates_cons):
            conf_int_cons = fc_cons['conf_int']
            fig.add_trace(go.Scatter(x=fc_dates_cons, y=conf_int_cons.iloc[:, 0], mode='lines', line_width=0, showlegend=False), row=3, col=1)
            fig.add_trace(go.Scatter(x=fc_dates_cons, y=conf_int_cons.iloc[:, 1], mode='lines', line_width=0, fill='tonexty',
                                    fillcolor=_get_color('conf_int_consumed_fill_plotly', 'rgba(165,42,42,0.1)'), name='Cons. 95% CI'), row=3, col=1)

    fig.update_layout(height=750, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), hovermode="x unified", margin=dict(l=40, r=20, t=60, b=20))
    fig.update_xaxes(tickformat='%Y-%m-%d %Hh')
    fig.update_yaxes(title_text="Power (kW)", row=1, col=1)
    fig.update_yaxes(title_text="Price ($/kWh)", row=2, col=1)
    fig.update_yaxes(title_text="Energy (kWh)", row=3, col=1)
    return fig