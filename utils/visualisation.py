# utils/visualisation.py
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from plotly.subplots import make_subplots
import plotly.graph_objects as go

try:
    from utils.config_loader import load_config
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from utils.config_loader import load_config

CONFIG = load_config()


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _get_plotting_config(key: str, default: Any) -> Any:
    """Safely retrieve a value from the plotting_config section of config."""
    return CONFIG.get("plotting_config", {}).get(key, default)


def _get_color(config_key: str, default_color: str) -> str:
    """Retrieve a named colour from plotting_colors config, or fall back to default."""
    return CONFIG.get("plotting_colors", {}).get(config_key, default_color)


# ---------------------------------------------------------------------------
# Matplotlib plots
# ---------------------------------------------------------------------------

def plot_forecasts_matplotlib(
    historical_data: Dict[str, pd.Series],
    forecast_data: Dict[str, Dict[str, Any]],
    aligned_hourly_overlay: Optional[pd.DataFrame] = None,
) -> plt.Figure:
    """Plot historical data and forecasts for generation, cost, and consumption.

    Args:
        historical_data: Mapping of ``{'generated': Series, 'cost': Series,
            'consumed': Series}`` for historical traces.
        forecast_data: Mapping of ``{'generated': {'dates': DatetimeIndex,
            'values': np.array, 'conf_int': DataFrame (opt.)}, …}`` for
            forecast traces.
        aligned_hourly_overlay: Optional DataFrame with a ``'timestamp'`` column
            and a ``'consumption_kwh'`` column (hourly aligned) overlaid on the
            generation subplot for demand comparison.

    Returns:
        Matplotlib Figure with three vertically-stacked subplots.
    """
    fig, axs = plt.subplots(
        3, 1,
        figsize=_get_plotting_config("figure_size_mpl", (14, 11)),
        sharex=False,
    )
    fig.tight_layout(pad=_get_plotting_config("tight_layout_pad_mpl", 4.5))

    lw_hist = _get_plotting_config("linewidth_hist_mpl", 1.5)
    lw_fcst = _get_plotting_config("linewidth_fcst_mpl", 1.8)
    ls_fcst = _get_plotting_config("linestyle_fcst_mpl", "--")
    date_fmt = mdates.DateFormatter(_get_plotting_config("date_format_mpl", "%Y-%m-%d %Hh"))
    date_fmt_weekly = mdates.DateFormatter(_get_plotting_config("date_format_weekly_mpl", "%Y-%m-%d"))

    # --- 1. Energy Generation ---
    ax_gen = axs[0]
    hist_gen = historical_data.get("generated")
    fc_gen = forecast_data.get("generated", {})

    if hist_gen is not None and not hist_gen.empty:
        ax_gen.plot(hist_gen.index, hist_gen.values, label="Historical Generation",
                    color=_get_color("historical_generated", "blue"), linewidth=lw_hist)
    if fc_gen.get("dates") is not None and len(fc_gen.get("values", [])) > 0:
        ax_gen.plot(fc_gen["dates"], fc_gen["values"], label="Forecast Generation",
                    color=_get_color("forecast_generated", "orange"), linewidth=lw_fcst, linestyle=ls_fcst)
    if (aligned_hourly_overlay is not None
            and "consumption_kwh" in aligned_hourly_overlay.columns
            and not aligned_hourly_overlay.empty
            and "timestamp" in aligned_hourly_overlay.columns):
        ax_gen.plot(
            aligned_hourly_overlay["timestamp"],
            aligned_hourly_overlay["consumption_kwh"],
            label="Forecast Aligned Consumption (Hourly)",
            color=_get_color("overlay_consumption", "grey"),
            linewidth=1.0, linestyle=":", alpha=0.7,
        )

    ax_gen.set_title(_get_plotting_config("title_gen_mpl", "Solar Energy Generation Forecast (kW)"), fontsize=14)
    ax_gen.set_ylabel(_get_plotting_config("ylabel_gen_mpl", "Power (kW)"), fontsize=12)
    ax_gen.grid(_get_plotting_config("grid_visible_mpl", True),
                linestyle=_get_plotting_config("grid_style_mpl", ":"), alpha=0.7)
    ax_gen.legend(fontsize=_get_plotting_config("legend_fontsize_mpl", 10))
    ax_gen.xaxis.set_major_formatter(date_fmt)

    # --- 2. Electricity Cost ---
    ax_cost = axs[1]
    hist_cost = historical_data.get("cost")
    fc_cost = forecast_data.get("cost", {})

    if hist_cost is not None and not hist_cost.empty:
        ax_cost.plot(hist_cost.index, hist_cost.values, label="Historical Cost",
                     color=_get_color("historical_cost", "green"), linewidth=lw_hist)
    if fc_cost.get("dates") is not None and len(fc_cost.get("values", [])) > 0:
        fc_dates_cost = pd.DatetimeIndex(fc_cost["dates"])
        ax_cost.plot(fc_dates_cost, fc_cost["values"], label="Forecast Cost",
                     color=_get_color("forecast_cost", "red"), linewidth=lw_fcst, linestyle=ls_fcst)
        if fc_cost.get("conf_int") is not None and not fc_cost["conf_int"].empty:
            ci = fc_cost["conf_int"]
            ax_cost.fill_between(fc_dates_cost, ci.iloc[:, 0], ci.iloc[:, 1],
                                 color=_get_color("conf_int_cost", "pink"), alpha=0.4, label="95% CI")

    ax_cost.set_title(_get_plotting_config("title_cost_mpl", "Electricity Cost Forecast ($/kWh)"), fontsize=14)
    ax_cost.set_ylabel(_get_plotting_config("ylabel_cost_mpl", "Cost ($/kWh)"), fontsize=12)
    ax_cost.grid(_get_plotting_config("grid_visible_mpl", True), linestyle=":", alpha=0.7)
    ax_cost.legend(fontsize=10)
    ax_cost.xaxis.set_major_formatter(date_fmt_weekly)

    # --- 3. Electricity Consumption ---
    ax_cons = axs[2]
    hist_cons = historical_data.get("consumed")
    fc_cons = forecast_data.get("consumed", {})

    if hist_cons is not None and not hist_cons.empty:
        ax_cons.plot(hist_cons.index, hist_cons.values, label="Historical Consumption",
                     color=_get_color("historical_consumed", "purple"), linewidth=lw_hist)
    if fc_cons.get("dates") is not None and len(fc_cons.get("values", [])) > 0:
        fc_dates_cons = pd.DatetimeIndex(fc_cons["dates"])
        ax_cons.plot(fc_dates_cons, fc_cons["values"], label="Forecast Consumption",
                     color=_get_color("forecast_consumed", "brown"), linewidth=lw_fcst, linestyle=ls_fcst)
        if fc_cons.get("conf_int") is not None and not fc_cons["conf_int"].empty:
            ci = fc_cons["conf_int"]
            ax_cons.fill_between(fc_dates_cons, ci.iloc[:, 0], ci.iloc[:, 1],
                                 color=_get_color("conf_int_consumed", "tan"), alpha=0.4, label="95% CI")

    ax_cons.set_title(_get_plotting_config("title_cons_mpl", "Electricity Consumption Forecast (kWh)"), fontsize=14)
    ax_cons.set_ylabel(_get_plotting_config("ylabel_cons_mpl", "Energy (kWh)"), fontsize=12)
    ax_cons.set_xlabel(_get_plotting_config("xlabel_mpl", "Date"), fontsize=12)
    ax_cons.grid(_get_plotting_config("grid_visible_mpl", True), linestyle=":", alpha=0.7)
    ax_cons.legend(fontsize=10)
    ax_cons.xaxis.set_major_formatter(date_fmt_weekly)

    for ax in axs:
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    return fig


def plot_residuals_autocorrelation(residuals: pd.Series, lags: int = 40, model_name: str = "") -> Optional[plt.Figure]:
    """Plot residuals over time and their autocorrelation function (ACF).

    Args:
        residuals: Residual series from a fitted model.
        lags: Maximum number of ACF lags to display.
        model_name: Label used in the figure suptitle.

    Returns:
        Matplotlib Figure, or None if residuals is empty.
    """
    if residuals.empty:
        print(f"Residuals series for {model_name} is empty, skipping plot.")
        return None

    from statsmodels.graphics.tsaplots import plot_acf  # local import to avoid hard dep at module load

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=_get_plotting_config("figure_size_residuals_mpl", (10, 8)))
    fig.suptitle(f"{model_name} Residual Analysis".strip(), fontsize=15)
    fig.tight_layout(pad=4.0, rect=[0, 0, 1, 0.95])

    residuals.plot(ax=ax1, title="Residuals Over Time", color=_get_color("residuals_line", "grey"))
    ax1.set_ylabel("Residual Value")
    ax1.grid(True, linestyle=":", alpha=0.7)

    plot_acf(
        residuals.dropna(),
        lags=min(lags, len(residuals.dropna()) // 2 - 1),
        ax=ax2,
        title="Autocorrelation of Residuals",
        color=_get_color("acf_bar", "steelblue"),
        vlines_kwargs={"colors": [_get_color("acf_vline", "steelblue")]},
    )
    ax2.set_xlabel("Lag")
    ax2.grid(True, linestyle=":", alpha=0.7)

    return fig


def plot_lstm_training_history(history: Dict[str, List[float]], title: str = "LSTM Model Training History") -> Optional[plt.Figure]:
    """Plot LSTM training loss and (optionally) validation loss curves.

    Args:
        history: Dictionary with keys ``'train_loss'`` and optionally
            ``'val_loss'``, each a list of per-epoch loss values.
        title: Figure title.

    Returns:
        Matplotlib Figure, or None if history is empty.
    """
    if not history or not any(history.values()):
        print("Training history is empty, skipping plot.")
        return None

    fig, ax = plt.subplots(figsize=_get_plotting_config("figure_size_history_mpl", (10, 6)))

    if history.get("train_loss"):
        ax.plot(history["train_loss"], label="Training Loss",
                color=_get_color("train_loss_line", "blue"), linewidth=1.5)
    if history.get("val_loss"):
        ax.plot(history["val_loss"], label="Validation Loss",
                color=_get_color("val_loss_line", "orange"), linewidth=1.5)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.grid(True, linestyle=":", alpha=0.7)
    ax.legend(fontsize=10)

    return fig


# ---------------------------------------------------------------------------
# Plotly interactive chart
# ---------------------------------------------------------------------------

def create_plotly_forecast_chart(
    historical_data: Dict[str, pd.Series],
    forecast_data: Dict[str, Dict[str, Any]],
    user_system_generation_forecast_kw: Optional[pd.Series] = None,
    aligned_hourly_overlay: Optional[pd.DataFrame] = None,
) -> go.Figure:
    """Create an interactive three-panel Plotly chart for forecasts.

    Panel 1: Solar generation (historical + user-system forecast).
    Panel 2: Electricity cost (historical + SARIMA forecast with CI).
    Panel 3: Electricity consumption (historical + SARIMA forecast with CI).

    Args:
        historical_data: ``{'generated': Series, 'cost': Series, 'consumed': Series}``.
        forecast_data: ``{'generated': {'dates': …, 'values': …}, 'cost': …, 'consumed': …}``.
        user_system_generation_forecast_kw: Optional Series of the forecast already
            scaled to the user's system size and efficiency. When provided it
            replaces the raw reference-system forecast in panel 1.
        aligned_hourly_overlay: Optional DataFrame with ``'timestamp'`` and
            ``'consumption_kwh'`` columns, overlaid on panel 1 as a demand trace.

    Returns:
        Plotly Figure object.
    """
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(
            "Solar Energy Generation (kW) — User's System",
            "Electricity Cost ($/kWh) — Forecasted",
            "Energy Consumption (kWh) — Forecasted",
        ),
        vertical_spacing=0.12,
        shared_xaxes=True,
    )

    # --- Panel 1: Generation ---
    hist_gen_user_scaled = historical_data.get("generated")
    if hist_gen_user_scaled is not None and not hist_gen_user_scaled.empty:
        fig.add_trace(go.Scatter(
            x=hist_gen_user_scaled.index, y=hist_gen_user_scaled.values,
            mode="lines", name="Hist. Gen. (User's System Scale)",
            line=dict(color=_get_color("historical_generated_user_system", "cornflowerblue")),
        ), row=1, col=1)

    if user_system_generation_forecast_kw is not None and not user_system_generation_forecast_kw.empty:
        fig.add_trace(go.Scatter(
            x=user_system_generation_forecast_kw.index,
            y=user_system_generation_forecast_kw.values,
            mode="lines", name="Fcst. Gen. (User's System)",
            line=dict(color=_get_color("forecast_generated_user_system", "darkorange"), dash="dash"),
        ), row=1, col=1)
    elif forecast_data.get("generated", {}).get("dates") is not None and len(forecast_data.get("generated", {}).get("values", [])) > 0:
        fc_gen_ref = forecast_data["generated"]
        fig.add_trace(go.Scatter(
            x=fc_gen_ref["dates"], y=fc_gen_ref["values"],
            mode="lines", name="Fcst. Gen. (Ref System)",
            line=dict(color=_get_color("forecast_generated", "orange"), dash="dash"),
        ), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(x=[], y=[], mode="lines", name="No Generation Forecast"), row=1, col=1)

    if (aligned_hourly_overlay is not None
            and "consumption_kwh" in aligned_hourly_overlay.columns
            and not aligned_hourly_overlay.empty
            and "timestamp" in aligned_hourly_overlay.columns):
        fig.add_trace(go.Scatter(
            x=aligned_hourly_overlay["timestamp"],
            y=aligned_hourly_overlay["consumption_kwh"],
            mode="lines", name="Fcst. Aligned Cons. (Hourly)",
            line=dict(color=_get_color("overlay_consumption", "grey"), dash="dot", width=1.5),
        ), row=1, col=1)

    # --- Panel 2: Cost ---
    hist_cost = historical_data.get("cost")
    fc_cost = forecast_data.get("cost", {})

    if hist_cost is not None and not hist_cost.empty:
        fig.add_trace(go.Scatter(
            x=hist_cost.index, y=hist_cost.values,
            mode="lines", name="Hist. Cost",
            line=dict(color=_get_color("historical_cost", "green")),
        ), row=2, col=1)
    if fc_cost.get("dates") is not None and len(fc_cost.get("values", [])) > 0:
        fc_dates_cost = pd.DatetimeIndex(fc_cost["dates"])
        fig.add_trace(go.Scatter(
            x=fc_dates_cost, y=fc_cost["values"],
            mode="lines", name="Fcst. Cost",
            line=dict(color=_get_color("forecast_cost", "red"), dash="dash"),
        ), row=2, col=1)
        ci_cost = fc_cost.get("conf_int")
        if ci_cost is not None and not ci_cost.empty and len(ci_cost) == len(fc_dates_cost):
            fig.add_trace(go.Scatter(x=fc_dates_cost, y=ci_cost.iloc[:, 0], mode="lines",
                                     line_width=0, showlegend=False), row=2, col=1)
            fig.add_trace(go.Scatter(x=fc_dates_cost, y=ci_cost.iloc[:, 1], mode="lines",
                                     line_width=0, fill="tonexty",
                                     fillcolor=_get_color("conf_int_cost_fill_plotly", "rgba(255,0,0,0.1)"),
                                     name="Cost 95% CI"), row=2, col=1)

    # --- Panel 3: Consumption ---
    hist_cons = historical_data.get("consumed")
    fc_cons = forecast_data.get("consumed", {})

    if hist_cons is not None and not hist_cons.empty:
        fig.add_trace(go.Scatter(
            x=hist_cons.index, y=hist_cons.values,
            mode="lines", name="Hist. Cons.",
            line=dict(color=_get_color("historical_consumed", "purple")),
        ), row=3, col=1)
    if fc_cons.get("dates") is not None and len(fc_cons.get("values", [])) > 0:
        fc_dates_cons = pd.DatetimeIndex(fc_cons["dates"])
        fig.add_trace(go.Scatter(
            x=fc_dates_cons, y=fc_cons["values"],
            mode="lines", name="Fcst. Cons.",
            line=dict(color=_get_color("forecast_consumed", "brown"), dash="dash"),
        ), row=3, col=1)
        ci_cons = fc_cons.get("conf_int")
        if ci_cons is not None and not ci_cons.empty and len(ci_cons) == len(fc_dates_cons):
            fig.add_trace(go.Scatter(x=fc_dates_cons, y=ci_cons.iloc[:, 0], mode="lines",
                                     line_width=0, showlegend=False), row=3, col=1)
            fig.add_trace(go.Scatter(x=fc_dates_cons, y=ci_cons.iloc[:, 1], mode="lines",
                                     line_width=0, fill="tonexty",
                                     fillcolor=_get_color("conf_int_consumed_fill_plotly", "rgba(165,42,42,0.1)"),
                                     name="Cons. 95% CI"), row=3, col=1)

    fig.update_layout(
        height=750,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        margin=dict(l=40, r=20, t=60, b=20),
    )
    fig.update_xaxes(tickformat="%Y-%m-%d %Hh")
    fig.update_yaxes(title_text="Power (kW)", row=1, col=1)
    fig.update_yaxes(title_text="Price ($/kWh)", row=2, col=1)
    fig.update_yaxes(title_text="Energy (kWh)", row=3, col=1)

    return fig
