# solar_forecasting_project/app.py
import streamlit as st
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from datetime import datetime, date
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, List

# Project-specific imports from refactored modules
from utils.config_loader import load_config, get_model_path
from utils.preprocessing import (
    get_device,
    load_all_models_and_scaler,
    load_energy_data_from_config,
    align_forecast_data,
    get_steps_from_config,
    scale_data,
    inverse_scale,
    add_cyclical_time_features  # Crucial for the new forecast method
)
from utils.evaluation import calculate_financial_summary
from utils.visualisation import plot_forecasts_matplotlib, create_plotly_forecast_chart

# --- Page and App Configuration ---
CONFIG = load_config()
st.set_page_config(
    page_title=CONFIG.get('app_config', {}).get('page_title', "Solar Forecasting & Savings"),
    layout=CONFIG.get('app_config', {}).get('layout', "wide"),
    initial_sidebar_state=CONFIG.get('app_config', {}).get('initial_sidebar_state', "expanded")
)

# --- Custom CSS ---
st.markdown("""
<style>
    .stMetricValue { font-size: 28px !important; }
    .stMetricLabel { font-size: 16px !important; }
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    button[data-testid="baseButton-primary"] {
        background-color: #FF8C00; color: white; border: none; padding: 10px 24px; border-radius: 5px;
    }
    button[data-testid="baseButton-primary"]:hover {
        background-color: #FFA500; color: white;
    }
</style>
""", unsafe_allow_html=True)

# --- Main Application ---
def run_app():
    st.title(CONFIG.get('app_config', {}).get('main_title', "🌞 Solar Panel Forecast & Savings Estimator"))

    # Load models once at the start of the app session
    with st.spinner("Loading predictive models and data utilities..."):
        loaded_models_and_scaler = load_all_models_and_scaler()

    # Check for critical components
    critical_components = ['lstm', 'lstm_scaler', 'sarima_cost', 'sarima_consumed']
    missing_components = [comp for comp in critical_components if loaded_models_and_scaler.get(comp) is None]
    if missing_components:
        st.error(f"Critical component(s) failed to load: {', '.join(missing_components)}. Please train models first.")
        st.stop()

    lstm_model = loaded_models_and_scaler['lstm']
    lstm_scaler = loaded_models_and_scaler['lstm_scaler']
    sarima_cost_model = loaded_models_and_scaler['sarima_cost']
    sarima_consumed_model = loaded_models_and_scaler['sarima_consumed']
    device = get_device()

    # --- Sidebar ---
    with st.sidebar:
        # (Your sidebar code remains the same)
        st.header("⚙️ Configuration")
        app_cfg = CONFIG.get('app_settings', {})
        horizon_options = list(CONFIG['forecast_horizons']['steps_map'].keys())
        default_horizon_key = CONFIG['forecast_horizons']['default_app_horizon']
        default_horizon_index = horizon_options.index(default_horizon_key) if default_horizon_key in horizon_options else 0
        selected_horizon_key = st.selectbox("Select Forecast Horizon:", options=horizon_options, index=default_horizon_index)
        st.markdown("---")
        st.subheader("☀️ Solar Panel System (Your Desired Setup)")
        num_panels = st.number_input("Number of Solar Panels:", min_value=1, value=int(app_cfg.get('default_num_panels', 10)), step=1)
        default_panel_cap_kw = float(CONFIG['app_settings'].get('default_panel_capacity_kw', 0.350))
        panel_capacity_kw = st.number_input("Capacity per Panel (kW):", min_value=0.010, value=default_panel_cap_kw, step=0.010, format="%.3f")
        panel_efficiency = st.slider("System Efficiency (%):", min_value=50, max_value=100, value=int(app_cfg.get('default_panel_efficiency', 0.85) * 100), step=1) / 100.0
        export_tariff = st.number_input("Export Tariff ($/kWh, if any):", min_value=0.00, value=float(app_cfg.get('default_export_tariff', 0.05)), step=0.001, format="%.3f")
        st.markdown(f"*Total User-Defined System Capacity: **{num_panels * panel_capacity_kw:.2f} kWp***")
        st.markdown("---")
        st.subheader("📊 Chart & Data")
        plot_engine = st.radio("Chart Engine:", ["Plotly (Interactive)", "Matplotlib (Static)"], index=0)

    # --- Main Content Tabs ---
    tab_dashboard, tab_historical, tab_details = st.tabs(["📈 Forecast Dashboard", "📊 Historical Data", "🛠️ Config Info"])

    with tab_dashboard:
        st.header("Forecasts & Financial Impact")
        if st.button("🚀 Generate Forecast & Estimate Savings", type="primary", use_container_width=True):
            
            # Initialize variables to hold results before the status block
            financial_summary = None
            historical_for_plot = None
            forecasts_for_plot = None
            aligned_forecast_df = None
            user_system_actual_power_kw_series_for_plot = None
            forecasts_raw_model_output = None # Hold raw forecasts for plotting
            
            # --- COMPUTATION BLOCK: Use st.status for all calculations ---
            with st.status("Running forecasting pipeline...", expanded=True) as forecast_pipeline_status:
                try:
                    # 1. Load Data
                    forecast_pipeline_status.update(label="Loading historical data...")
                    hist_cost, hist_consumed, hist_gen_actual_reference_df = load_energy_data_from_config()
                    if hist_gen_actual_reference_df.empty or hist_cost.empty or hist_consumed.empty:
                        st.error("Failed to load historical data."); st.stop()
                    target_col = CONFIG['data_paths']['lstm_target_column_name']
                    hist_gen_actual_reference = hist_gen_actual_reference_df[target_col]

                    # 2. Get Steps
                    forecast_steps_config = get_steps_from_config(selected_horizon_key, CONFIG)
                    hourly_steps = forecast_steps_config.get('hourly', 720)
                    weekly_steps_sarima = forecast_steps_config.get('weekly', 4)

                    # 3. LSTM Generation Forecast (Iterative Direct Multi-Step)
                    forecast_pipeline_status.update(label="Forecasting solar generation (LSTM)...")
                    lstm_params = CONFIG['lstm_params']; time_steps = CONFIG['preprocessing']['lstm_time_steps']
                    output_chunk_size = lstm_params.get('output_chunk_size', 24)
                    
                    all_predictions_unscaled = []
                    # (Full forecasting loop logic as provided before) ...
                    if len(hist_gen_actual_reference_df) >= time_steps:
                        last_sequence_df = hist_gen_actual_reference_df.iloc[-time_steps:]
                        current_sequence_scaled_df = last_sequence_df.copy()
                        current_sequence_scaled_df[target_col] = scale_data(last_sequence_df[target_col], lstm_scaler)
                        lstm_model.eval()
                        with torch.no_grad():
                            for _ in range(int(np.ceil(hourly_steps / output_chunk_size))):
                                input_tensor = torch.tensor(current_sequence_scaled_df.values, dtype=torch.float32).unsqueeze(0).to(device)
                                prediction_chunk_scaled = lstm_model(input_tensor).squeeze(0).cpu().numpy()
                                prediction_chunk_unscaled = inverse_scale(prediction_chunk_scaled, lstm_scaler)
                                all_predictions_unscaled.append(prediction_chunk_unscaled)
                                last_timestamp = current_sequence_scaled_df.index[-1]
                                future_dates = pd.date_range(start=last_timestamp + pd.Timedelta(hours=1), periods=output_chunk_size, freq='h')
                                future_chunk_df = pd.DataFrame(index=future_dates)
                                future_chunk_df = add_cyclical_time_features(future_chunk_df)
                                future_chunk_df[target_col] = prediction_chunk_unscaled
                                future_chunk_df[target_col] = scale_data(future_chunk_df[target_col], lstm_scaler)
                                future_chunk_df = future_chunk_df[current_sequence_scaled_df.columns]
                                current_sequence_scaled_df = pd.concat([current_sequence_scaled_df.iloc[output_chunk_size:], future_chunk_df])
                        final_predictions_unscaled = np.concatenate(all_predictions_unscaled)[:hourly_steps]
                        forecast_start_date = hist_gen_actual_reference.index[-1] + pd.Timedelta(hours=1)
                        forecast_dates = pd.date_range(start=forecast_start_date, periods=len(final_predictions_unscaled), freq='h')
                        forecasts_raw_model_output = {'generated': {'values': final_predictions_unscaled, 'dates': forecast_dates}}
                    else: st.warning(f"Not enough historical data for LSTM."); forecasts_raw_model_output = {'generated': {}}
                    
                    # 4. SARIMA Forecasts
                    forecast_pipeline_status.update(label="Forecasting cost & consumption (SARIMA)...")
                    fc_cost_mean, fc_cost_ci = sarima_cost_model.forecast(steps=weekly_steps_sarima)
                    forecasts_raw_model_output['cost'] = {'values': fc_cost_mean.values, 'dates': fc_cost_mean.index, 'conf_int': fc_cost_ci}
                    fc_cons_mean, fc_cons_ci = sarima_consumed_model.forecast(steps=weekly_steps_sarima)
                    forecasts_raw_model_output['consumed'] = {'values': fc_cons_mean.values, 'dates': fc_cons_mean.index, 'conf_int': fc_cons_ci}

                    # 5. Align Data
                    forecast_pipeline_status.update(label="Aligning data to hourly resolution...")
                    master_hourly_idx = forecasts_raw_model_output.get('generated', {}).get('dates')
                    if master_hourly_idx is None or master_hourly_idx.empty:
                         master_hourly_idx = pd.date_range(start=hist_gen_actual_reference.index[-1] + pd.Timedelta(hours=1), periods=hourly_steps, freq='h')
                    aligned_forecast_df = align_forecast_data(forecasts_raw_model_output, master_hourly_idx)
                    
                    # 6. Financial Evaluation with Scaling
                    forecast_pipeline_status.update(label="Calculating financial impact...")
                    ref_system_cfg = CONFIG.get('lstm_reference_system', {})
                    ref_num_panels = ref_system_cfg.get('num_panels', 0); ref_panel_capacity_w = ref_system_cfg.get('panel_capacity_w', 0)
                    if not (ref_num_panels > 0 and ref_panel_capacity_w > 0):
                        st.error("CRITICAL CONFIG ERROR: 'lstm_reference_system' in config.yaml is missing or zero."); financial_summary = {}
                    else:
                        financial_summary = calculate_financial_summary(
                            aligned_forecast_df_hourly=aligned_forecast_df.copy(),
                            user_num_panels=num_panels, user_panel_capacity_kw=panel_capacity_kw,
                            reference_num_panels=ref_num_panels, reference_panel_capacity_w=ref_panel_capacity_w,
                            panel_efficiency=panel_efficiency, export_tariff_rate=export_tariff,
                            import_tariff_col='cost_per_kwh'
                        )
                    
                    # 7. Prepare Scaled Data for Plotting
                    scaled_hist_gen_series_for_plot = hist_gen_actual_reference.copy()
                    user_system_actual_power_kw_series_for_plot = pd.Series(dtype=float)
                    if ref_num_panels > 0 and ref_panel_capacity_w > 0:
                        ref_panel_cap_kw = float(ref_panel_capacity_w) / 1000.0; ref_total_kwp = float(ref_num_panels) * ref_panel_cap_kw
                        user_total_kwp = float(num_panels) * float(panel_capacity_kw)
                        scaling_factor = user_total_kwp / ref_total_kwp if ref_total_kwp > 0 else 0
                        if not hist_gen_actual_reference.empty: scaled_hist_gen_series_for_plot = hist_gen_actual_reference * scaling_factor
                        if 'generation_kw' in aligned_forecast_df:
                            user_gen_potential = aligned_forecast_df['generation_kw'] * scaling_factor
                            user_gen_actual_kw = user_gen_potential * panel_efficiency
                            user_system_actual_power_kw_series_for_plot = pd.Series(user_gen_actual_kw.values, index=master_hourly_idx)
                    
                    forecast_pipeline_status.update(label="Pipeline completed successfully!", state="complete", expanded=False)

                except Exception as e:
                    st.error(f"An error occurred: {e}"); import traceback; st.error(f"Traceback: {traceback.format_exc()}");
                    financial_summary = None # Ensure summary is None on error
                    forecast_pipeline_status.update(label=f"Pipeline Error: {e}", state="error")
            
            # --- DISPLAY BLOCK: This is now OUTSIDE the st.status block ---
            if financial_summary: # Only display results if calculations were successful
                horizon_days = financial_summary.get('total_hours_forecasted', hourly_steps) // 24
                st.subheader("📊 Key Forecast Metrics")
                cols_metrics = st.columns(4)
                cols_metrics[0].metric("☀️ Energy Generated (Your System)", f"{financial_summary.get('total_energy_generated_kwh_user_system', 0):.0f} kWh", help=f"Total solar energy your system is forecasted to generate over {horizon_days} days.")
                cols_metrics[1].metric("🏠 Energy Consumed", f"{financial_summary.get('total_energy_consumed_kwh', 0):.0f} kWh", help=f"Total electricity forecasted for consumption over {horizon_days} days.")
                cols_metrics[2].metric("💰 Avg. Import Price", f"${financial_summary.get('average_import_price_forecasted', 0.0):.3f} /kWh", help=f"Average forecasted price of grid electricity over {horizon_days} days.")
                cols_metrics[3].metric("💸 Gross Value of Solar", f"${financial_summary.get('net_savings_usd_simplified', 0.0):.2f}", help=f"Total Non-Negative Generated kWh * Avg. Import Price.")

                with st.expander("Visual Forecasts", expanded=True):
                    historical_for_plot = {'generated': scaled_hist_gen_series_for_plot, 'cost': hist_cost, 'consumed': hist_consumed}
                    if plot_engine == "Plotly (Interactive)":
                        plotly_fig = create_plotly_forecast_chart(historical_for_plot, forecasts_raw_model_output, user_system_generation_forecast_kw=user_system_actual_power_kw_series_for_plot, aligned_hourly_overlay=aligned_forecast_df)
                        st.plotly_chart(plotly_fig, use_container_width=True)
                    else:
                        forecasts_for_mpl_plot = {'cost': forecasts_raw_model_output.get('cost',{}), 'consumed': forecasts_raw_model_output.get('consumed',{})}
                        if not user_system_actual_power_kw_series_for_plot.empty:
                            forecasts_for_mpl_plot['generated'] = {'dates': user_system_actual_power_kw_series_for_plot.index, 'values': user_system_actual_power_kw_series_for_plot.values}
                        else: forecasts_for_mpl_plot['generated'] = forecasts_raw_model_output.get('generated',{})
                        mpl_fig = plot_forecasts_matplotlib(historical_for_plot, forecasts_for_mpl_plot, aligned_hourly_overlay=aligned_forecast_df)
                        st.pyplot(mpl_fig)
                
                with st.expander("Detailed Financial Summary & Data Export", expanded=False):
                    st.subheader("Detailed Financial Breakdown")
                    fin_cols = st.columns(2)
                    fin_cols[0].metric("Cost without Solar:", f"${financial_summary.get('cost_without_solar_usd',0.0):.2f}")
                    fin_cols[0].metric("Net Cost with Solar (Import - Export Revenue):", f"${financial_summary.get('cost_with_solar_usd',0.0):.2f}")
                    fin_cols[1].metric("Comprehensive Net Savings:", f"${financial_summary.get('net_savings_usd_comprehensive',0.0):.2f}")
                    fin_cols[1].metric("Revenue from Export:", f"${financial_summary.get('revenue_from_export_usd',0.0):.2f}")
                    st.markdown(f"_*Calculations based on a {num_panels}-panel, {panel_capacity_kw*1000:.0f}W/panel system with {panel_efficiency*100:.0f}% efficiency.*_")

    # === HISTORICAL DATA & CONFIG INFO TABS ===
    with tab_details:
        st.header("🛠️ Configuration Information")
        st.info("This tab shows the key parameters loaded from your `config.yaml` file, which define how the models were trained and how the application runs.")
        
        st.subheader("File & Data Paths")
        dc = CONFIG['data_paths']
        st.json({
            "Preprocessed Data Directory": dc.get('preprocessed_dir'),
            "Model Artifacts Directory": dc.get('models_dir'),
            "Generation Data File (Hourly)": dc.get('energy_generated_csv'),
            "LSTM Target Column": dc.get('lstm_target_column_name'),
            "Cost Data File (Weekly)": dc.get('energy_cost_csv'),
            "Consumption Data File (Weekly)": dc.get('energy_consumed_csv'),
        })

        st.subheader("LSTM Reference System")
        ref_sys = CONFIG.get('lstm_reference_system', {})
        st.json({
            "Number of Panels": ref_sys.get('num_panels'),
            "Capacity per Panel (Watts)": ref_sys.get('panel_capacity_w')
        })

        st.subheader("Key LSTM Model & Training Parameters")
        lstm_p = CONFIG['lstm_params']
        prep_p = CONFIG['preprocessing']
        train_p = CONFIG['training_params']
        st.json({
            "Input Features": lstm_p.get('input_size'),
            "Hidden Layer Size": lstm_p.get('hidden_size'),
            "Number of Layers": lstm_p.get('num_layers'),
            "Dropout Rate": lstm_p.get('dropout'),
            "Bidirectional": lstm_p.get('bidirectional'),
            "Output Chunk Size (hours)": lstm_p.get('output_chunk_size'),
            "Input Sequence Length (hours)": prep_p.get('lstm_time_steps'),
            "Batch Size": train_p.get('lstm_batch_size'),
            "Initial Learning Rate": train_p.get('lstm_learning_rate'),
            "Weight Decay": train_p.get('lstm_weight_decay')
        })
        
        st.subheader("Key SARIMA Model Parameters")
        sarima_p = CONFIG.get('sarima_params', {})
        st.markdown("**Cost Model:**")
        st.json(sarima_p.get('cost', {}))
        st.markdown("**Consumption Model:**")
        st.json(sarima_p.get('consumed', {}))

# --- Run the App ---
if __name__ == "__main__":
    run_app()