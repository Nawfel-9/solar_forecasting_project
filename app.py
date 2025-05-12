# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import joblib
import numpy as np
from datetime import datetime
from models.lstm_model import LSTM
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from utils.preprocessing import align_frequencies,create_lstm_sequences,fit_scaler,scale_data,inverse_scale,get_device
from utils.visualization import plot_multiple_forecasts

# Configure page settings
st.set_page_config(
    page_title="Solar Forecast Interface",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .metric-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stPlot {
        border-radius: 12px;
        padding: 15px;
        background-color: white;
    }
    .summary-card {
        padding: 15px;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 10px 0;
        transition: transform 0.3s;
    }
    .summary-card:hover {
        transform: translateY(-5px);
    }
    .summary-icon {
        font-size: 24px;
        float: right;
        margin-top: -5px;
    }
</style>
""", unsafe_allow_html=True)

def main_interface():
    st.title("🌞 Solar Forecast Interface")
    
    # Sidebar controls
    with st.sidebar:
        st.header("Settings")
        forecast_horizon = st.selectbox(
            "Forecast Horizon",
            ["1 Week", "2 Weeks", "1 Month"],
            index=2
        )
        show_raw_data = st.checkbox("Show Raw Data Preview", False)
        
        # Plot type selection
        st.subheader("Visualization")
        plot_type = st.radio(
            "Chart Type",
            ["Matplotlib", "Plotly"],
            index=0
        )
        
        # Add date range selector for historical data
        st.subheader("Historical Data Range")
        start_date = st.date_input("Start Date", value=pd.to_datetime("now") - pd.Timedelta(days=30))
        end_date = st.date_input("End Date", value=pd.to_datetime("now"))
        
        # Theme selection
        st.subheader("Appearance")
        theme = st.selectbox(
            "Application Theme",
            ["Light", "Dark", "Solar"],
            index=0
        )
        
        # Apply selected theme
        if theme == "Dark":
            st.markdown("""
            <style>
                .stApp {
                    background-color: #121212;
                    color: #E0E0E0;
                }
                .metric-box, .summary-card {
                    background-color: #1E1E1E !important;
                    color: #E0E0E0 !important;
                }
            </style>
            """, unsafe_allow_html=True)
        elif theme == "Solar":
            st.markdown("""
            <style>
                .stApp {
                    background-color: #263238;
                    color: #FAFAFA;
                }
                .metric-box, .summary-card {
                    background-color: #37474F !important;
                    color: #FAFAFA !important;
                }
            </style>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("**Model Paths**")
        st.code(f"""
        Generated: models/best_model_energy_generated.pth
        Cost: models/sarima_cost_model.pkl 
        Consumed: models/sarima_consumed_model.pkl
        """)
    
    # Create tabs
    tab1, tab2 = st.tabs(["Forecast Dashboard", "Historical Analysis"])
    
    # Main content area - Tab 1: Forecast Dashboard
    with tab1:
        if st.button("Run Full Forecast Pipeline", type="primary"):
            # Initialize status container and progress tracking first
            status_container = st.status("Initializing...", expanded=True)
            progress_bar = st.progress(0)
            
            try:
                # Configuration
                device = get_device()
                hourly_steps = 24 * 30 * 1  # 1 month
                weekly_steps = 4
                model_paths = {
                    'generated': Path("models/best_model_energy_generated.pth"),
                    'cost': Path("models/sarima_cost_model.pkl"),
                    'consumed': Path("models/sarima_consumed_model.pkl")
                }

                # Load data
                status_container.update(label="**1/6** Loading data...", state="running")
                generated_hourly = pd.read_csv("data/preprocessed/energy_generated.csv", 
                                             index_col=0, parse_dates=True).squeeze()
                cost_weekly = pd.read_csv("data/preprocessed/energy_cost.csv", 
                                        index_col=0, parse_dates=True).squeeze()
                consumed_weekly = pd.read_csv("data/preprocessed/energy_consumed.csv", 
                                            index_col=0, parse_dates=True).squeeze()
                progress_bar.progress(15)

                # Align frequencies
                status_container.update(label="**2/6** Aligning frequencies...", state="running")
                generated, cost, consumed = align_frequencies(generated_hourly, cost_weekly, consumed_weekly)
                progress_bar.progress(30)

                # Generate forecasts
                forecasts = {
                    'generated': {'freq': 'H', 'values': None, 'dates': None},
                    'cost': {'freq': 'W', 'values': None, 'dates': None},
                    'consumed': {'freq': 'W', 'values': None, 'dates': None}
                }

                # LSTM Forecast
                status_container.update(label="**3/6** Running LSTM forecast...", state="running")
                scaler = fit_scaler(generated)
                scaled_data = scale_data(generated, scaler)
                X, _ = create_lstm_sequences(scaled_data, time_steps=24*7*2)
                
                model = LSTM().to(device)
                model.load_state_dict(torch.load(model_paths['generated'], map_location=device), strict=True)
                last_input = torch.tensor(X[-1], dtype=torch.float32).view(1, -1, 1).to(device)
                scaled_forecast = model.forecast_series(model, last_input, hourly_steps)
                
                forecasts['generated']['values'] = inverse_scale(np.array(scaled_forecast), scaler)
                forecasts['generated']['dates'] = pd.date_range(
                    start=generated.index[-1] + pd.Timedelta(hours=1), 
                    periods=hourly_steps, 
                    freq='h'
                )
                progress_bar.progress(50)

                # SARIMA Forecasts
                status_container.update(label="**4/6** Running SARIMA forecasts...", state="running")
                sarima_cost = joblib.load(model_paths['cost'])
                sarima_consumed = joblib.load(model_paths['consumed'])
                
                for target, model in [('cost', sarima_cost), ('consumed', sarima_consumed)]:
                    forecast = model.get_forecast(steps=weekly_steps)
                    forecasts[target]['values'] = forecast.predicted_mean.values
                    forecasts[target]['dates'] = pd.date_range(
                        start=locals()[f"{target}_weekly"].index[-1] + pd.Timedelta(weeks=1), 
                        periods=weekly_steps, 
                        freq='W'
                    )
                    forecasts[target]['conf_int'] = forecast.conf_int()
                progress_bar.progress(75)

                # Visualization
                status_container.update(label="**5/6** Generating visualizations...", state="running")
                
                if plot_type == "Matplotlib":
                    fig = plot_multiple_forecasts(
                        historical={
                            'generated': generated,
                            'cost': cost_weekly,
                            'consumed': consumed_weekly
                        },
                        forecasts=forecasts
                    )
                else:  # Plotly
                    fig = create_plotly_forecast_chart(
                        historical={
                            'generated': generated,
                            'cost': cost_weekly,
                            'consumed': consumed_weekly
                        },
                        forecasts=forecasts
                    )
                progress_bar.progress(90)

                # Results display
                status_container.update(label="**6/6** Displaying results...", state="running")
                
                # Enhanced summary cards
                st.subheader("Forecast Summary")
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown(f"""
                    <div class="summary-card" style="border-left: 4px solid #FFA726; background: #ffd588;">
                        <div class="summary-icon">⚡</div>
                        <h3 style="margin:0;font-size:18px;">Generation</h3>
                        <p style="font-size:26px;font-weight:bold;margin:10px 0;">{forecasts['generated']['values'].max():.2f} kW</p>
                        <p style="margin:0;font-size:14px;color:#666;">Peak forecasted generation</p>
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    st.markdown(f"""
                    <div class="summary-card" style="border-left: 4px solid #EF5350; background: #ff7777;">
                        <div class="summary-icon">💲</div>
                        <h3 style="margin:0;font-size:18px;">Cost</h3>
                        <p style="font-size:26px;font-weight:bold;margin:10px 0;">${forecasts['cost']['values'].mean():.2f}</p>
                        <p style="margin:0;font-size:14px;color:#666;">Avg. cost per kWh</p>
                    </div>
                    """, unsafe_allow_html=True)

                with col3:
                    st.markdown(f"""
                    <div class="summary-card" style="border-left: 4px solid #66BB6A; background: #87bd7a;">
                        <div class="summary-icon">🔋</div>
                        <h3 style="margin:0;font-size:18px;">Consumption</h3>
                        <p style="font-size:26px;font-weight:bold;margin:10px 0;">{forecasts['consumed']['values'].sum():.0f} kWh</p>
                        <p style="margin:0;font-size:14px;color:#666;">Total consumption forecast</p>
                    </div>
                    """, unsafe_allow_html=True)

                # Show plot in expandable section
                with st.expander("Forecast Visualization", expanded=True):
                    if plot_type == "Matplotlib":
                        st.pyplot(fig)
                    else:  # Plotly
                        st.plotly_chart(fig, use_container_width=True)
                
                # Data export
                with st.expander("Data Export", expanded=False):
                    # 1) Aggregate hourly generation into daily sums
                    gen_series = pd.Series(
                        forecasts['generated']['values'],
                        index=pd.DatetimeIndex(forecasts['generated']['dates'])
                    )
                    gen_daily = gen_series.resample('D').sum()

                    # 2) Turn weekly cost/consumption into daily values
                    cost_series = pd.Series(
                        forecasts['cost']['values'],
                        index=pd.DatetimeIndex(forecasts['cost']['dates'])
                    )
                    cons_series = pd.Series(
                        forecasts['consumed']['values'],
                        index=pd.DatetimeIndex(forecasts['consumed']['dates'])
                    )

                    # forward‐fill the same weekly value to each day, then split evenly
                    cost_daily = cost_series.resample('D').ffill() / 7
                    cons_daily = cons_series.resample('D').ffill() / 7

                    # 3) Build a daily‐frequency DataFrame
                    daily_index = gen_daily.index  # all the days you have generation for
                    forecast_df = pd.DataFrame({
                        'date':                   daily_index,
                        'generation_kwh':         gen_daily.values,
                        'cost_per_kwh':           cost_daily.reindex(daily_index).values,
                        'consumption_kwh':        cons_daily.reindex(daily_index).values,
                    })

                    # then your CSV export as before:
                    csv = forecast_df.to_csv(index=False)
                    st.download_button(
                        label="Download Forecast Data (CSV)",
                        data=csv,
                        file_name=f"solar_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )

                    if show_raw_data:
                        st.dataframe(forecast_df.head(20), use_container_width=True)

                progress_bar.progress(100)
                status_container.update(label="Forecast pipeline completed successfully!", state="complete")

            except Exception as e:
                # Error handling that won't reference undefined variables
                if 'progress_bar' in locals():
                    progress_bar.progress(0)
                if 'status_container' in locals():
                    status_container.update(label="Pipeline Error", state="error", expanded=True)
                    status_container.write(f"""
                    ```
                    {str(e)}
                    ```
                    **Troubleshooting Steps:**
                    1. Check model files exist in /models/
                    2. Verify data files in /data/preprocessed/
                    3. Ensure all dependencies are installed
                    """)
                else:
                    st.error(f"""
                    **Pipeline Error**
                    ```
                    {str(e)}
                    ```
                    **Troubleshooting Steps:**
                    1. Check model files exist in /models/
                    2. Verify data files in /data/preprocessed/
                    3. Ensure all dependencies are installed
                    """)

    # Tab 2: Historical Analysis
    with tab2:
        st.subheader("Historical Data Analysis")
        st.info("Select a date range in the sidebar to view historical data analysis")
        
        # Placeholder for historical analysis charts
        st.markdown("#### Energy Generation History")
        # This would be replaced with actual historical data plotting
        st.line_chart({"Generation (kW)": [random.random() * 10 + 20 for _ in range(30)]})
        
        with st.expander("Monthly Trends", expanded=False):
            st.markdown("Historical monthly generation and consumption patterns would appear here")
    
# Function to create Plotly charts for forecasts
def create_plotly_forecast_chart(historical, forecasts):
    # Create subplots with 3 rows
    fig = make_subplots(rows=3, cols=1, 
                        subplot_titles=("Energy Generation", "Energy Cost", "Energy Consumption"),
                        vertical_spacing=0.1,
                        shared_xaxes=True)
    
    # Colors for consistency
    colors = {
        'generated': '#FFA726',
        'cost': '#EF5350', 
        'consumed': '#66BB6A'
    }
    
    # Add Generation data and forecast
    fig.add_trace(
        go.Scatter(
            x=historical['generated'].index,
            y=historical['generated'].values,
            mode='lines',
            name='Historical Generation',
            line=dict(color=colors['generated'], width=2)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=forecasts['generated']['dates'], 
            y=forecasts['generated']['values'],
            mode='lines',
            name='Forecast Generation',
            line=dict(color=colors['generated'], width=2, dash='dash')
        ),
        row=1, col=1
    )
    
    # Add Cost data and forecast
    fig.add_trace(
        go.Scatter(
            x=historical['cost'].index,
            y=historical['cost'].values,
            mode='lines',
            name='Historical Cost',
            line=dict(color=colors['cost'], width=2)
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=forecasts['cost']['dates'], 
            y=forecasts['cost']['values'],
            mode='lines',
            name='Forecast Cost',
            line=dict(color=colors['cost'], width=2, dash='dash')
        ),
        row=2, col=1
    )
    
    # Add Cost confidence intervals if available
    if 'conf_int' in forecasts['cost']:
        fig.add_trace(
            go.Scatter(
                x=forecasts['cost']['dates'],
                y=forecasts['cost']['conf_int'].iloc[:, 0],
                line=dict(color='rgba(239, 83, 80, 0.0)'),
                showlegend=False
            ),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=forecasts['cost']['dates'],
                y=forecasts['cost']['conf_int'].iloc[:, 1],
                fill='tonexty',
                fillcolor='rgba(239, 83, 80, 0.2)',
                line=dict(color='rgba(239, 83, 80, 0.0)'),
                name='Cost 95% CI'
            ),
            row=2, col=1
        )
    
    # Add Consumption data and forecast
    fig.add_trace(
        go.Scatter(
            x=historical['consumed'].index,
            y=historical['consumed'].values,
            mode='lines',
            name='Historical Consumption',
            line=dict(color=colors['consumed'], width=2)
        ),
        row=3, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=forecasts['consumed']['dates'], 
            y=forecasts['consumed']['values'],
            mode='lines',
            name='Forecast Consumption',
            line=dict(color=colors['consumed'], width=2, dash='dash')
        ),
        row=3, col=1
    )
    
    # Add Consumption confidence intervals if available
    if 'conf_int' in forecasts['consumed']:
        fig.add_trace(
            go.Scatter(
                x=forecasts['consumed']['dates'],
                y=forecasts['consumed']['conf_int'].iloc[:, 0],
                line=dict(color='rgba(102, 187, 106, 0.0)'),
                showlegend=False
            ),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=forecasts['consumed']['dates'],
                y=forecasts['consumed']['conf_int'].iloc[:, 1],
                fill='tonexty',
                fillcolor='rgba(102, 187, 106, 0.2)',
                line=dict(color='rgba(102, 187, 106, 0.0)'),
                name='Consumption 95% CI'
            ),
            row=3, col=1
        )
    
    # Update layout
    fig.update_layout(
        height=800,
        title_text="Energy Forecasts",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified"
    )
    
    # Update y-axis labels
    fig.update_yaxes(title_text="Energy (kW)", row=1, col=1)
    fig.update_yaxes(title_text="Cost ($/kWh)", row=2, col=1)
    fig.update_yaxes(title_text="Energy (kWh)", row=3, col=1)
    
    return fig

if __name__ == "__main__":
    # Add import for random at the top of the file if you keep the placeholder data
    import random
    main_interface()