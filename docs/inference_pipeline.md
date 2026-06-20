**Navigation:** [README](../README.md) · [Architecture](architecture.md) · [Data Pipeline](data_pipeline.md) · [Models](models.md)

---

# Inference Pipeline

This document traces exactly what happens from the moment the user clicks **"Generate Forecast
& Estimate Savings"** in the Streamlit app to the charts and financial metrics on screen.

---

## End-to-End Flow

```mermaid
flowchart TD
    CLICK["User clicks\nGenerate Forecast & Estimate Savings"]

    CLICK --> LOAD_DATA["1 · Load historical data\nload_energy_data_from_config()\n\n→ hist_cost (weekly Series)\n→ hist_consumed (weekly Series)\n→ hist_gen_df (hourly DataFrame, 5 cols)"]

    LOAD_DATA --> RESOLVE_STEPS["2 · Resolve forecast horizon\nget_steps_from_config(selected_horizon_key)\n\nExample — '1 Month':\n  hourly_steps = 720\n  weekly_steps_sarima = 4"]

    RESOLVE_STEPS --> LSTM_LOOP["3 · LSTM iterative forecast\nSee detail below"]

    LSTM_LOOP --> SARIMA_FORECAST["4 · SARIMA forecasts\nsarima_cost_model.forecast(steps=4)\nsarima_consumed_model.forecast(steps=4)\n\n→ fc_cost_mean, fc_cost_ci\n→ fc_cons_mean, fc_cons_ci"]

    SARIMA_FORECAST --> ALIGN["5 · Align to hourly index\nalign_forecast_data()\n\nGeneration: reindex hourly forecast to master index\nCost: derive $/kWh = weekly_$ / weekly_kWh → ffill to hourly\nConsumption: weekly_kWh / (7×24) → average hourly → ffill\n\n→ aligned_df columns: generation_kw, cost_per_kwh, consumption_kwh"]

    ALIGN --> FINANCIAL["6 · Financial simulation\ncalculate_financial_summary()\n\nSee financial detail below"]

    FINANCIAL --> SCALE_HIST["7 · Scale historical generation\nfor plotting only\nscaled_hist = hist_gen × (user_kWp / ref_kWp)"]

    SCALE_HIST --> CHARTS["8 · Render charts\nPlotly: create_plotly_forecast_chart()\nMatplotlib: plot_forecasts_matplotlib()\n\n3 subplots: generation, cost, consumption"]

    CHARTS --> METRICS["9 · Display KPI metrics\n─ Total energy generated (kWh)\n─ Total energy consumed (kWh)\n─ Avg import price ($/kWh)\n─ Gross value of solar ($)"]

    style CLICK fill:#fff9c4,stroke:#d69e2e
    style METRICS fill:#f0fff4,stroke:#38a169
    style LSTM_LOOP fill:#e8f4fd,stroke:#2b6cb0
    style FINANCIAL fill:#fce8e6,stroke:#c53030
```

---

## Step 3 Detail — LSTM Iterative Loop

```mermaid
flowchart TD
    INIT["Take last 168 rows of hist_gen_df\n(current window: 168 h × 5 features)"]
    INIT --> SCALE_WIN["Scale target column only\nscale_data(window[target_col], lstm_scaler)"]

    SCALE_WIN --> LOOP_START{"Remaining hours\n> 0?"}
    LOOP_START -->|Yes| TENSOR["Build input tensor\ntorch.tensor(window.values)\nshape: 1 × 168 × 5 → to device"]
    TENSOR --> FORWARD["lstm_model(tensor)\nForward pass\noutput: 1 × 12 (scaled predictions)"]
    FORWARD --> UNSCALE["inverse_scale(output, lstm_scaler)\n→ 12 hours of predicted watts"]
    UNSCALE --> APPEND["Append to all_predictions list"]
    APPEND --> BUILD_FUTURE["Build next 12 rows:\n1. Create DatetimeIndex (last_ts + 1h … + 12h)\n2. add_cyclical_time_features() → time encodings\n3. Scale the 12 predicted watts\n4. Assemble DataFrame with same 5 columns as window"]
    BUILD_FUTURE --> SLIDE["Slide window:\nnew_window = concat(window[12:], future_12_rows)"]
    SLIDE --> LOOP_START
    LOOP_START -->|No| CONCAT["Concatenate all prediction chunks\ntrim to exact hourly_steps length"]
    CONCAT --> OUT["forecasts_raw_model_output['generated']\n= {values: np.array, dates: DatetimeIndex}"]

    style INIT fill:#e8f4fd,stroke:#2b6cb0
    style OUT fill:#f0fff4,stroke:#38a169
```

---

## Step 5 Detail — Frequency Alignment

SARIMA forecasts are weekly; LSTM forecasts are hourly. The app needs a single hourly DataFrame.

```mermaid
flowchart TD
    GEN_HOURLY["LSTM output\nhourly dates + watt values"]
    COST_WEEKLY["SARIMA cost output\nweekly dates + weekly $ totals"]
    CONS_WEEKLY["SARIMA consumption output\nweekly dates + weekly kWh totals"]

    GEN_HOURLY --> REINDEX["Reindex to master_hourly_index\nmethod='nearest', tolerance=30 min\n→ generation_kw column"]

    COST_WEEKLY & CONS_WEEKLY --> DERIVE["Derive hourly $/kWh:\nprice_per_kwh = weekly_$ / weekly_kWh\nresample('h').ffill()\nreindex(master_hourly_index, method='ffill')\n→ cost_per_kwh column"]

    CONS_WEEKLY --> DISSAGG["Disaggregate consumption:\navg_hourly = weekly_kWh / (7 × 24)\nresample('h').ffill()\nreindex(master_hourly_index, method='ffill')\n→ consumption_kwh column"]

    REINDEX & DERIVE & DISSAGG --> ALIGNED_DF["aligned_df (hourly)\nindex = timestamp\ncolumns: generation_kw, cost_per_kwh, consumption_kwh"]

    style ALIGNED_DF fill:#f0fff4,stroke:#38a169
```

---

## Step 6 Detail — Financial Simulation (hourly)

```mermaid
flowchart LR
    subgraph SCALE_GEN ["Scale generation to user system"]
        SG1["ref_total_kWp = ref_panels × ref_capacity_kW"]
        SG2["user_total_kWp = user_panels × user_capacity_kW"]
        SG3["factor = user_total_kWp / ref_total_kWp"]
        SG4["actual_gen_kwh = generation_kw × factor × efficiency"]
        SG1 & SG2 --> SG3 --> SG4
    end

    subgraph BALANCE ["Hourly energy balance"]
        B1["self_consumed = min(actual_gen, consumption)"]
        B2["exported = max(0, actual_gen − self_consumed)"]
        B3["imported = max(0, consumption − actual_gen)"]
        SG4 --> B1 & B2 & B3
    end

    subgraph COSTS ["Hourly cost calculation"]
        C1["cost_no_solar = consumption × cost_per_kwh"]
        C2["import_cost = imported × cost_per_kwh"]
        C3["export_revenue = exported × export_tariff"]
        B1 & B2 & B3 --> C1 & C2 & C3
    end

    subgraph SUM ["Aggregate over all hours"]
        S1["cost_without_solar = Σ cost_no_solar"]
        S2["cost_with_solar = Σ import_cost − Σ export_revenue"]
        S3["net_savings = cost_without_solar − cost_with_solar"]
        C1 --> S1
        C2 & C3 --> S2
        S1 & S2 --> S3
    end
```

---

## Model Loading & Caching

Models are loaded once per browser session via `@st.cache_resource` and reused on every
"Generate Forecast" click. This avoids re-deserialising the ~50 MB PyTorch checkpoint on
every interaction.

```mermaid
sequenceDiagram
    participant Browser as Browser / User
    participant ST as Streamlit session
    participant Cache as @st.cache_resource
    participant Disk as models/artifacts/

    Browser->>ST: First page load
    ST->>Cache: load_all_models_and_scaler()
    Cache->>Disk: torch.load(lstm_solar_generator.pth)
    Cache->>Disk: joblib.load(lstm_scaler.pkl)
    Cache->>Disk: joblib.load(sarima_cost_model.pkl)
    Cache->>Disk: joblib.load(sarima_consumed_model.pkl)
    Disk-->>Cache: All four objects in memory
    Cache-->>ST: Cached dict {lstm, lstm_scaler, sarima_cost, sarima_consumed}

    Browser->>ST: Click "Generate Forecast"
    ST->>Cache: load_all_models_and_scaler()
    Cache-->>ST: Returns from cache (no disk I/O)
    ST->>ST: Run inference pipeline
    ST-->>Browser: Charts + metrics
```

Cache TTL is set by `caching.model_ttl_seconds` in `config.yaml` (default: 3600 s).
