**Navigation:** [README](../README.md) · [Architecture](architecture.md) · [Data Pipeline](data_pipeline.md) · [Inference Pipeline](inference_pipeline.md)

---

# Models

## 1. LSTM with Attention — Solar Generation Forecasting

### Purpose

Predicts hourly solar DC power output (W) for the next N hours, given the last 168 hours of
historical generation and cyclical time features.  Uses a **direct multi-step** strategy:
the model outputs a chunk of 12 future hours in a single forward pass, then slides forward
iteratively to cover the full forecast horizon.

### Architecture (`models/lstm_model.py`)

```mermaid
graph TD
    INPUT["Input Tensor\nshape: batch × 168 time steps × 5 features\n─ Generated Energy W scaled\n─ hour_sin, hour_cos\n─ day_of_year_sin, day_of_year_cos"]

    INPUT --> LSTM["Bidirectional LSTM\ninput_size = 5\nhidden_size = 32\nnum_layers = 1\ndropout = 0.303\n\nOutput shape: batch × 168 × 64\n  64 = 32 hidden × 2 directions"]

    LSTM --> ATTN_SCORE["Attention Network\nLinear(64 → 32) → Tanh → Linear(32 → 1)\nApplied at every time step\nOutput: energy score per step  batch × 168 × 1"]

    ATTN_SCORE --> SOFTMAX["Softmax over time steps\nOutput: attention weights  batch × 168\nSums to 1.0 across the 168 steps"]

    SOFTMAX --> WEIGHTED_SUM["Weighted Sum\nbmm(weights.unsqueeze(1), lstm_out)\n= batch × 1 × 64 → squeeze → batch × 64\nContext vector: one rich 64-d summary of the sequence"]

    WEIGHTED_SUM --> FC["Linear(64 → 12)\nOutput: 12 future hour predictions (scaled)"]

    FC --> RELU["ReLU\nClamps predictions to ≥ 0\nEnforces physical constraint: no negative power"]

    RELU --> OUTPUT["Output: batch × 12\n12 predicted future hours (scaled)"]

    style INPUT fill:#e8f4fd,stroke:#2b6cb0
    style OUTPUT fill:#f0fff4,stroke:#38a169
    style ATTN_SCORE fill:#fffde7,stroke:#d69e2e
    style WEIGHTED_SUM fill:#fffde7,stroke:#d69e2e
```

### Why Attention?

A plain LSTM compresses the entire 168-step sequence into a single hidden state. Attention lets
the model **look back** at all 168 outputs simultaneously and learn which time steps matter most
for the current prediction — for example, the same hour yesterday, or sunrise yesterday relative
to today's forecast window.

### Direct Multi-Step Forecasting Strategy

Instead of predicting one hour and feeding it back (recursive / autoregressive), the model
predicts a **chunk of 12 hours at once**. This avoids error accumulation across many auto-
regressive steps but requires re-running the model iteratively to cover a longer horizon.

```mermaid
sequenceDiagram
    participant H as Historical data (168 h)
    participant M as LSTM Model
    participant F as Forecast buffer

    Note over H: Last 168 hours from energy_generated.csv

    H->>M: Forward pass #1 (168 h context)
    M-->>F: chunk_1 = hours 1–12 (unscaled)

    Note over H,F: Slide window: append chunk_1, drop oldest 12 h

    H->>M: Forward pass #2 (168 h context, updated)
    M-->>F: chunk_2 = hours 13–24

    Note over H,F: Repeat ⌈horizon / 12⌉ times

    F->>F: Concatenate all chunks → trim to exact horizon length
```

**At each step:**
1. Inverse-scale the 12 predicted values (watts) for downstream use
2. Re-scale them back to feed as the target column of the next context window
3. Generate new `hour_sin/cos`, `day_of_year_sin/cos` for the future timestamps via
   `add_cyclical_time_features()` — these are computed from timestamps, not predicted

### Training (`train/train_lstm.py`)

```mermaid
flowchart TD
    A["Load energy_generated.csv\n5 columns, ~122k rows"] --> B["Split: train / val / test\n68% / 12% / 20%"]
    B --> C["Fit StandardScaler on train target only\nSave → models/artifacts/lstm_scaler.pkl"]
    C --> D["Scale target column only\nTime features left as-is"]
    D --> E["create_lstm_sequences()\nSliding window: 168 in → 12 out\nX shape: N × 168 × 5\ny shape: N × 12"]
    E --> F["DataLoader with batch_size=128, shuffle=True"]

    F --> G{"run_optuna_search\nin config.yaml?"}

    G -->|true| H["Optuna study\n50 trials\nMedian pruner\nSearch: hidden_size, num_layers,\ndropout, lr, weight_decay, batch_size"]
    G -->|false| I["Single training run\nwith config.yaml params"]

    H --> I
    I --> J["Training loop\nAdamW optimizer\nMSELoss\nReduceLROnPlateau scheduler\nGradient clipping norm=1.0\nEarly stopping patience=15"]
    J --> K["Save best checkpoint\nmodels/artifacts/lstm_solar_generator.pth\n{model_state_dict, lstm_params_used, best_val_loss}"]
    J --> L["Save training history plot\nreports/figures/lstm_training_history_final_model.png"]

    style K fill:#f0fff4,stroke:#38a169
```

---

## 2. SARIMA — Cost & Consumption Forecasting

### Purpose

Generates multi-week forecasts of electricity cost ($/week) and consumption (kWh/week) with
confidence intervals.  The weekly SARIMA forecasts are later disaggregated to hourly resolution
by the alignment step in the app.

### Model (`models/sarima_model.py`)

A thin wrapper around `statsmodels.tsa.statespace.SARIMAX`.

```
SARIMA(p, d, q)(P, D, Q, s)

Cost model:        SARIMA(1,1,1)(1,1,0,52)
Consumption model: SARIMA(1,1,1)(1,1,0,52)

p=1: one autoregressive term
d=1: first-order non-seasonal differencing (makes series stationary)
q=1: one moving-average term
P=1: seasonal AR term
D=1: seasonal differencing
Q=0: no seasonal MA term
s=52: seasonal period = 52 weeks (annual cycle)
```

```mermaid
graph LR
    SERIES["Weekly time series\n~780 points (2010–2025)"]
    SERIES --> SARIMA["SARIMAX.fit()\ndisp=False\nenforce_stationarity=False\nenforce_invertibility=False"]
    SARIMA --> FITTED["Fitted SARIMAXResults object"]
    FITTED --> FORECAST["get_forecast(steps=N)\nReturns:\n  predicted_mean: pd.Series\n  conf_int: pd.DataFrame (lower, upper 95% CI)"]
    FITTED --> RESID["resid\nIn-sample residuals\nPlotted as ACF diagnostic"]
```

### Training (`train/train_sarima.py`)

```mermaid
flowchart TD
    A["Load energy_cost.csv\nLoad energy_consumed.csv"] --> B["Train on full series\n(no train/test split — forecasts extend beyond the data end)"]
    B --> C["SARIMAModel.train(series)\nStores SARIMAXResults in sarima_instance.model_fit"]
    C --> D["Save whole SARIMAModel instance\njoblib.dump → models/artifacts/sarima_{cost|consumed}_model.pkl"]
    C --> E["Plot in-sample residual ACF\nplot_residuals_autocorrelation()\nSave → reports/figures/sarima_*_insample_residuals_acf.png"]
```

**Why save the whole instance instead of just `model_fit`?**  
The `SARIMAModel` wrapper stores the orders (`order`, `seasonal_order`) alongside the fitted
results.  Saving the whole object means you get those metadata fields back on load without
needing to re-read config.

---

## 3. Financial Evaluation (`utils/evaluation.py`)

Not a predictive model, but a deterministic simulation layer that sits on top of the two
forecasting models.

```mermaid
flowchart TD
    INPUT["Inputs:\n─ aligned_forecast_df (hourly: generation_kw, cost_per_kwh, consumption_kwh)\n─ user system: num_panels, panel_capacity_kw\n─ reference system: num_panels, panel_capacity_w\n─ panel_efficiency, export_tariff_rate"]

    INPUT --> SCALE["Scale generation to user system\nscaling_factor = user_total_kWp / ref_total_kWp\nuser_gen_kw = generation_kw × scaling_factor\nactual_gen_kwh = user_gen_kw × panel_efficiency"]

    SCALE --> BALANCE["Hourly energy balance\nself_consumption = min(gen_kwh, consumption_kwh)\nexported_kwh = max(0, gen_kwh − self_consumption)\nimported_kwh = max(0, consumption_kwh − gen_kwh)"]

    BALANCE --> FINANCE["Financial calculation (per hour)\ncost_without_solar = consumption_kwh × cost_per_kwh\ncost_of_import = imported_kwh × cost_per_kwh\nrevenue_from_export = exported_kwh × export_tariff_rate"]

    FINANCE --> AGGREGATE["Aggregate over all hours\nnet_savings = cost_without_solar − (cost_of_import − revenue_from_export)"]

    AGGREGATE --> OUTPUT["Output dict:\n─ total_energy_generated_kwh_user_system\n─ energy_self_consumed_kwh\n─ energy_exported_kwh / imported_kwh\n─ cost_without_solar_usd\n─ cost_with_solar_usd\n─ revenue_from_export_usd\n─ net_savings_usd_comprehensive\n─ average_import_price_forecasted\n─ net_savings_usd_simplified (simplified estimate)"]

    style INPUT fill:#e8f4fd,stroke:#2b6cb0
    style OUTPUT fill:#f0fff4,stroke:#38a169
```
