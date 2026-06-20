**Navigation:** [README](../README.md) · [Architecture](architecture.md) · [Models](models.md) · [Inference Pipeline](inference_pipeline.md)

---

# Data Pipeline

## Overview

There are two independent data pipelines — one for solar generation and one for electricity
consumption/cost. Both run through dedicated scripts and produce the preprocessed CSVs that
the training and app code consumes.

---

## Generation Pipeline

Converts raw meteorological data into a multi-feature hourly DataFrame suitable for the LSTM.

```mermaid
flowchart TD
    A["Raw CSV: generated_2009_2023.csv\n~14 years of hourly weather data\nColumns: Temperature, GHI, DNI, DHI,\nSolar Zenith Angle, Year, Month, Day, Hour"]

    A --> B["Normalise DatetimeIndex\nIf no DatetimeIndex → build from Year/Month/Day/Hour columns"]

    B --> C["Compute Plane-of-Array Irradiance\npvlib.irradiance.get_total_irradiance()\nsurface_tilt=35°, azimuth=180°, albedo=0.18\nOutputs: poa_global, poa_diffuse, poa_direct"]

    C --> D["Compute Cell Temperature\npvlib.temperature.sapm_cell()\npoa_global + ambient_temp + wind_speed=1 m/s\nNOCT=45°C"]

    D --> E["Compute DC Power — PVWatts model\npvlib.pvsystem.pvwatts_dc()\npdc0 = 750 W × 10,000 panels = 7.5 MW\ngamma_pdc = −0.0037 %/°C\nClamp to ≥ 0 (no negative generation)"]

    E --> F["Create target column\n'Generated Energy W' = DC power series"]

    F --> G["Add cyclical time features\nhour_sin = sin(2π × hour / 24)\nhour_cos = cos(2π × hour / 24)\nday_of_year_sin = sin(2π × doy / 365.25)\nday_of_year_cos = cos(2π × doy / 365.25)"]

    G --> H["Reorder columns\n['Generated Energy W', hour_sin, hour_cos,\n day_of_year_sin, day_of_year_cos]"]

    H --> I["Save: data/preprocessed/energy_generated.csv\nHourly, 5 columns, ~122,000 rows"]

    style A fill:#f0f4ff,stroke:#4a6cf7
    style I fill:#f0fff4,stroke:#38a169
```

**Why cyclical encoding?** A plain integer hour column jumps from 23 → 0 at midnight —
a large discontinuity that looks like a meaningful signal to the model. Sine/cosine pairs
encode hour as a smooth circle: hour 23 and hour 0 are neighbours in the encoding space.

---

## Consumption & Cost Pipeline

Disaggregates monthly utility billing records into synthetic weekly data.

```mermaid
flowchart TD
    A["Raw CSV: Electric_Consumption_And_Cost…csv\nMonthly utility records, NYC open data\nColumns: Revenue Month, Current Charges, Consumption KWH"]

    A --> B["Select & rename columns\nRevenue Month → Datetime\nCurrent Charges → Cost $\nKeep: Consumption KWH"]

    B --> C["Parse dates, set index\npd.to_datetime on Revenue Month\nGroup by index.sum() to collapse\nany duplicate billing rows"]

    C --> D{"For each monthly value…"}

    D --> E["Draw number of weeks: 4 or 5\nnp.random.choice 0 or 1 + 4\n seed=42 for reproducibility"]

    E --> F["Generate variation factors\nnp.random.normal mean=1.0, std=0.15, n=weeks\nNormalise: factors = factors / factors.sum()\nGuarantees weekly values sum to the monthly total"]

    F --> G["weekly_value = monthly_value × factor\nAppend to list with current_date\ncurrent_date += timedelta(days=7)"]

    G --> D

    D --> H["pd.Series of weekly values\nwith 7-day DatetimeIndex starting 2010-01-01"]

    H --> I["Save: data/preprocessed/energy_consumed.csv\nWeekly kWh consumption, ~780 rows"]
    H --> J["Save: data/preprocessed/energy_cost.csv\nWeekly $ cost, ~780 rows"]

    style A fill:#f0f4ff,stroke:#4a6cf7
    style I fill:#f0fff4,stroke:#38a169
    style J fill:#f0fff4,stroke:#38a169
```

**Why disaggregate to weekly?** SARIMA requires a consistent frequency. Monthly data has only
~180 points (2010–2025), which is marginal for a seasonal ARIMA with `s=12`. Expanding to
~780 weekly points with `s=52` provides a much richer training set.

---

## Feature Engineering Detail

`add_cyclical_time_features()` in `utils/preprocessing.py` — called by both the generation
script (to build the training CSV) and by the LSTM inference loop (to generate features for
future time steps on the fly).

```mermaid
graph LR
    IDX["DatetimeIndex\nhourly"] --> HS["hour_sin\nsin(2π × h / 24)"]
    IDX --> HC["hour_cos\ncos(2π × h / 24)"]
    IDX --> DS["day_of_year_sin\nsin(2π × doy / 365.25)"]
    IDX --> DC["day_of_year_cos\ncos(2π × doy / 365.25)"]

    HS & HC --> HOUR_CIRCLE["Hour circle\n(0–23 maps to 0–2π)"]
    DS & DC --> YEAR_CIRCLE["Year circle\n(day 1–366 maps to 0–2π)"]
```

---

## Train/Validation/Test Split

Applied inside the training scripts using `prepare_train_test()` from `utils/preprocessing.py`.

```mermaid
gantt
    title Dataset splits (proportional, not to scale)
    dateFormat X
    axisFormat %s

    section Generation data (hourly)
    Training set (68%)   : 0, 68
    Validation set (12%) : 68, 80
    Test set (20%)       : 80, 100

    section Cost & Consumption (weekly)
    Full series used for SARIMA training : 0, 100
```

- `test_size = 0.20` (config: `preprocessing.test_size`)
- `validation_size_from_train = 0.15` (config: `preprocessing.validation_size_from_train`)
- SARIMA trains on the full series (there is no held-out test split at training time — forecasts
  are by definition out-of-sample since they extend beyond the end of the data).

---

## Scaler

A `StandardScaler` (zero mean, unit variance) is fit on the **training portion only** of the
`Generated Energy (W)` target column, then saved to `models/artifacts/lstm_scaler.pkl`.

At inference time the same scaler is used to:
1. Scale the last 168-hour context window before feeding it to the LSTM
2. Inverse-transform the LSTM's scaled predictions back to watts

The four time-feature columns (`hour_sin`, `hour_cos`, `day_of_year_sin`, `day_of_year_cos`)
are already in `[-1, 1]` by construction and are **not scaled**.
