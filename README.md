# 🌞 Solar Panel Forecast & Savings Estimator

An end-to-end machine-learning pipeline that estimates the financial savings achievable from a
solar panel installation. The app forecasts solar generation, electricity consumption, and
electricity cost, then runs an hourly energy-balance simulation so users can size and evaluate
a virtual solar system interactively.

![Project banner](docs/Banner.png)

---

## Documentation

| Doc | Contents |
|---|---|
| [Architecture](docs/architecture.md) | System design, module map, config-driven design overview |
| [Data Pipeline](docs/data_pipeline.md) | Raw data → preprocessed CSVs, feature engineering, train/val/test splits |
| [Models](docs/models.md) | LSTM+Attention and SARIMA architecture, training flows |
| [Inference Pipeline](docs/inference_pipeline.md) | Step-by-step walkthrough of what happens when you click *Generate Forecast* |

---

## What the App Does

1. **Forecasts solar generation** using a bidirectional LSTM with Attention, trained on ~14 years of hourly irradiance data for New York City.
2. **Forecasts electricity cost and consumption** using SARIMA models, trained on NYC utility billing records.
3. **Runs a financial simulation**: for any solar system you configure (number of panels, capacity, efficiency), it calculates self-consumption, grid import, export revenue, and net savings over the chosen forecast horizon.
4. **Renders interactive charts** via Plotly (or static Matplotlib) showing historical and forecasted generation, cost, and consumption side by side.

---

## Quick Start

### Requirements

- Python 3.10 or newer
- A working [Conda](https://docs.conda.io/en/latest/miniconda.html) or virtualenv setup
- PyTorch installed separately (see below)

### 1 · Clone & create environment

```bash
git clone https://github.com/Nawfel-9/solar_forecasting_project
cd solar_forecasting_project

conda create -n solar_env python=3.11 -y
conda activate solar_env
```

### 2 · Install PyTorch

Choose the command that matches your hardware:

```bash
# NVIDIA GPU (CUDA)
pip install --no-cache-dir torch torchvision torchaudio

# AMD GPU (ROCm)
pip install --no-cache-dir torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/rocm6.3

# CPU only
pip install --no-cache-dir torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu
```

> `--no-cache-dir` prevents `ValueError: Memoryview is too large` from PyTorch's large wheel files.

### 3 · Install remaining dependencies

```bash
pip install -r requirements.txt
```

### 4 · Launch the app

```bash
streamlit run app.py
```

Opens at [http://localhost:8501](http://localhost:8501).  The LSTM checkpoint and scaler are
committed to the repo — no training needed to run the app.

> **SARIMA models are not committed.** The app will report an error for cost/consumption
> forecasts until you train them: `python train/train_sarima.py` (takes a few minutes).

---

## Using the App

### Sidebar controls

| Control | What it does |
|---|---|
| **Forecast Horizon** | 1 Month / 6 Months / 1 Year — controls how far ahead the models forecast |
| **Number of Solar Panels** | How many panels your hypothetical system has |
| **Capacity per Panel (kW)** | Rated capacity of each panel |
| **System Efficiency (%)** | Combined loss factor (panel degradation, inverter, wiring) |
| **Export Tariff ($/kWh)** | Revenue earned for surplus energy sent back to the grid |
| **Chart Engine** | Plotly (interactive zoom/hover) or Matplotlib (static) |

### Tabs

- **Forecast Dashboard** — click *Generate Forecast & Estimate Savings* to run the full pipeline; shows KPI metrics, visual forecasts, and a detailed financial breakdown.
- **Historical Data** — browse the raw historical generation, cost, and consumption series.
- **Config Info** — inspect the key parameters loaded from `config.yaml` (model architecture, SARIMA orders, reference system specs).

---

## (Optional) Rebuild from Raw Data

Only needed if you want to regenerate the preprocessed CSVs or retrain models from scratch.

### Get the raw datasets

Place in `data/`:

| File | Source |
|---|---|
| `generated_2009_2023.csv` | Kaggle — *Solar Power Generation and Consumption Dataset* |
| `Electric_Consumption_And_Cost__2010_-_Feb_2025__20250311.csv` | NYC Open Data |

### Run preprocessing

```bash
python scripts/build_generation_data.py   # → data/preprocessed/energy_generated.csv
python scripts/build_consumption_data.py  # → data/preprocessed/energy_consumed.csv
                                          #    data/preprocessed/energy_cost.csv
```

### Train models

```bash
python train/train_sarima.py   # fast (~minutes)
python train/train_lstm.py     # slow; GPU recommended
```

To tune the LSTM automatically with Optuna, set `training_params.run_optuna_search: true` in
`config.yaml` before running `train_lstm.py`.

---

## Project Layout

```
solar_forecasting_project/
├── app.py                         ← Streamlit app (entry point)
├── config.yaml                    ← All paths, hyperparameters, and app defaults
├── requirements.txt
├── scripts/                       ← One-time data preparation
├── models/                        ← Model code + trained artifacts
│   └── artifacts/                 ← lstm_solar_generator.pth, lstm_scaler.pkl, …
├── train/                         ← Training scripts
├── utils/                         ← Shared utilities (preprocessing, evaluation, visualisation)
├── data/preprocessed/             ← Pre-built CSVs (committed)
├── reports/figures/               ← Training and diagnostic plots
├── docs/                          ← Architecture and pipeline documentation
└── notebook/                      ← Exploratory analysis notebook
```

---

## Technology Stack

| Category | Libraries |
|---|---|
| Deep learning | PyTorch (LSTM + Attention) |
| Statistical modelling | Statsmodels (SARIMA), Scikit-learn |
| Solar physics simulation | PVLib |
| Hyperparameter optimisation | Optuna |
| Web app | Streamlit |
| Visualisation | Plotly, Matplotlib |
| Data | Pandas, NumPy |
