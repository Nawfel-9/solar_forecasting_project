# solar_forecasting_project/train/train_sarima.py
import sys
from pathlib import Path
import pandas as pd
import joblib # For saving SARIMA models
import matplotlib.pyplot as plt # For saving plots

# Append project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path: # Avoid adding duplicates
    sys.path.append(str(PROJECT_ROOT))

from models.sarima_model import SARIMAModel
from utils.config_loader import load_config, get_model_path
from utils.visualisation import plot_residuals_autocorrelation # For plotting residuals

CONFIG = load_config()

def train_and_save_sarima_model(
    full_series: pd.Series, 
    model_name_key: str, 
    sarima_order_params: dict,
    series_name_for_log: str):
    """
    Helper function to train a SARIMA model on the full provided series,
    save the model instance, and plot in-sample residuals.
    """
    print(f"\n--- Processing SARIMA for: {series_name_for_log} (using full dataset) ---")
    
    if full_series.empty:
        print(f"Error: Input series for {series_name_for_log} is empty. Skipping.")
        return

    # 1. Initialize and Train Model on Full Data
    sarima_instance = SARIMAModel(
        order=tuple(sarima_order_params['order']),
        seasonal_order=tuple(sarima_order_params['seasonal_order'])
    )
    print(f"Training {series_name_for_log} SARIMA model on full dataset with order={sarima_instance.order}, seasonal_order={sarima_instance.seasonal_order}...")
    try:
        # The .train() method of SARIMAModel stores the fitted model in sarima_instance.model_fit
        fitted_model_results = sarima_instance.train(full_series, disp=False) # Capture the SARIMAXResults object
    except Exception as e:
        print(f"Error training SARIMA model for {series_name_for_log}: {e}")
        import traceback
        print(traceback.format_exc())
        return
        
    # 2. Save Model Instance
    model_save_path = get_model_path(model_name_key)
    Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
    try:
        joblib.dump(sarima_instance, model_save_path) # Save the whole SARIMAModel instance
        print(f"{series_name_for_log} SARIMA model (wrapper instance, trained on full data) saved to {model_save_path}")
    except Exception as e:
        print(f"Error saving SARIMA model for {series_name_for_log} to {model_save_path}: {e}")
        return

    # 3. In-Sample Residual Analysis (Optional but recommended)
    if fitted_model_results and hasattr(fitted_model_results, 'resid'):
        residuals = fitted_model_results.resid
        if not residuals.empty:
            print(f"Plotting in-sample residuals ACF for {series_name_for_log}...")
            try:
                # Remove initial NaNs from residuals if differencing was used
                cleaned_residuals = residuals.dropna()
                if not cleaned_residuals.empty:
                    lags_for_acf = min(40, len(cleaned_residuals) // 2 - 1)
                    if lags_for_acf > 0 :
                        residuals_fig = plot_residuals_autocorrelation(cleaned_residuals, lags=lags_for_acf, model_name=f"{series_name_for_log} (In-sample)")
                        if residuals_fig:
                            plot_save_path = Path(CONFIG['data_paths']['models_dir']) / f"sarima_{series_name_for_log.lower().replace(' ', '_').replace('/', '_')}_insample_residuals_acf.png"
                            residuals_fig.savefig(plot_save_path)
                            print(f"In-sample residuals ACF plot for {series_name_for_log} saved to {plot_save_path}")
                            plt.close(residuals_fig)
                        else:
                            print(f"Could not generate in-sample residuals plot for {series_name_for_log} (plot function returned None).")
                    else:
                        print(f"Not enough in-sample residuals data points to plot ACF for {series_name_for_log}.")
                else:
                    print(f"Cleaned in-sample residuals for {series_name_for_log} are empty. Skipping ACF plot.")
            except Exception as e:
                print(f"Error during in-sample residuals plotting for {series_name_for_log}: {e}")
                import traceback
                print(traceback.format_exc())
        else:
            print(f"No residuals found in fitted model for {series_name_for_log}.")


def run_all_sarima_training():
    """
    Main function to orchestrate the training of SARIMA models for cost and consumption
    using their respective full historical datasets.
    """
    data_cfg = CONFIG['data_paths']
    sarima_params_cfg = CONFIG['sarima_params']
    
    # Loop through 'cost' and 'consumed' (or 'consumption' based on your config preference)
    # This assumes your config.yaml for sarima_params uses 'cost' and 'consumed' as keys.
    # If it uses 'consumption', change the second item in the list below.
    for series_key_in_config in ['cost', 'consumed']: # Ensure these keys match your sarima_params in config.yaml
        
        print(f"\nAttempting to train SARIMA model for: {series_key_in_config.capitalize()} using full dataset.")
        
        # Construct keys for data_paths and model_names
        csv_file_key = f'energy_{series_key_in_config}_csv' 
        model_name_key = f'sarima_{series_key_in_config}_model_name'
        series_name_for_log = f"Energy {series_key_in_config.capitalize()}"
        
        # Basic check if keys exist in config to provide clearer error messages
        if csv_file_key not in data_cfg:
            print(f"Error: Config key '{csv_file_key}' not found in data_paths for {series_name_for_log}.")
            continue
        if model_name_key not in data_cfg:
            print(f"Error: Config key '{model_name_key}' not found in data_paths for {series_name_for_log}.")
            continue
        if series_key_in_config not in sarima_params_cfg:
            print(f"Error: Config key '{series_key_in_config}' not found in sarima_params for {series_name_for_log}.")
            continue

        data_path = Path(data_cfg['preprocessed_dir']) / data_cfg[csv_file_key]
        
        try:
            current_series_df = pd.read_csv(data_path, index_col=0, parse_dates=True)
            if current_series_df.empty:
                print(f"Error: Data series from {data_path} for {series_name_for_log} is empty.")
                continue 
            
            if current_series_df.shape[1] == 1:
                current_series = current_series_df.iloc[:, 0]
            else: # Handle multi-column CSVs: take the first column, warn user.
                target_column_name = current_series_df.columns[0]
                print(f"Warning: CSV {data_path} for {series_name_for_log} has multiple columns. "
                      f"Using the first column: '{target_column_name}'. "
                      "Ensure this is the correct target variable for SARIMA training.")
                current_series = current_series_df.iloc[:, 0]

            # Explicitly set frequency if known (e.g., from config or inferred and confirmed)
            # This helps statsmodels and suppresses warnings. Example:
            # known_freq = CONFIG.get('data_settings', {}).get(f'{series_key_in_config}_freq', None)
            # if known_freq:
            #     current_series = current_series.asfreq(known_freq)
            # else:
            #     # If not specified, statsmodels will try to infer.
            #     # You might still get a warning if it infers.
            #     pass

            train_and_save_sarima_model( # Renamed helper function
                full_series=current_series, # Pass the full series
                model_name_key=model_name_key,
                sarima_order_params=sarima_params_cfg[series_key_in_config],
                series_name_for_log=series_name_for_log
            )
        except FileNotFoundError:
            print(f"Error: Data file not found at {data_path} for {series_name_for_log}.")
        except KeyError as e:
            print(f"Error: A configuration key was unexpectedly missing for {series_name_for_log}. Details: {e}")
        except Exception as e:
            print(f"An unexpected error occurred while processing {series_name_for_log}: {e}")
            import traceback
            print(traceback.format_exc())
    
    print("\nAll SARIMA training processes (on full data) attempted.")

if __name__ == "__main__":
    run_all_sarima_training() # Renamed main execution function