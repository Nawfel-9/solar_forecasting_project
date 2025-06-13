# solar_forecasting_project/train/train_lstm.py

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import copy
from tqdm import tqdm
import optuna

# --- PROJECT SETUP ---
# Append project root to sys.path to allow for module imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Import all necessary modules from the project
from models.lstm_model import LSTMAttention
from utils.config_loader import load_config, get_model_path
from utils.preprocessing import (
    prepare_train_test,
    fit_scaler_and_save,
    scale_data,
    create_lstm_sequences,
    get_device
)
from utils.visualisation import plot_lstm_training_history

# Load configuration at the start
CONFIG = load_config()


def _run_training_loop(params: dict, train_loader: DataLoader, val_loader: DataLoader, device: torch.device) -> tuple[float, dict, dict]:
    """
    Internal helper function to run the core training/validation loop.
    This version includes detailed per-epoch logging to the console.

    Returns:
        tuple[float, dict, dict]: A tuple containing:
            - The best validation loss achieved.
            - The state dictionary of the best performing model.
            - The training history dictionary (losses and learning rate).
    """
    model = LSTMAttention(
        input_size=params['input_size'],
        hidden_size=params['hidden_size'],
        num_layers=params['num_layers'],
        dropout=params['dropout'],
        bidirectional=params.get('bidirectional', True),
        output_chunk_size=params['output_chunk_size']
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=params['learning_rate'], weight_decay=params.get('weight_decay', 0))
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, min_lr=1e-7)

    history = {'train_loss': [], 'val_loss': [], 'lr': []}
    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0
    early_stopping_patience = CONFIG['training_params'].get('early_stopping_patience', 20)

    for epoch in range(params['epochs']):
        model.train()
        epoch_train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{params['epochs']} [Train]", unit="batch", ncols=110, leave=False)
        for batch_X, batch_y in train_pbar:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_train_loss += loss.item()
            train_pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_epoch_train_loss = epoch_train_loss / len(train_loader) if len(train_loader) > 0 else 0
        history['train_loss'].append(avg_epoch_train_loss)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        avg_epoch_val_loss = float('inf')
        if val_loader:
            model.eval()
            epoch_val_loss = 0.0
            with torch.no_grad():
                for batch_X_val, batch_y_val in val_loader:
                    batch_X_val, batch_y_val = batch_X_val.to(device), batch_y_val.to(device)
                    outputs_val = model(batch_X_val)
                    val_loss = criterion(outputs_val, batch_y_val)
                    epoch_val_loss += val_loss.item()
            avg_epoch_val_loss = epoch_val_loss / len(val_loader) if len(val_loader) > 0 else float('inf')
            scheduler.step(avg_epoch_val_loss)
            
            if avg_epoch_val_loss < best_val_loss:
                best_val_loss = avg_epoch_val_loss
                best_model_state = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            
            history['val_loss'].append(avg_epoch_val_loss)
            
            if epochs_no_improve >= early_stopping_patience:
                # Log the final epoch's stats before breaking
                val_loss_display = f"{avg_epoch_val_loss:.6f}"
                best_val_loss_display = f"{best_val_loss:.6f}"
                patience_display = f"{epochs_no_improve}/{early_stopping_patience}"
                print(f"Epoch [{epoch+1:03d}] | Train Loss: {avg_epoch_train_loss:.6f} | Val Loss: {val_loss_display} | Best Val Loss: {best_val_loss_display} | Patience: {patience_display}")
                print(f"Early stopping at epoch {epoch+1}.")
                break
        else:
             history['val_loss'].append(None) # No validation loss to record
        
        # Per-epoch detailed logging
        val_loss_display = f"{avg_epoch_val_loss:.6f}" if val_loader and avg_epoch_val_loss != float('inf') else "N/A"
        best_val_loss_display = f"{best_val_loss:.6f}" if val_loader and best_val_loss != float('inf') else "N/A"
        patience_display = f"{epochs_no_improve}/{early_stopping_patience}" if val_loader else "N/A"
        print(f"Epoch [{epoch+1:03d}] | Train Loss: {avg_epoch_train_loss:.6f} | Val Loss: {val_loss_display} | Best Val Loss: {best_val_loss_display} | Patience: {patience_display}")

    # If no validation set was used, save the final model state
    if not val_loader and best_model_state is None:
        best_model_state = copy.deepcopy(model.state_dict())

    return best_val_loss, best_model_state, history


def objective(trial: optuna.Trial) -> float:
    """
    The objective function for Optuna hyperparameter search.
    Defines search space, prepares data, and calls the training loop.
    """
    device = get_device()
    prep_cfg = CONFIG['preprocessing']
    lstm_params_cfg = CONFIG['lstm_params']
    train_cfg = CONFIG['training_params']
    
    # 1. Suggest Hyperparameters for this trial
    params = {
        'input_size': lstm_params_cfg['input_size'],
        'output_chunk_size': lstm_params_cfg['output_chunk_size'],
        'bidirectional': lstm_params_cfg.get('bidirectional', True),
        'epochs': train_cfg['lstm_epochs'], # Use fixed epochs for fair comparison
        'hidden_size': trial.suggest_categorical('hidden_size', [32, 64, 128, 256]),
        'num_layers': trial.suggest_int('num_layers', 1, 3),
        'dropout': trial.suggest_float('dropout', 0.1, 0.5),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256])
    }
    
    try:
        # 2. Load and Prepare Data (done inside each trial for variable batch_size)
        data_cfg = CONFIG['data_paths']
        generated_data_path = Path(data_cfg['preprocessed_dir']) / data_cfg['energy_generated_csv']
        target_column = data_cfg['lstm_target_column_name']
        full_df = pd.read_csv(generated_data_path, index_col=0, parse_dates=True)
        
        train_val_raw_df, _ = prepare_train_test(full_df, test_size=prep_cfg.get('test_size', 0.2))
        train_raw_df, val_raw_df = prepare_train_test(train_val_raw_df, test_size=prep_cfg.get('validation_size_from_train', 0.15))

        scaler = fit_scaler_and_save(train_raw_df[target_column], scaler_filename_key='lstm_scaler_name')
        
        train_scaled_df = train_raw_df.copy()
        train_scaled_df[target_column] = scale_data(train_raw_df[target_column], scaler)
        val_scaled_df = val_raw_df.copy()
        if not val_raw_df.empty: val_scaled_df[target_column] = scale_data(val_raw_df[target_column], scaler)

        time_steps = prep_cfg['lstm_time_steps']
        X_train, y_train = create_lstm_sequences(train_scaled_df.values, time_steps, params['output_chunk_size'])
        X_val, y_val = create_lstm_sequences(val_scaled_df.values, time_steps, params['output_chunk_size']) if not val_scaled_df.empty else (np.array([]), np.array([]))

        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True, drop_last=True)
        val_loader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)), batch_size=params['batch_size'], shuffle=False) if X_val.size > 0 else None
        
        # 3. Run training and get the validation loss to be optimized
        validation_loss, _, _ = _run_training_loop(params, train_loader, val_loader, device)
        
        trial.report(validation_loss, 0)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
        return validation_loss

    except Exception as e:
        print(f"Trial failed with error: {e}")
        return float('inf') # Return a high loss value to signal failure


def run_optuna_study():
    """
    Initializes and runs the Optuna study to find the best hyperparameters.
    """
    n_trials = CONFIG['training_params'].get('optuna_n_trials', 50)
    print(f"Starting Optuna hyperparameter search for {n_trials} trials...")
    
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=n_trials)

    print("\n--- Optuna Search Complete ---")
    print(f"Number of finished trials: {len(study.trials)}")
    print(f"Best trial value (validation loss): {study.best_value:.6f}")
    print("Best Parameters Found: ")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")
    
    print("\nUpdate your config.yaml with these values and run the script again with 'run_optuna_search: false' to train and save the final model.")


def train_lstm_with_config_params():
    """
    Runs a single, standard training session using the parameters defined in config.yaml.
    Saves the best model and a plot of the training history.
    """
    print("Starting single training run with parameters from config.yaml...")
    device = get_device()
    data_cfg = CONFIG['data_paths']
    prep_cfg = CONFIG['preprocessing']
    lstm_params_cfg = CONFIG['lstm_params']
    train_cfg = CONFIG['training_params']
    
    # Combine all parameters into a single dict for the training loop
    params = {**lstm_params_cfg, **train_cfg}
    params['epochs'] = train_cfg['lstm_epochs'] # Ensure correct epochs key from training_params
    params['learning_rate'] = train_cfg['lstm_learning_rate']
    params['weight_decay'] = train_cfg.get('lstm_weight_decay', 0)
    
    generated_data_path = Path(data_cfg['preprocessed_dir']) / data_cfg['energy_generated_csv']
    target_column = data_cfg['lstm_target_column_name']
    
    # Load and prepare data
    full_df = pd.read_csv(generated_data_path, index_col=0, parse_dates=True)
    train_val_raw_df, _ = prepare_train_test(full_df, test_size=prep_cfg.get('test_size', 0.2))
    train_raw_df, val_raw_df = prepare_train_test(train_val_raw_df, test_size=prep_cfg.get('validation_size_from_train', 0.15))

    scaler = fit_scaler_and_save(train_raw_df[target_column], scaler_filename_key='lstm_scaler_name')
    
    train_scaled_df = train_raw_df.copy()
    train_scaled_df[target_column] = scale_data(train_raw_df[target_column], scaler)
    
    val_scaled_df = val_raw_df.copy()
    if not val_raw_df.empty:
        val_scaled_df[target_column] = scale_data(val_raw_df[target_column], scaler)

    time_steps = prep_cfg['lstm_time_steps']
    output_chunk_size = lstm_params_cfg['output_chunk_size']
    X_train, y_train = create_lstm_sequences(train_scaled_df.values, time_steps, output_chunk_size)
    X_val, y_val = create_lstm_sequences(val_scaled_df.values, time_steps, output_chunk_size) if not val_scaled_df.empty else (np.array([]), np.array([]))

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=train_cfg['lstm_batch_size'], shuffle=True, drop_last=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)), batch_size=train_cfg['lstm_batch_size'], shuffle=False) if X_val.size > 0 else None

    # Run the main training loop and get all return values
    best_val_loss, best_model_state, training_history = _run_training_loop(params, train_loader, val_loader, device)
    
    # Save the best model found
    if best_model_state:
        model_save_path = get_model_path('lstm_model_name')
        torch.save({
            'model_state_dict': best_model_state,
            'lstm_params_used': lstm_params_cfg,
            'best_val_loss': best_val_loss,
        }, model_save_path)
        print(f"\nBest LSTM model state saved to {model_save_path} (Val Loss: {best_val_loss:.6f})")
    else:
        print("\nWarning: No best model state was captured to save.")

    # Plot the training history and save the figure
    history_fig = plot_lstm_training_history(training_history, title=f"LSTM Training - {data_cfg['lstm_model_name']}")
    if history_fig:
        plot_save_path = Path(CONFIG['data_paths']['models_dir']) / "lstm_training_history_final_model.png"
        history_fig.savefig(plot_save_path)
        print(f"LSTM training history plot saved to {plot_save_path}")
        plt.close(history_fig)
        
    print("LSTM training process complete.")


if __name__ == "__main__":
    # Main execution block: checks the config to decide which mode to run
    should_run_optuna = CONFIG.get('training_params', {}).get('run_optuna_search', False)

    if should_run_optuna:
        run_optuna_study()
    else:
        train_lstm_with_config_params()