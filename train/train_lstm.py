import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt

from models.lstm_model import LSTM
from utils.preprocessing import load_energy_data, prepare_train_test, create_sequences
from utils.visualization import reset_losses

# ==================== CONFIGURATION ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_len = 24*30*12
forecast_horizon = 1
batch_size = 128
num_epochs = 100
learning_rate = 1e-4
hidden_size = 256
num_layers = 3
dropout = 0.3
weight_decay = 1e-5

# Loss trackers
train_losses = []
val_losses = []
test_losses = []
best_val_loss = float('inf')

# ==================== DATA LOADING ====================
__, __, generated = load_energy_data()
series = generated  # change to consumed or generated if needed
train_series, test_series = prepare_train_test(series)
train_series, val_series = prepare_train_test(train_series, test_size=0.2)

X_train, y_train = create_sequences(train_series.values, time_steps=input_len)
X_val, y_val = create_sequences(val_series.values, time_steps=input_len)
X_test, y_test = create_sequences(test_series.values, time_steps=input_len)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

# ==================== MODEL SETUP ====================
model = LSTM(input_size=1, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, weight_decay=weight_decay).to(device)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# ==================== TRAINING FUNCTIONS ====================
def train_one_epoch(epoch):
    model.train()
    epoch_loss = 0.0

    # Barre principale pour l'entraînement
    with tqdm(train_loader, unit="batch", leave=False,
              desc=f"Epoch {epoch+1}/{num_epochs} [Train]") as pbar:
        for x_batch, y_batch in pbar:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            # Forward + backward
            output = model(x_batch)
            loss = loss_function(output, y_batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # MàJ des métriques
            batch_loss = loss.item()
            epoch_loss += batch_loss
            avg_loss = epoch_loss / (pbar.n + 1)

            # Affichage dans la barre
            pbar.set_postfix({
                "loss": f"{batch_loss:.4f}",
                "avg": f"{avg_loss:.4f}"
            })

    avg_epoch_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_epoch_loss)
    tqdm.write(f"[Train] Epoch {epoch+1} – avg loss: {avg_epoch_loss:.4f}")
    return avg_epoch_loss


def validate_one_epoch(epoch):
    global best_val_loss
    model.eval()
    running_loss = 0.0
    all_targets, all_preds = [], []

    # Barre principale pour la validation
    with tqdm(val_loader, unit="batch", leave=False,
              desc=f"Epoch {epoch+1}/{num_epochs} [Val]") as pbar:
        for x_batch, y_batch in pbar:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            with torch.no_grad():
                outputs = model(x_batch)
                loss = loss_function(outputs, y_batch)

            # Collecte pour métriques finales
            all_targets.append(y_batch.cpu().numpy())
            all_preds.append(outputs.cpu().numpy())

            batch_loss = loss.item()
            running_loss += batch_loss
            avg_loss = running_loss / (pbar.n + 1)
            pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

    avg_val_loss = running_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    targets = np.concatenate(all_targets)
    preds = np.concatenate(all_preds)
    val_mae = np.mean(np.abs(preds - targets))
    val_rmse = np.sqrt(np.mean((preds - targets)**2))

    # Sauvegarde du meilleur modèle
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save({
            'model_state_dict': model.state_dict(),
            'val_loss': best_val_loss,
            'val_mae': val_mae,
            'val_rmse': val_rmse,
        }, 'models/best_model_energy_generated.pth')
        tqdm.write(
            f"[Val] Epoch {epoch+1} – Nouvelle meilleure performance ! "
            f"loss={avg_val_loss:.4f}, MAE={val_mae:.4f}, RMSE={val_rmse:.4f}"
        )
    else:
        tqdm.write(
            f"[Val] Epoch {epoch+1} – loss={avg_val_loss:.4f} "
            f"(best={best_val_loss:.4f}), MAE={val_mae:.4f}, RMSE={val_rmse:.4f}"
        )
    return avg_val_loss


def train_and_validate():
    for epoch in range(num_epochs):
        train_one_epoch(epoch)
        validate_one_epoch(epoch)

def test_model():
    """Final evaluation on test set using best model"""
    # Load best model
    model.load_state_dict(torch.load('models/best_model_energy_generated.pth', weights_only=True)['model_state_dict'])
    model.eval()
    
    test_loss = 0.0
    all_preds = []
    all_targets = []

    # Inference with progress bar
    with torch.no_grad():
        for x_batch, y_batch in tqdm(test_loader, desc="Testing", unit="batch"):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)
            
            loss = loss_function(outputs, y_batch)
            test_loss += loss.item()
            
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())

    # Calculate metrics
    avg_loss = test_loss / len(test_loader)
    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    
    metrics = {
        'loss': avg_loss,
        'mae': np.mean(np.abs(preds - targets)),
        'rmse': np.sqrt(np.mean((preds - targets)**2)),
        'R2': 1 - np.sum((targets - preds) ** 2) / np.sum((targets - np.mean(targets)) ** 2)
    }


    # Print results
    print("\n" + "="*40)
    print("FINAL TEST RESULTS")
    print("="*40)
    print(f"{'Loss:':<10}{metrics['loss']:.4f}")
    print(f"{'MAE:':<10}{metrics['mae']:.4f}")
    print(f"{'RMSE:':<10}{metrics['rmse']:.4f}")
    print(f"{'R²:':<10}{metrics['R2']:.4f}")
    print("="*40)

    # Plot last 100 points
    plt.figure(figsize=(12,5))
    plt.plot(targets[-100:], label='True', alpha=0.7)
    plt.plot(preds[-100:], label='Predicted', linestyle='--')
    plt.title("Test Predictions (Last 100 Points)")
    plt.legend()
    plt.show()

    return metrics
# ==================== MAIN ====================
if __name__ == "__main__":
    reset_losses(train_losses, val_losses, test_losses)
    train_and_validate()
    test_metrics = test_model()
