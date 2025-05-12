import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=3, dropout=0.2, weight_decay=1e-5):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer with dropout between layers (if multiple layers)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True,
                           dropout=dropout if num_layers > 1 else 0)
        
        # Regularization
        self.dropout = nn.Dropout(dropout)
        self.weight_decay = weight_decay

        # Final dense layer
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))
        
        # Apply dropout and ReLU to last time step output
        out = self.dropout(out[:, -1, :])
        out = F.relu(out)
        out = self.fc(out)
        return out

    @staticmethod
    def forecast_series(model, last_input, forecast_steps):
        """Generate multi-step forecast (compatible with main.py)"""
        model.eval()
        forecasts = []
        current_input = last_input.clone()
        
        with torch.no_grad():
            for _ in range(forecast_steps):
                pred = model(current_input)
                forecasts.append(pred.item())
                current_input = torch.cat((current_input[:, 1:, :], pred.unsqueeze(1)), dim=1)
        
        return forecasts