# solar_forecasting_project/models/lstm_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Attention(nn.Module):
    """
    An Attention layer that calculates alignment scores between the LSTM's
    sequence of outputs to produce a weighted context vector.

    Args:
        hidden_size (int): The number of features in the hidden state of the LSTM.
        num_directions (int): 2 if the LSTM is bidirectional, 1 otherwise.
    """
    def __init__(self, hidden_size: int, num_directions: int):
        super(Attention, self).__init__()
        # The attention network, which learns to compute the energy score
        # It takes the LSTM output (hidden_size * num_directions) and maps it to a new space
        self.attn_net = nn.Sequential(
            nn.Linear(hidden_size * num_directions, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, lstm_outputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lstm_outputs (torch.Tensor): The output from all time steps of the LSTM layer.
                                         Shape: (batch_size, sequence_length, hidden_size * num_directions)
        Returns:
            torch.Tensor: The attention weights for each time step in the sequence.
                          Shape: (batch_size, sequence_length)
        """
        # Calculate energy scores for each LSTM output time step
        energy = self.attn_net(lstm_outputs).squeeze(2) # Shape: (batch_size, sequence_length)
        
        # Apply softmax to get a probability distribution (attention weights)
        return F.softmax(energy, dim=1)


class LSTMAttention(nn.Module):
    """
    A Long Short-Term Memory (LSTM) model with an integrated Attention mechanism,
    designed for Direct Multi-Step Time Series Forecasting with multiple features.
    """
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 dropout: float,
                 bidirectional: bool,
                 output_chunk_size: int):
        """
        Args:
            input_size (int): The number of features in the input (e.g., 5).
            hidden_size (int): The number of features in the LSTM's hidden state.
            num_layers (int): Number of recurrent layers.
            dropout (float): Dropout probability.
            bidirectional (bool): If True, becomes a bidirectional LSTM.
            output_chunk_size (int): The number of future time steps to predict at once (e.g., 24).
        """
        super(LSTMAttention, self).__init__()
        # Save key parameters
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        num_directions = 2 if bidirectional else 1
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Instantiate the separate Attention layer
        self.attention = Attention(hidden_size, num_directions)
        
        # The input to the final linear layer is the context vector from the attention mechanism
        fc_input_features = hidden_size * num_directions
        self.fc = nn.Linear(fc_input_features, output_chunk_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the LSTMAttention model.
        """
        # LSTM layer does not need an initial hidden state; it defaults to zeros
        # lstm_out shape: (batch_size, sequence_length, hidden_size * num_directions)
        lstm_out, _ = self.lstm(x)
        
        # Pass the LSTM outputs to the Attention layer to get weights
        # attention_weights shape: (batch_size, sequence_length)
        attention_weights = self.attention(lstm_out)
        
        # Reshape weights to be (batch_size, 1, sequence_length) for batch matrix multiplication
        attention_weights = attention_weights.unsqueeze(1)
        
        # Compute the context vector by taking a weighted sum of LSTM outputs
        # torch.bmm performs batch matrix multiplication: (b, 1, n) @ (b, n, f) -> (b, 1, f)
        context_vector = torch.bmm(attention_weights, lstm_out).squeeze(1) # Shape: (batch_size, hidden_size * num_directions)
        
        # Pass the rich context vector to the final layer to make a prediction
        out = self.fc(context_vector)
        
        # Apply ReLU to ensure non-negative output
        return torch.relu(out)