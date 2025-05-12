import torch
import torch.nn as nn

class TransformerRegressor(nn.Module):
    def __init__(self, input_dim=1, d_model=64, nhead=4, num_layers=2, dropout=0.1, max_seq_len=512):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Learnable positional embeddings
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            dropout=dropout,
            batch_first=True  # Required to match (batch, seq, dim) format
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, 1)

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        batch_size, seq_len, _ = x.size()
        
        # Project inputs
        x = self.input_proj(x)  # -> (batch, seq_len, d_model)
        
        # Create and add positional encoding
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        pos_encoded = self.pos_embedding(positions)  # -> (batch, seq_len, d_model)
        x = x + pos_encoded
        
        # Transformer
        x = self.transformer(x)  # -> (batch, seq_len, d_model)
        
        # Predict
        return self.output_proj(x).squeeze(-1)  # -> (batch, seq_len)
