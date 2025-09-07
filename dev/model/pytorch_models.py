"""
PyTorch Model - accept GPU acceleration, way more engineering required than TensorFlow abstracted model builder :( 
GPU-optimized LSTM, GRU, RNN models with flexible layer sizes

https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.module.register_module_forward_hook.html
"""

import torch
import torch.nn as nn
from typing import List, Optional

# Cell type mapping
_CELL = {"lstm": nn.LSTM, "gru": nn.GRU, "rnn": nn.RNN, "bilstm": nn.LSTM}

class PyTorchBaseModel(nn.Module):
    """
    PyTorch model for time series prediction:
    """
    def __init__(self, input_size: int, model_name: str = "lstm", layers: Optional[List[int]] = None,
                 dropout_rate: float = 0.2):
        super().__init__()
        
        # Model configuration
        self.input_size = int(input_size)
        self.model_name = model_name.lower()
        self.layers = layers or [64, 32]
        self.dropout_rate = float(dropout_rate)
        
        # Determine if bidirectional
        self.bidirectional = (self.model_name == 'bilstm')
        
        # Build RNN blocks
        self.rnn_blocks = nn.ModuleList()
        in_size = self.input_size
        
        Cell = _CELL[self.model_name]
        for h in self.layers:
            self.rnn_blocks.append(
                Cell(
                    input_size=in_size,
                    hidden_size=int(h),
                    num_layers=1,
                    batch_first=True,
                    bidirectional=self.bidirectional,
                    dropout=0.0  # Manual dropout between layers
                )
            )
            in_size = int(h) * (2 if self.bidirectional else 1)
        
        # Dropout between layers
        self.inter_dropout = nn.Dropout(self.dropout_rate) if len(self.layers) > 1 and self.dropout_rate > 0 else nn.Identity()
        
        # Final prediction layer
        self.fc = nn.Linear(in_size, 1)
    
    def forward(self, x):
        """Forward pass through the model."""
        out = x
        last_idx = len(self.rnn_blocks) - 1
        
        for i, rnn_block in enumerate(self.rnn_blocks):
            out, _ = rnn_block(out)    # (B, T, hidden*directions)
            if i < last_idx:
                out = self.inter_dropout(out)
        
        feats = out[:, -1, :]          # Take last timestep
        return self.fc(feats)

def pytorch_model_factory(
    input_size: int,
    model_name: str = 'lstm',
    layers: Optional[List[int]] = None,
    dropout_rate: float = 0.2
) -> PyTorchBaseModel:
    """Factory function to create PyTorch models."""
    return PyTorchBaseModel(
        input_size=input_size,
        model_name=model_name,
        layers=layers,
        dropout_rate=dropout_rate
    )