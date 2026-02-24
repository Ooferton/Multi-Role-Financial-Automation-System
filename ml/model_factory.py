from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseModel(nn.Module, ABC):
    """
    Abstract base class for all financial ML models.
    """
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
    
    @abstractmethod
    def forward(self, x):
        pass

class LSTMForecaster(BaseModel):
    """
    LSTM-based time-series forecaster.
    """
    def __init__(self, config: Dict):
        super().__init__(config)
        self.input_size = config.get('input_size', 10)
        self.hidden_size = config.get('hidden_size', 64)
        self.num_layers = config.get('num_layers', 2)
        
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(self.hidden_size, 1)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        out, _ = self.lstm(x)
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

class ModelFactory:
    """
    Factory to instantiate ML models based on configuration.
    """
    @staticmethod
    def create_model(model_type: str, config: Dict) -> nn.Module:
        if model_type == "LSTM":
            return LSTMForecaster(config)
        elif model_type == "TRANSFORMER":
            # Placeholder for Transformer implementation
            raise NotImplementedError("Transformer not yet implemented")
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    @staticmethod
    def load_model(path: str, model_type: str, config: Dict) -> nn.Module:
        model = ModelFactory.create_model(model_type, config)
        model.load_state_dict(torch.load(path))
        model.eval()
        return model
