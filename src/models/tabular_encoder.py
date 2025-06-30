"""Tabular data encoder models for multi-modal AI."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple, Union
import math


class MLPTabularEncoder(nn.Module):
    """Multi-layer perceptron for tabular data encoding."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_sizes: List[int] = [512, 256],
        hidden_size: int = 768,
        num_classes: Optional[int] = None,
        dropout_rate: float = 0.2,
        activation: str = "relu",
        batch_norm: bool = True,
        residual_connections: bool = False
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_sizes = hidden_sizes
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.residual_connections = residual_connections
        
        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU()
        else:
            self.activation = nn.ReLU()
        
        # Build MLP layers
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if batch_norm else None
        self.dropouts = nn.ModuleList()
        
        # Input layer
        prev_size = input_dim
        for hidden_size_layer in hidden_sizes:
            self.layers.append(nn.Linear(prev_size, hidden_size_layer))
            if batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_size_layer))
            self.dropouts.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size_layer
        
        # Final feature projection
        self.feature_projection = nn.Linear(prev_size, hidden_size)
        
        # Classification head if num_classes is provided
        if num_classes:
            self.classifier = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_size, hidden_size // 2),
                self.activation,
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_size // 2, num_classes)
            )
        else:
            self.classifier = None
    
    def forward(
        self,
        tabular_data: torch.Tensor,
        return_features: bool = True,
        return_logits: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through MLP encoder."""
        x = tabular_data
        
        # Pass through hidden layers
        for i, layer in enumerate(self.layers):
            residual = x if self.residual_connections and x.size(-1) == layer.out_features else None
            
            x = layer(x)
            
            if self.batch_norms:
                x = self.batch_norms[i](x)
            
            x = self.activation(x)
            x = self.dropouts[i](x)
            
            if residual is not None:
                x = x + residual
        
        # Final feature projection
        features = self.feature_projection(x)
        
        result = {}
        
        if return_features:
            result['features'] = features
            result['raw_features'] = x
        
        if return_logits and self.classifier is not None:
            result['logits'] = self.classifier(features)
        
        return result


class TransformerTabularEncoder(nn.Module):
    """Transformer-based tabular encoder."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 768,
        num_heads: int = 8,
        num_layers: int = 4,
        num_classes: Optional[int] = None,
        dropout_rate: float = 0.1,
        max_seq_length: int = 1000
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_size)
        
        # Positional encoding (for feature positions)
        self.pos_encoding = PositionalEncoding(hidden_size, max_seq_length)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout_rate,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout_rate)
        )
        
        # Classification head if num_classes is provided
        if num_classes:
            self.classifier = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_size // 2, num_classes)
            )
        else:
            self.classifier = None
    
    def forward(
        self,
        tabular_data: torch.Tensor,
        return_features: bool = True,
        return_logits: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through transformer encoder."""
        batch_size = tabular_data.size(0)
        
        # Project input to hidden dimension
        x = self.input_projection(tabular_data.unsqueeze(1))  # [batch, 1, hidden_size]
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Pass through transformer
        x = self.transformer(x)  # [batch, 1, hidden_size]
        
        # Apply output projection and squeeze
        features = self.output_projection(x).squeeze(1)  # [batch, hidden_size]
        
        result = {}
        
        if return_features:
            result['features'] = features
            result['raw_features'] = x.squeeze(1)
        
        if return_logits and self.classifier is not None:
            result['logits'] = self.classifier(features)
        
        return result


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :].transpose(0, 1)


class EmbeddingTabularEncoder(nn.Module):
    """Tabular encoder with categorical embeddings."""
    
    def __init__(
        self,
        numerical_features: int,
        categorical_features: List[int],  # List of vocab sizes for each categorical feature
        embedding_dims: Optional[List[int]] = None,  # Custom embedding dimensions
        hidden_sizes: List[int] = [512, 256],
        hidden_size: int = 768,
        num_classes: Optional[int] = None,
        dropout_rate: float = 0.2
    ):
        super().__init__()
        
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        
        # Categorical embeddings
        self.embeddings = nn.ModuleList()
        embedding_total_dim = 0
        
        if embedding_dims is None:
            # Rule of thumb: embedding_dim = min(50, (vocab_size + 1) // 2)
            embedding_dims = [min(50, (vocab_size + 1) // 2) for vocab_size in categorical_features]
        
        for vocab_size, embed_dim in zip(categorical_features, embedding_dims):
            self.embeddings.append(nn.Embedding(vocab_size, embed_dim))
            embedding_total_dim += embed_dim
        
        # Input dimension is numerical features + embedding dimensions
        total_input_dim = numerical_features + embedding_total_dim
        
        # MLP for processing combined features
        self.mlp = MLPTabularEncoder(
            input_dim=total_input_dim,
            hidden_sizes=hidden_sizes,
            hidden_size=hidden_size,
            num_classes=num_classes,
            dropout_rate=dropout_rate
        )
    
    def forward(
        self,
        numerical_data: torch.Tensor,
        categorical_data: torch.Tensor,
        return_features: bool = True,
        return_logits: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through embedding encoder."""
        batch_size = numerical_data.size(0)
        
        # Process categorical features through embeddings
        embedded_features = []
        for i, embedding in enumerate(self.embeddings):
            cat_feature = categorical_data[:, i].long()
            embedded = embedding(cat_feature)
            embedded_features.append(embedded)
        
        # Concatenate all features
        if embedded_features:
            embedded_cat = torch.cat(embedded_features, dim=1)
            combined_features = torch.cat([numerical_data, embedded_cat], dim=1)
        else:
            combined_features = numerical_data
        
        # Pass through MLP
        return self.mlp(combined_features, return_features, return_logits)


class TabularEncoder(nn.Module):
    """Unified tabular encoder with multiple architecture options."""
    
    def __init__(
        self,
        encoder_type: str = "mlp",  # mlp, transformer, embedding
        **kwargs
    ):
        super().__init__()
        
        self.encoder_type = encoder_type
        
        if encoder_type == "mlp":
            self.encoder = MLPTabularEncoder(**kwargs)
        elif encoder_type == "transformer":
            self.encoder = TransformerTabularEncoder(**kwargs)
        elif encoder_type == "embedding":
            self.encoder = EmbeddingTabularEncoder(**kwargs)
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
    
    def forward(self, *args, **kwargs):
        """Forward pass through selected encoder."""
        return self.encoder(*args, **kwargs)


# Export classes
__all__ = [
    "TabularEncoder",
    "MLPTabularEncoder",
    "TransformerTabularEncoder", 
    "EmbeddingTabularEncoder"
]
