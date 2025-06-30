"""Text encoder models for multi-modal AI."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModel, AutoTokenizer, AutoConfig,
    BertModel, RobertaModel, DistilBertModel
)
from typing import Dict, Optional, Union, List, Tuple
import warnings


class TransformerTextEncoder(nn.Module):
    """Transformer-based text encoder with multiple architecture support."""
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        hidden_size: int = 768,
        num_classes: Optional[int] = None,
        dropout_rate: float = 0.1,
        freeze_backbone: bool = False,
        pooling_strategy: str = "cls",  # cls, mean, max, attention
        fine_tune_layers: int = -1,  # -1 for all layers
    ):
        super().__init__()
        
        self.model_name = model_name
        self.hidden_size = hidden_size
        self.pooling_strategy = pooling_strategy
        self.num_classes = num_classes
        
        # Load pre-trained transformer
        try:
            self.config = AutoConfig.from_pretrained(model_name)
            self.transformer = AutoModel.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            warnings.warn(f"Could not load {model_name}: {e}")
            # Fallback to BERT
            self.config = AutoConfig.from_pretrained("bert-base-uncased")
            self.transformer = AutoModel.from_pretrained("bert-base-uncased")
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        # Get actual hidden size from config
        self.hidden_size = self.config.hidden_size
        
        # Freeze backbone if requested
        if freeze_backbone:
            self._freeze_backbone()
        elif fine_tune_layers > 0:
            self._freeze_partial_backbone(fine_tune_layers)
        
        # Pooling layer for attention-based pooling
        if self.pooling_strategy == "attention":
            self.attention_pooling = nn.MultiheadAttention(
                embed_dim=self.hidden_size,
                num_heads=8,
                dropout=dropout_rate,
                batch_first=True
            )
            self.attention_query = nn.Parameter(torch.randn(1, 1, self.hidden_size))
        
        # Classification head if num_classes is provided
        if num_classes:
            self.classifier = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(self.hidden_size, self.hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(self.hidden_size // 2, num_classes)
            )
        else:
            self.classifier = None
        
        # Feature projection layer
        self.feature_projection = nn.Sequential(
            nn.Linear(self.hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
    
    def _freeze_backbone(self):
        """Freeze all transformer parameters."""
        for param in self.transformer.parameters():
            param.requires_grad = False
    
    def _freeze_partial_backbone(self, fine_tune_layers: int):
        """Freeze all but the last N layers."""
        # Freeze embeddings
        for param in self.transformer.embeddings.parameters():
            param.requires_grad = False
        
        # Freeze early encoder layers
        total_layers = len(self.transformer.encoder.layer)
        freeze_layers = total_layers - fine_tune_layers
        
        for i in range(freeze_layers):
            for param in self.transformer.encoder.layer[i].parameters():
                param.requires_grad = False
    
    def _pool_features(
        self, 
        hidden_states: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Pool token-level features to sentence-level."""
        if self.pooling_strategy == "cls":
            # Use [CLS] token (first token)
            return hidden_states[:, 0]
        
        elif self.pooling_strategy == "mean":
            # Mean pooling with attention mask
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
                sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                return sum_embeddings / sum_mask
            else:
                return torch.mean(hidden_states, dim=1)
        
        elif self.pooling_strategy == "max":
            # Max pooling
            if attention_mask is not None:
                hidden_states = hidden_states.masked_fill(
                    attention_mask.unsqueeze(-1) == 0, -1e9
                )
            return torch.max(hidden_states, dim=1)[0]
        
        elif self.pooling_strategy == "attention":
            # Attention-based pooling
            batch_size = hidden_states.size(0)
            query = self.attention_query.expand(batch_size, -1, -1)
            
            if attention_mask is not None:
                key_padding_mask = ~attention_mask.bool()
            else:
                key_padding_mask = None
            
            attended, _ = self.attention_pooling(
                query, hidden_states, hidden_states,
                key_padding_mask=key_padding_mask
            )
            return attended.squeeze(1)
        
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
    
    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        return_features: bool = True,
        return_logits: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through text encoder."""
        # Get transformer outputs
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        
        # Pool features
        pooled_features = self._pool_features(
            outputs.last_hidden_state, 
            attention_mask
        )
        
        # Project features
        projected_features = self.feature_projection(pooled_features)
        
        result = {}
        
        if return_features:
            result['features'] = projected_features
            result['raw_features'] = pooled_features
        
        if return_logits and self.classifier is not None:
            result['logits'] = self.classifier(pooled_features)
        
        return result
    
    def encode_text(
        self, 
        texts: Union[str, List[str]], 
        max_length: int = 512,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """Encode text directly from strings."""
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        
        if device:
            encoded = {k: v.to(device) for k, v in encoded.items()}
        
        # Forward pass
        with torch.no_grad():
            outputs = self.forward(**encoded, return_features=True, return_logits=False)
        
        return outputs['features']


class CNNTextEncoder(nn.Module):
    """CNN-based text encoder for comparison."""
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 300,
        filter_sizes: List[int] = [3, 4, 5],
        num_filters: int = 100,
        hidden_size: int = 768,
        num_classes: Optional[int] = None,
        dropout_rate: float = 0.5
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Convolutional layers
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        
        # Feature dimension after concatenating all conv outputs
        conv_output_size = len(filter_sizes) * num_filters
        
        # Feature projection
        self.feature_projection = nn.Sequential(
            nn.Linear(conv_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Classification head
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
        input_ids: torch.Tensor,
        return_features: bool = True,
        return_logits: bool = False,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through CNN text encoder."""
        # Embedding
        embedded = self.embedding(input_ids)  # [batch_size, seq_len, embed_dim]
        embedded = self.dropout(embedded)
        
        # Transpose for conv1d: [batch_size, embed_dim, seq_len]
        embedded = embedded.transpose(1, 2)
        
        # Apply convolutions and max pooling
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(embedded))  # [batch_size, num_filters, conv_seq_len]
            pooled = F.max_pool1d(conv_out, kernel_size=conv_out.size(2))  # [batch_size, num_filters, 1]
            conv_outputs.append(pooled.squeeze(2))  # [batch_size, num_filters]
        
        # Concatenate all conv outputs
        concatenated = torch.cat(conv_outputs, dim=1)  # [batch_size, total_filters]
        
        # Project features
        features = self.feature_projection(concatenated)
        
        result = {}
        
        if return_features:
            result['features'] = features
            result['raw_features'] = concatenated
        
        if return_logits and self.classifier is not None:
            result['logits'] = self.classifier(features)
        
        return result


class TextEncoder(nn.Module):
    """Unified text encoder with multiple architecture options."""
    
    def __init__(
        self,
        encoder_type: str = "transformer",  # transformer, cnn
        **kwargs
    ):
        super().__init__()
        
        self.encoder_type = encoder_type
        
        if encoder_type == "transformer":
            self.encoder = TransformerTextEncoder(**kwargs)
        elif encoder_type == "cnn":
            self.encoder = CNNTextEncoder(**kwargs)
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
    
    def forward(self, *args, **kwargs):
        """Forward pass through selected encoder."""
        return self.encoder(*args, **kwargs)
    
    def encode_text(self, *args, **kwargs):
        """Encode text if supported by encoder."""
        if hasattr(self.encoder, 'encode_text'):
            return self.encoder.encode_text(*args, **kwargs)
        else:
            raise NotImplementedError(f"encode_text not implemented for {self.encoder_type}")


# Export classes
__all__ = [
    "TextEncoder",
    "TransformerTextEncoder", 
    "CNNTextEncoder"
]
