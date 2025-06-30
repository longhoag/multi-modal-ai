"""Multi-modal fusion strategies for combining different modality features."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import math


class ConcatenationFusion(nn.Module):
    """Simple concatenation-based fusion."""
    
    def __init__(
        self,
        input_dims: Dict[str, int],
        hidden_size: int = 768,
        num_classes: Optional[int] = None,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        self.input_dims = input_dims
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        
        # Calculate total input dimension
        total_dim = sum(input_dims.values())
        
        # Fusion projection
        self.fusion_projection = nn.Sequential(
            nn.Linear(total_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
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
        features: Dict[str, torch.Tensor],
        return_features: bool = True,
        return_logits: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through concatenation fusion."""
        # Concatenate all features
        feature_list = [features[modality] for modality in sorted(features.keys())]
        concatenated = torch.cat(feature_list, dim=1)
        
        # Apply fusion projection
        fused_features = self.fusion_projection(concatenated)
        
        result = {}
        
        if return_features:
            result['features'] = fused_features
            result['raw_features'] = concatenated
        
        if return_logits and self.classifier is not None:
            result['logits'] = self.classifier(fused_features)
        
        return result


class AttentionFusion(nn.Module):
    """Attention-based fusion for multi-modal features."""
    
    def __init__(
        self,
        input_dims: Dict[str, int],
        hidden_size: int = 768,
        num_heads: int = 8,
        num_classes: Optional[int] = None,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        self.input_dims = input_dims
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.modalities = sorted(input_dims.keys())
        
        # Project each modality to hidden size
        self.modality_projections = nn.ModuleDict({
            modality: nn.Linear(dim, hidden_size)
            for modality, dim in input_dims.items()
        })
        
        # Multi-head attention for fusion
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Layer norm and feedforward
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        
        # Final projection
        self.final_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
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
        features: Dict[str, torch.Tensor],
        return_features: bool = True,
        return_logits: bool = False,
        return_attention_weights: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through attention fusion."""
        batch_size = next(iter(features.values())).size(0)
        
        # Project each modality
        projected_features = []
        for modality in self.modalities:
            if modality in features:
                proj = self.modality_projections[modality](features[modality])
                projected_features.append(proj.unsqueeze(1))  # [batch, 1, hidden]
        
        if not projected_features:
            raise ValueError("No valid modalities provided")
        
        # Stack features for attention: [batch, num_modalities, hidden]
        stacked_features = torch.cat(projected_features, dim=1)
        
        # Self-attention across modalities
        attended, attention_weights = self.attention(
            stacked_features, stacked_features, stacked_features
        )
        
        # Residual connection and layer norm
        attended = self.layer_norm1(attended + stacked_features)
        
        # Feedforward with residual connection
        ff_output = self.feedforward(attended)
        attended = self.layer_norm2(attended + ff_output)
        
        # Global pooling across modalities (mean pooling)
        fused_features = torch.mean(attended, dim=1)  # [batch, hidden]
        
        # Final projection
        fused_features = self.final_projection(fused_features)
        
        result = {}
        
        if return_features:
            result['features'] = fused_features
            result['raw_features'] = stacked_features
        
        if return_logits and self.classifier is not None:
            result['logits'] = self.classifier(fused_features)
        
        if return_attention_weights:
            result['attention_weights'] = attention_weights
        
        return result


class BilinearFusion(nn.Module):
    """Bilinear fusion for pairwise modality interactions."""
    
    def __init__(
        self,
        input_dims: Dict[str, int],
        hidden_size: int = 768,
        num_classes: Optional[int] = None,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        self.input_dims = input_dims
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.modalities = sorted(input_dims.keys())
        
        # Project each modality to hidden size
        self.modality_projections = nn.ModuleDict({
            modality: nn.Sequential(
                nn.Linear(dim, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            )
            for modality, dim in input_dims.items()
        })
        
        # Bilinear interaction layers for each pair of modalities
        self.bilinear_interactions = nn.ModuleDict()
        for i, mod1 in enumerate(self.modalities):
            for j, mod2 in enumerate(self.modalities):
                if i <= j:  # Avoid duplicate pairs
                    key = f"{mod1}_{mod2}"
                    self.bilinear_interactions[key] = nn.Bilinear(
                        hidden_size, hidden_size, hidden_size
                    )
        
        # Calculate fusion dimension
        num_modalities = len(self.modalities)
        num_pairs = (num_modalities * (num_modalities + 1)) // 2
        fusion_dim = num_modalities * hidden_size + num_pairs * hidden_size
        
        # Final fusion projection
        self.fusion_projection = nn.Sequential(
            nn.Linear(fusion_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size)
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
        features: Dict[str, torch.Tensor],
        return_features: bool = True,
        return_logits: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through bilinear fusion."""
        # Project each modality
        projected_features = {}
        for modality in self.modalities:
            if modality in features:
                projected_features[modality] = self.modality_projections[modality](
                    features[modality]
                )
        
        if not projected_features:
            raise ValueError("No valid modalities provided")
        
        # Collect individual features
        individual_features = list(projected_features.values())
        
        # Compute bilinear interactions
        interaction_features = []
        modality_list = list(projected_features.keys())
        
        for i, mod1 in enumerate(modality_list):
            for j, mod2 in enumerate(modality_list):
                if i <= j:
                    key1 = f"{mod1}_{mod2}"
                    key2 = f"{mod2}_{mod1}"
                    
                    if key1 in self.bilinear_interactions:
                        bilinear_key = key1
                    elif key2 in self.bilinear_interactions:
                        bilinear_key = key2
                    else:
                        continue
                    
                    interaction = self.bilinear_interactions[bilinear_key](
                        projected_features[mod1], projected_features[mod2]
                    )
                    interaction_features.append(interaction)
        
        # Concatenate individual and interaction features
        all_features = individual_features + interaction_features
        concatenated = torch.cat(all_features, dim=1)
        
        # Apply fusion projection
        fused_features = self.fusion_projection(concatenated)
        
        result = {}
        
        if return_features:
            result['features'] = fused_features
            result['raw_features'] = concatenated
        
        if return_logits and self.classifier is not None:
            result['logits'] = self.classifier(fused_features)
        
        return result


class CrossModalAttentionFusion(nn.Module):
    """Cross-modal attention fusion with query-key-value mechanism."""
    
    def __init__(
        self,
        input_dims: Dict[str, int],
        hidden_size: int = 768,
        num_heads: int = 8,
        num_layers: int = 2,
        num_classes: Optional[int] = None,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        self.input_dims = input_dims
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.modalities = sorted(input_dims.keys())
        
        # Project each modality to hidden size
        self.modality_projections = nn.ModuleDict({
            modality: nn.Linear(dim, hidden_size)
            for modality, dim in input_dims.items()
        })
        
        # Cross-attention layers
        self.cross_attention_layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.ModuleDict()
            for query_mod in self.modalities:
                for key_mod in self.modalities:
                    if query_mod != key_mod:
                        layer[f"{query_mod}_to_{key_mod}"] = nn.MultiheadAttention(
                            embed_dim=hidden_size,
                            num_heads=num_heads,
                            dropout=dropout_rate,
                            batch_first=True
                        )
            self.cross_attention_layers.append(layer)
        
        # Layer norms
        self.layer_norms = nn.ModuleList([
            nn.ModuleDict({
                modality: nn.LayerNorm(hidden_size)
                for modality in self.modalities
            })
            for _ in range(num_layers)
        ])
        
        # Final fusion
        self.final_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        
        self.final_norm = nn.LayerNorm(hidden_size)
        
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
        features: Dict[str, torch.Tensor],
        return_features: bool = True,
        return_logits: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through cross-modal attention fusion."""
        # Project each modality
        projected_features = {}
        for modality in self.modalities:
            if modality in features:
                proj = self.modality_projections[modality](features[modality])
                projected_features[modality] = proj.unsqueeze(1)  # [batch, 1, hidden]
        
        if not projected_features:
            raise ValueError("No valid modalities provided")
        
        # Cross-attention layers
        current_features = projected_features.copy()
        
        for layer_idx in range(self.num_layers):
            updated_features = {}
            
            for query_mod in current_features.keys():
                query = current_features[query_mod]
                attended_features = [query]  # Self-connection
                
                for key_mod in current_features.keys():
                    if query_mod != key_mod:
                        attention_key = f"{query_mod}_to_{key_mod}"
                        if attention_key in self.cross_attention_layers[layer_idx]:
                            key = value = current_features[key_mod]
                            attended, _ = self.cross_attention_layers[layer_idx][attention_key](
                                query, key, value
                            )
                            attended_features.append(attended)
                
                # Combine attended features
                if len(attended_features) > 1:
                    combined = torch.mean(torch.stack(attended_features), dim=0)
                else:
                    combined = attended_features[0]
                
                # Residual connection and layer norm
                updated_features[query_mod] = self.layer_norms[layer_idx][query_mod](
                    combined + query
                )
            
            current_features = updated_features
        
        # Final fusion across modalities
        all_features = torch.cat(list(current_features.values()), dim=1)  # [batch, num_mod, hidden]
        
        # Self-attention for final fusion
        fused, _ = self.final_attention(all_features, all_features, all_features)
        fused = self.final_norm(fused + all_features)
        
        # Global pooling
        fused_features = torch.mean(fused, dim=1)  # [batch, hidden]
        
        result = {}
        
        if return_features:
            result['features'] = fused_features
            result['raw_features'] = all_features
        
        if return_logits and self.classifier is not None:
            result['logits'] = self.classifier(fused_features)
        
        return result


class MultiModalFusion(nn.Module):
    """Unified multi-modal fusion with multiple strategy options."""
    
    def __init__(
        self,
        fusion_type: str = "attention",  # concatenation, attention, bilinear, cross_attention
        **kwargs
    ):
        super().__init__()
        
        self.fusion_type = fusion_type
        
        if fusion_type == "concatenation":
            self.fusion = ConcatenationFusion(**kwargs)
        elif fusion_type == "attention":
            self.fusion = AttentionFusion(**kwargs)
        elif fusion_type == "bilinear":
            self.fusion = BilinearFusion(**kwargs)
        elif fusion_type == "cross_attention":
            self.fusion = CrossModalAttentionFusion(**kwargs)
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
    
    def forward(self, *args, **kwargs):
        """Forward pass through selected fusion strategy."""
        return self.fusion(*args, **kwargs)


# Export classes
__all__ = [
    "MultiModalFusion",
    "ConcatenationFusion",
    "AttentionFusion",
    "BilinearFusion",
    "CrossModalAttentionFusion"
]
