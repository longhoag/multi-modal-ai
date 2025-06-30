"""Image encoder models for multi-modal AI."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import timm
from typing import Dict, Optional, Union, List, Tuple
import warnings


class CNNImageEncoder(nn.Module):
    """CNN-based image encoder using pre-trained models."""
    
    def __init__(
        self,
        model_name: str = "resnet50",
        pretrained: bool = True,
        hidden_size: int = 768,
        num_classes: Optional[int] = None,
        dropout_rate: float = 0.1,
        freeze_backbone: bool = False,
        fine_tune_layers: int = -1,  # -1 for all layers
        pooling_type: str = "adaptive_avg",  # adaptive_avg, adaptive_max, attention
    ):
        super().__init__()
        
        self.model_name = model_name
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.pooling_type = pooling_type
        
        # Load pre-trained model
        try:
            if model_name.startswith('timm_'):
                # Use timm models
                timm_name = model_name.replace('timm_', '')
                self.backbone = timm.create_model(
                    timm_name, 
                    pretrained=pretrained,
                    num_classes=0  # Remove classification head
                )
                self.feature_dim = self.backbone.num_features
            else:
                # Use torchvision models
                self.backbone = getattr(models, model_name)(pretrained=pretrained)
                
                # Remove classification layers and get feature dimension
                if hasattr(self.backbone, 'classifier'):
                    if isinstance(self.backbone.classifier, nn.Linear):
                        self.feature_dim = self.backbone.classifier.in_features
                        self.backbone.classifier = nn.Identity()
                    else:
                        # For models like AlexNet with sequential classifier
                        self.feature_dim = self.backbone.classifier[-1].in_features
                        self.backbone.classifier = self.backbone.classifier[:-1]
                elif hasattr(self.backbone, 'fc'):
                    self.feature_dim = self.backbone.fc.in_features
                    self.backbone.fc = nn.Identity()
                elif hasattr(self.backbone, 'head'):
                    self.feature_dim = self.backbone.head.in_features
                    self.backbone.head = nn.Identity()
                else:
                    raise ValueError(f"Cannot determine feature dimension for {model_name}")
                    
        except Exception as e:
            warnings.warn(f"Could not load {model_name}: {e}")
            # Fallback to ResNet50
            self.backbone = models.resnet50(pretrained=pretrained)
            self.feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        
        # Freeze backbone if requested
        if freeze_backbone:
            self._freeze_backbone()
        elif fine_tune_layers > 0:
            self._freeze_partial_backbone(fine_tune_layers)
        
        # Adaptive pooling layers
        if pooling_type == "adaptive_avg":
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        elif pooling_type == "adaptive_max":
            self.global_pool = nn.AdaptiveMaxPool2d((1, 1))
        elif pooling_type == "attention":
            self.attention_pool = nn.Sequential(
                nn.Conv2d(self.feature_dim, self.feature_dim // 8, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.feature_dim // 8, 1, 1),
                nn.Sigmoid()
            )
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Feature projection layer
        self.feature_projection = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Classification head if num_classes is provided
        if num_classes:
            self.classifier = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_size // 2, num_classes)
            )
        else:
            self.classifier = None
    
    def _freeze_backbone(self):
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def _freeze_partial_backbone(self, fine_tune_layers: int):
        """Freeze all but the last N layers."""
        # Get all modules
        modules = list(self.backbone.children())
        freeze_modules = modules[:-fine_tune_layers] if fine_tune_layers > 0 else modules
        
        for module in freeze_modules:
            for param in module.parameters():
                param.requires_grad = False
    
    def _apply_attention_pooling(self, features: torch.Tensor) -> torch.Tensor:
        """Apply attention-based global pooling."""
        # features: [batch_size, channels, height, width]
        attention_weights = self.attention_pool(features)  # [batch_size, 1, height, width]
        
        # Apply attention weights
        attended_features = features * attention_weights
        
        # Global average pooling
        pooled = self.global_pool(attended_features)  # [batch_size, channels, 1, 1]
        return pooled.flatten(1)  # [batch_size, channels]
    
    def forward(
        self, 
        images: torch.Tensor,
        return_features: bool = True,
        return_logits: bool = False,
        return_attention_maps: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through image encoder."""
        # Extract features from backbone
        if self.model_name.startswith('timm_'):
            # For timm models, features are already pooled
            backbone_features = self.backbone(images)
            pooled_features = backbone_features
        else:
            # For torchvision models, need to handle feature extraction
            if hasattr(self.backbone, 'features'):
                # For models like VGG, DenseNet
                feature_maps = self.backbone.features(images)
            else:
                # For ResNet, etc., extract features before final pooling
                x = images
                for name, module in self.backbone.named_children():
                    if name in ['avgpool', 'classifier', 'fc', 'head']:
                        break
                    x = module(x)
                feature_maps = x
            
            # Apply global pooling
            if self.pooling_type == "attention":
                pooled_features = self._apply_attention_pooling(feature_maps)
                if return_attention_maps:
                    attention_weights = self.attention_pool(feature_maps)
            else:
                pooled_features = self.global_pool(feature_maps).flatten(1)
        
        # Project features
        projected_features = self.feature_projection(pooled_features)
        
        result = {}
        
        if return_features:
            result['features'] = projected_features
            result['raw_features'] = pooled_features
        
        if return_logits and self.classifier is not None:
            result['logits'] = self.classifier(projected_features)
        
        if return_attention_maps and self.pooling_type == "attention":
            result['attention_maps'] = attention_weights
        
        return result


class ViTImageEncoder(nn.Module):
    """Vision Transformer (ViT) image encoder."""
    
    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        hidden_size: int = 768,
        num_classes: Optional[int] = None,
        dropout_rate: float = 0.1,
        freeze_backbone: bool = False,
        fine_tune_layers: int = -1,
    ):
        super().__init__()
        
        self.model_name = model_name
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        
        # Load pre-trained ViT
        try:
            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=0  # Remove classification head
            )
            self.feature_dim = self.backbone.num_features
        except Exception as e:
            warnings.warn(f"Could not load {model_name}: {e}")
            # Fallback to base ViT
            self.backbone = timm.create_model(
                "vit_base_patch16_224",
                pretrained=pretrained,
                num_classes=0
            )
            self.feature_dim = self.backbone.num_features
        
        # Freeze backbone if requested
        if freeze_backbone:
            self._freeze_backbone()
        elif fine_tune_layers > 0:
            self._freeze_partial_backbone(fine_tune_layers)
        
        # Feature projection layer
        self.feature_projection = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Classification head if num_classes is provided
        if num_classes:
            self.classifier = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_size // 2, num_classes)
            )
        else:
            self.classifier = None
    
    def _freeze_backbone(self):
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def _freeze_partial_backbone(self, fine_tune_layers: int):
        """Freeze all but the last N transformer blocks."""
        # Freeze patch embedding and positional embedding
        for param in self.backbone.patch_embed.parameters():
            param.requires_grad = False
        for param in self.backbone.pos_embed:
            param.requires_grad = False
        if hasattr(self.backbone, 'cls_token'):
            self.backbone.cls_token.requires_grad = False
        
        # Freeze early transformer blocks
        if hasattr(self.backbone, 'blocks'):
            total_blocks = len(self.backbone.blocks)
            freeze_blocks = total_blocks - fine_tune_layers
            
            for i in range(freeze_blocks):
                for param in self.backbone.blocks[i].parameters():
                    param.requires_grad = False
    
    def forward(
        self, 
        images: torch.Tensor,
        return_features: bool = True,
        return_logits: bool = False,
        return_attention_maps: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through ViT encoder."""
        # Extract features
        if return_attention_maps:
            # Need to modify this to extract attention maps from ViT
            backbone_features = self.backbone(images)
            attention_maps = None  # Placeholder - would need custom forward hook
        else:
            backbone_features = self.backbone(images)
        
        # Project features
        projected_features = self.feature_projection(backbone_features)
        
        result = {}
        
        if return_features:
            result['features'] = projected_features
            result['raw_features'] = backbone_features
        
        if return_logits and self.classifier is not None:
            result['logits'] = self.classifier(projected_features)
        
        if return_attention_maps:
            result['attention_maps'] = attention_maps
        
        return result


class ImageEncoder(nn.Module):
    """Unified image encoder with multiple architecture options."""
    
    def __init__(
        self,
        encoder_type: str = "cnn",  # cnn, vit
        **kwargs
    ):
        super().__init__()
        
        self.encoder_type = encoder_type
        
        if encoder_type == "cnn":
            self.encoder = CNNImageEncoder(**kwargs)
        elif encoder_type == "vit":
            self.encoder = ViTImageEncoder(**kwargs)
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
    
    def forward(self, *args, **kwargs):
        """Forward pass through selected encoder."""
        return self.encoder(*args, **kwargs)


# Export classes
__all__ = [
    "ImageEncoder",
    "CNNImageEncoder",
    "ViTImageEncoder"
]
