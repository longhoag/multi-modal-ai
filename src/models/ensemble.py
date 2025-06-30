"""Ensemble methods for combining multiple models."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Any
import numpy as np


class VotingEnsemble(nn.Module):
    """Voting ensemble for combining multiple models."""
    
    def __init__(
        self,
        models: List[nn.Module],
        voting_method: str = "soft",  # soft, hard
        weights: Optional[List[float]] = None
    ):
        super().__init__()
        
        self.models = nn.ModuleList(models)
        self.voting_method = voting_method
        self.weights = weights
        
        if weights is not None:
            if len(weights) != len(models):
                raise ValueError("Number of weights must match number of models")
            # Normalize weights
            self.weights = [w / sum(weights) for w in weights]
    
    def forward(
        self,
        *args,
        return_individual_predictions: bool = False,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through ensemble."""
        individual_outputs = []
        
        # Get predictions from each model
        for model in self.models:
            with torch.no_grad():
                output = model(*args, **kwargs)
                individual_outputs.append(output)
        
        # Combine predictions
        if self.voting_method == "soft":
            # Soft voting: average probabilities
            if 'logits' in individual_outputs[0]:
                logits_list = [output['logits'] for output in individual_outputs]
                
                if self.weights:
                    weighted_logits = []
                    for logits, weight in zip(logits_list, self.weights):
                        weighted_logits.append(logits * weight)
                    ensemble_logits = torch.stack(weighted_logits).sum(dim=0)
                else:
                    ensemble_logits = torch.stack(logits_list).mean(dim=0)
                
                ensemble_output = {'logits': ensemble_logits}
            else:
                raise ValueError("Models must return logits for soft voting")
        
        elif self.voting_method == "hard":
            # Hard voting: majority vote
            if 'logits' in individual_outputs[0]:
                predictions_list = []
                for output in individual_outputs:
                    predictions = torch.argmax(output['logits'], dim=-1)
                    predictions_list.append(predictions)
                
                # Stack and find mode (most frequent prediction)
                stacked_predictions = torch.stack(predictions_list)  # [num_models, batch_size]
                ensemble_predictions = torch.mode(stacked_predictions, dim=0)[0]
                
                ensemble_output = {'predictions': ensemble_predictions}
            else:
                raise ValueError("Models must return logits for hard voting")
        
        else:
            raise ValueError(f"Unknown voting method: {self.voting_method}")
        
        if return_individual_predictions:
            ensemble_output['individual_outputs'] = individual_outputs
        
        return ensemble_output


class StackingEnsemble(nn.Module):
    """Stacking ensemble with meta-learner."""
    
    def __init__(
        self,
        base_models: List[nn.Module],
        meta_learner: nn.Module,
        use_original_features: bool = True
    ):
        super().__init__()
        
        self.base_models = nn.ModuleList(base_models)
        self.meta_learner = meta_learner
        self.use_original_features = use_original_features
    
    def forward(
        self,
        *args,
        return_base_predictions: bool = False,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through stacking ensemble."""
        base_predictions = []
        
        # Get predictions from base models
        for model in self.base_models:
            with torch.no_grad():
                output = model(*args, **kwargs)
                if 'logits' in output:
                    base_predictions.append(output['logits'])
                elif 'features' in output:
                    base_predictions.append(output['features'])
                else:
                    raise ValueError("Base models must return logits or features")
        
        # Concatenate base predictions
        stacked_predictions = torch.cat(base_predictions, dim=-1)
        
        # Optionally include original features
        if self.use_original_features and len(args) > 0:
            original_input = args[0]
            if isinstance(original_input, dict):
                # Multi-modal input
                meta_input = stacked_predictions
            else:
                # Single modal input - concatenate with predictions
                if original_input.dim() == 2:  # Tabular data
                    meta_input = torch.cat([original_input, stacked_predictions], dim=-1)
                else:
                    meta_input = stacked_predictions
        else:
            meta_input = stacked_predictions
        
        # Meta-learner prediction
        meta_output = self.meta_learner(meta_input)
        
        result = meta_output
        
        if return_base_predictions:
            result['base_predictions'] = base_predictions
        
        return result


class BaggingEnsemble(nn.Module):
    """Bagging ensemble with bootstrap sampling."""
    
    def __init__(
        self,
        model_constructor: callable,
        num_models: int = 5,
        model_kwargs: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        
        self.num_models = num_models
        model_kwargs = model_kwargs or {}
        
        # Create models
        self.models = nn.ModuleList([
            model_constructor(**model_kwargs) for _ in range(num_models)
        ])
    
    def forward(
        self,
        *args,
        return_individual_predictions: bool = False,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through bagging ensemble."""
        individual_outputs = []
        
        # Get predictions from each model
        for model in self.models:
            output = model(*args, **kwargs)
            individual_outputs.append(output)
        
        # Average predictions
        if 'logits' in individual_outputs[0]:
            logits_list = [output['logits'] for output in individual_outputs]
            ensemble_logits = torch.stack(logits_list).mean(dim=0)
            ensemble_output = {'logits': ensemble_logits}
        elif 'features' in individual_outputs[0]:
            features_list = [output['features'] for output in individual_outputs]
            ensemble_features = torch.stack(features_list).mean(dim=0)
            ensemble_output = {'features': ensemble_features}
        else:
            raise ValueError("Models must return logits or features")
        
        if return_individual_predictions:
            ensemble_output['individual_outputs'] = individual_outputs
        
        return ensemble_output


class AdaptiveWeightedEnsemble(nn.Module):
    """Ensemble with learned adaptive weights."""
    
    def __init__(
        self,
        models: List[nn.Module],
        input_dim: int,
        hidden_dim: int = 64,
        temperature: float = 1.0
    ):
        super().__init__()
        
        self.models = nn.ModuleList(models)
        self.num_models = len(models)
        self.temperature = temperature
        
        # Weight prediction network
        self.weight_predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, self.num_models)
        )
    
    def forward(
        self,
        input_features: torch.Tensor,
        *args,
        return_weights: bool = False,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through adaptive weighted ensemble."""
        # Predict weights based on input
        weight_logits = self.weight_predictor(input_features)
        weights = F.softmax(weight_logits / self.temperature, dim=-1)
        
        individual_outputs = []
        
        # Get predictions from each model
        for model in self.models:
            output = model(*args, **kwargs)
            individual_outputs.append(output)
        
        # Weighted combination
        if 'logits' in individual_outputs[0]:
            logits_list = [output['logits'] for output in individual_outputs]
            
            # Apply weights
            weighted_logits = []
            for i, logits in enumerate(logits_list):
                weight = weights[:, i:i+1]  # [batch_size, 1]
                weighted_logits.append(logits * weight)
            
            ensemble_logits = torch.stack(weighted_logits).sum(dim=0)
            ensemble_output = {'logits': ensemble_logits}
        
        elif 'features' in individual_outputs[0]:
            features_list = [output['features'] for output in individual_outputs]
            
            # Apply weights
            weighted_features = []
            for i, features in enumerate(features_list):
                weight = weights[:, i:i+1]  # [batch_size, 1]
                weighted_features.append(features * weight)
            
            ensemble_features = torch.stack(weighted_features).sum(dim=0)
            ensemble_output = {'features': ensemble_features}
        
        else:
            raise ValueError("Models must return logits or features")
        
        if return_weights:
            ensemble_output['weights'] = weights
        
        return ensemble_output


class MultiModalEnsemble(nn.Module):
    """Ensemble specifically designed for multi-modal models."""
    
    def __init__(
        self,
        text_models: Optional[List[nn.Module]] = None,
        image_models: Optional[List[nn.Module]] = None,
        tabular_models: Optional[List[nn.Module]] = None,
        fusion_models: Optional[List[nn.Module]] = None,
        ensemble_method: str = "weighted_average",
        fusion_weights: Optional[Dict[str, float]] = None
    ):
        super().__init__()
        
        self.text_models = nn.ModuleList(text_models or [])
        self.image_models = nn.ModuleList(image_models or [])
        self.tabular_models = nn.ModuleList(tabular_models or [])
        self.fusion_models = nn.ModuleList(fusion_models or [])
        
        self.ensemble_method = ensemble_method
        self.fusion_weights = fusion_weights or {}
    
    def forward(
        self,
        data: Dict[str, torch.Tensor],
        return_modality_predictions: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through multi-modal ensemble."""
        modality_predictions = {}
        
        # Process each modality separately
        if 'text' in data and self.text_models:
            text_outputs = []
            for model in self.text_models:
                output = model(data['text'])
                text_outputs.append(output)
            
            # Average text predictions
            if text_outputs:
                if 'logits' in text_outputs[0]:
                    text_logits = torch.stack([out['logits'] for out in text_outputs]).mean(dim=0)
                    modality_predictions['text'] = {'logits': text_logits}
        
        if 'images' in data and self.image_models:
            image_outputs = []
            for model in self.image_models:
                output = model(data['images'])
                image_outputs.append(output)
            
            # Average image predictions
            if image_outputs:
                if 'logits' in image_outputs[0]:
                    image_logits = torch.stack([out['logits'] for out in image_outputs]).mean(dim=0)
                    modality_predictions['images'] = {'logits': image_logits}
        
        if 'tabular' in data and self.tabular_models:
            tabular_outputs = []
            for model in self.tabular_models:
                output = model(data['tabular'])
                tabular_outputs.append(output)
            
            # Average tabular predictions
            if tabular_outputs:
                if 'logits' in tabular_outputs[0]:
                    tabular_logits = torch.stack([out['logits'] for out in tabular_outputs]).mean(dim=0)
                    modality_predictions['tabular'] = {'logits': tabular_logits}
        
        # Multi-modal fusion ensemble
        if self.fusion_models:
            fusion_outputs = []
            for model in self.fusion_models:
                output = model(data)
                fusion_outputs.append(output)
            
            # Average fusion predictions
            if fusion_outputs:
                if 'logits' in fusion_outputs[0]:
                    fusion_logits = torch.stack([out['logits'] for out in fusion_outputs]).mean(dim=0)
                    modality_predictions['fusion'] = {'logits': fusion_logits}
        
        # Combine modality predictions
        if self.ensemble_method == "weighted_average":
            final_logits = None
            total_weight = 0
            
            for modality, prediction in modality_predictions.items():
                if 'logits' in prediction:
                    weight = self.fusion_weights.get(modality, 1.0)
                    if final_logits is None:
                        final_logits = prediction['logits'] * weight
                    else:
                        final_logits += prediction['logits'] * weight
                    total_weight += weight
            
            if final_logits is not None:
                final_logits /= total_weight
        
        result = {'logits': final_logits}
        
        if return_modality_predictions:
            result['modality_predictions'] = modality_predictions
        
        return result


class EnsembleModel(nn.Module):
    """Unified ensemble model with multiple strategy options."""
    
    def __init__(
        self,
        ensemble_type: str = "voting",  # voting, stacking, bagging, adaptive, multimodal
        **kwargs
    ):
        super().__init__()
        
        self.ensemble_type = ensemble_type
        
        if ensemble_type == "voting":
            self.ensemble = VotingEnsemble(**kwargs)
        elif ensemble_type == "stacking":
            self.ensemble = StackingEnsemble(**kwargs)
        elif ensemble_type == "bagging":
            self.ensemble = BaggingEnsemble(**kwargs)
        elif ensemble_type == "adaptive":
            self.ensemble = AdaptiveWeightedEnsemble(**kwargs)
        elif ensemble_type == "multimodal":
            self.ensemble = MultiModalEnsemble(**kwargs)
        else:
            raise ValueError(f"Unknown ensemble type: {ensemble_type}")
    
    def forward(self, *args, **kwargs):
        """Forward pass through selected ensemble strategy."""
        return self.ensemble(*args, **kwargs)


# Export classes
__all__ = [
    "EnsembleModel",
    "VotingEnsemble",
    "StackingEnsemble",
    "BaggingEnsemble",
    "AdaptiveWeightedEnsemble",
    "MultiModalEnsemble"
]
