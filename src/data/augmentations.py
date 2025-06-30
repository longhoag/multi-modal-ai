"""Data augmentation utilities for multi-modal training."""

import random
import numpy as np
from typing import Dict, List, Optional, Union, Any
import torch
import torchvision.transforms as transforms
from PIL import Image
import albumentations as A
import cv2


class TextAugmentation:
    """Text augmentation techniques."""
    
    def __init__(self, augmentation_prob: float = 0.3):
        self.augmentation_prob = augmentation_prob
    
    def synonym_replacement(self, text: str, n: int = 1) -> str:
        """Replace n words with synonyms (placeholder implementation)."""
        # This would require NLTK WordNet or similar
        # For now, return original text
        return text
    
    def random_insertion(self, text: str, n: int = 1) -> str:
        """Insert n random words (placeholder implementation)."""
        # This would require a vocabulary
        # For now, return original text
        return text
    
    def random_swap(self, text: str, n: int = 1) -> str:
        """Randomly swap n pairs of words."""
        words = text.split()
        if len(words) < 2:
            return text
        
        for _ in range(n):
            if random.random() < self.augmentation_prob:
                idx1, idx2 = random.sample(range(len(words)), 2)
                words[idx1], words[idx2] = words[idx2], words[idx1]
        
        return ' '.join(words)
    
    def random_deletion(self, text: str, p: float = 0.1) -> str:
        """Randomly delete words with probability p."""
        words = text.split()
        if len(words) == 1:
            return text
        
        new_words = []
        for word in words:
            if random.random() > p:
                new_words.append(word)
        
        # Ensure at least one word remains
        if len(new_words) == 0:
            new_words = [random.choice(words)]
        
        return ' '.join(new_words)
    
    def augment(self, text: str) -> str:
        """Apply random text augmentation."""
        if random.random() < self.augmentation_prob:
            augmentation_methods = [
                self.random_swap,
                self.random_deletion,
            ]
            method = random.choice(augmentation_methods)
            return method(text)
        return text


class ImageAugmentation:
    """Advanced image augmentation using Albumentations."""
    
    def __init__(
        self,
        image_size: tuple = (224, 224),
        augmentation_prob: float = 0.5,
        heavy_augmentation: bool = False
    ):
        self.image_size = image_size
        self.augmentation_prob = augmentation_prob
        
        if heavy_augmentation:
            self.transform = self._create_heavy_augmentation()
        else:
            self.transform = self._create_light_augmentation()
    
    def _create_light_augmentation(self) -> A.Compose:
        """Create light augmentation pipeline."""
        return A.Compose([
            A.Resize(height=self.image_size[0], width=self.image_size[1]),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=self.augmentation_prob
            ),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=15,
                p=self.augmentation_prob
            ),
            A.GaussNoise(var_limit=(10.0, 50.0), p=self.augmentation_prob * 0.5),
        ])
    
    def _create_heavy_augmentation(self) -> A.Compose:
        """Create heavy augmentation pipeline."""
        return A.Compose([
            A.Resize(height=self.image_size[0], width=self.image_size[1]),
            A.OneOf([
                A.RandomRotate90(),
                A.Rotate(limit=30),
            ], p=0.5),
            A.OneOf([
                A.HorizontalFlip(p=1),
                A.VerticalFlip(p=1),
            ], p=0.5),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
                A.RandomGamma(gamma_limit=(80, 120)),
                A.CLAHE(),
                A.HueSaturationValue(),
            ], p=0.7),
            A.OneOf([
                A.Blur(blur_limit=3),
                A.GaussianBlur(blur_limit=3),
                A.MedianBlur(blur_limit=3),
                A.MotionBlur(blur_limit=3),
            ], p=0.4),
            A.ShiftScaleRotate(
                shift_limit=0.2,
                scale_limit=0.2,
                rotate_limit=30,
                p=0.7
            ),
            A.OneOf([
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(p=0.1),
                A.ElasticTransform(p=0.3),
            ], p=0.3),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0)),
                A.ISONoise(),
                A.MultiplicativeNoise(),
            ], p=0.3),
            A.OneOf([
                A.Cutout(num_holes=8, max_h_size=8, max_w_size=8, p=0.5),
                A.CoarseDropout(
                    max_holes=8,
                    max_height=8,
                    max_width=8,
                    p=0.5
                ),
            ], p=0.3),
        ])
    
    def augment(self, image: Union[np.ndarray, str]) -> np.ndarray:
        """Apply augmentation to image."""
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        augmented = self.transform(image=image)
        return augmented['image']


class MixupAugmentation:
    """Mixup augmentation for multi-modal data."""
    
    def __init__(self, alpha: float = 0.2, prob: float = 0.5):
        self.alpha = alpha
        self.prob = prob
    
    def mixup_data(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        alpha: Optional[float] = None
    ) -> tuple:
        """Mixup data augmentation."""
        if alpha is None:
            alpha = self.alpha
        
        if alpha > 0 and random.random() < self.prob:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam
    
    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        """Mixup loss computation."""
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class CutMixAugmentation:
    """CutMix augmentation for images."""
    
    def __init__(self, alpha: float = 1.0, prob: float = 0.5):
        self.alpha = alpha
        self.prob = prob
    
    def cutmix_data(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        alpha: Optional[float] = None
    ) -> tuple:
        """CutMix data augmentation."""
        if alpha is None:
            alpha = self.alpha
        
        if alpha > 0 and random.random() < self.prob:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size)
        
        bbx1, bby1, bbx2, bby2 = self._rand_bbox(x.size(), lam)
        x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
        
        # Adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
        
        y_a, y_b = y, y[index]
        return x, y_a, y_b, lam
    
    def _rand_bbox(self, size, lam):
        """Generate random bounding box."""
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int32(W * cut_rat)
        cut_h = np.int32(H * cut_rat)
        
        # Uniform sampling
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        return bbx1, bby1, bbx2, bby2


class MultiModalAugmentation:
    """Combined augmentation for multi-modal data."""
    
    def __init__(
        self,
        text_config: Optional[Dict[str, Any]] = None,
        image_config: Optional[Dict[str, Any]] = None,
        use_mixup: bool = False,
        use_cutmix: bool = False,
        mixup_config: Optional[Dict[str, Any]] = None,
        cutmix_config: Optional[Dict[str, Any]] = None
    ):
        # Initialize individual augmentations
        self.text_augmentation = None
        self.image_augmentation = None
        self.mixup_augmentation = None
        self.cutmix_augmentation = None
        
        if text_config:
            self.text_augmentation = TextAugmentation(**text_config)
        
        if image_config:
            self.image_augmentation = ImageAugmentation(**image_config)
        
        if use_mixup:
            mixup_config = mixup_config or {}
            self.mixup_augmentation = MixupAugmentation(**mixup_config)
        
        if use_cutmix:
            cutmix_config = cutmix_config or {}
            self.cutmix_augmentation = CutMixAugmentation(**cutmix_config)
    
    def augment(
        self,
        data: Dict[str, Any],
        targets: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """Apply multi-modal augmentation."""
        augmented_data = {}
        
        # Text augmentation
        if 'text' in data and self.text_augmentation:
            if isinstance(data['text'], list):
                augmented_data['text'] = [
                    self.text_augmentation.augment(text) for text in data['text']
                ]
            else:
                augmented_data['text'] = self.text_augmentation.augment(data['text'])
        else:
            augmented_data['text'] = data.get('text')
        
        # Image augmentation
        if 'images' in data and self.image_augmentation:
            if isinstance(data['images'], list):
                augmented_data['images'] = [
                    self.image_augmentation.augment(img) for img in data['images']
                ]
            else:
                augmented_data['images'] = self.image_augmentation.augment(data['images'])
        else:
            augmented_data['images'] = data.get('images')
        
        # Copy other data
        for key, value in data.items():
            if key not in ['text', 'images']:
                augmented_data[key] = value
        
        # Apply mixup/cutmix if targets are provided
        if targets is not None:
            if self.mixup_augmentation and 'images' in augmented_data:
                # Apply mixup to images
                mixed_images, y_a, y_b, lam = self.mixup_augmentation.mixup_data(
                    augmented_data['images'], targets
                )
                augmented_data['images'] = mixed_images
                augmented_data['targets'] = {'y_a': y_a, 'y_b': y_b, 'lam': lam}
            
            elif self.cutmix_augmentation and 'images' in augmented_data:
                # Apply cutmix to images
                mixed_images, y_a, y_b, lam = self.cutmix_augmentation.cutmix_data(
                    augmented_data['images'], targets
                )
                augmented_data['images'] = mixed_images
                augmented_data['targets'] = {'y_a': y_a, 'y_b': y_b, 'lam': lam}
        
        return augmented_data
