"""Data preprocessing utilities for multi-modal data."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any
import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
import albumentations as A
from transformers import AutoTokenizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class TextPreprocessor:
    """Text preprocessing for multi-modal models."""
    
    def __init__(
        self,
        tokenizer_name: str = "bert-base-uncased",
        max_length: int = 512,
        clean_text: bool = True,
        remove_stopwords: bool = False
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.clean_text = clean_text
        self.remove_stopwords = remove_stopwords
        
        if remove_stopwords:
            try:
                self.stop_words = set(stopwords.words('english'))
            except LookupError:
                nltk.download('stopwords')
                self.stop_words = set(stopwords.words('english'))
    
    def clean_text_content(self, text: str) -> str:
        """Clean text content."""
        if not isinstance(text, str):
            return ""
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove mentions and hashtags (keep content)
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.,!?;:]', '', text)
        
        return text
    
    def remove_stop_words(self, text: str) -> str:
        """Remove stopwords from text."""
        try:
            word_tokens = word_tokenize(text.lower())
        except LookupError:
            nltk.download('punkt')
            word_tokens = word_tokenize(text.lower())
        
        filtered_text = [word for word in word_tokens if word not in self.stop_words]
        return ' '.join(filtered_text)
    
    def preprocess(self, texts: Union[str, List[str]]) -> Dict[str, torch.Tensor]:
        """Preprocess text data."""
        if isinstance(texts, str):
            texts = [texts]
        
        processed_texts = []
        for text in texts:
            if self.clean_text:
                text = self.clean_text_content(text)
            if self.remove_stopwords:
                text = self.remove_stop_words(text)
            processed_texts.append(text)
        
        # Tokenize
        encoded = self.tokenizer(
            processed_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return encoded


class ImagePreprocessor:
    """Image preprocessing for multi-modal models."""
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        normalize: bool = True,
        augment: bool = False,
        augmentation_prob: float = 0.5
    ):
        self.image_size = image_size
        self.normalize = normalize
        self.augment = augment
        
        # Basic transforms
        self.basic_transforms = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
        ])
        
        if normalize:
            # ImageNet normalization
            self.basic_transforms = transforms.Compose([
                self.basic_transforms,
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        
        # Augmentation pipeline
        if augment:
            self.augment_pipeline = A.Compose([
                A.RandomRotate90(p=augmentation_prob),
                A.Flip(p=augmentation_prob),
                A.OneOf([
                    A.RandomBrightnessContrast(p=0.5),
                    A.RandomGamma(p=0.5),
                    A.CLAHE(p=0.5),
                ], p=augmentation_prob),
                A.OneOf([
                    A.Blur(blur_limit=3, p=0.5),
                    A.GaussianBlur(blur_limit=3, p=0.5),
                    A.MedianBlur(blur_limit=3, p=0.5),
                ], p=augmentation_prob * 0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.1,
                    rotate_limit=15,
                    p=augmentation_prob
                ),
                A.CoarseDropout(
                    max_holes=8,
                    max_height=8,
                    max_width=8,
                    p=augmentation_prob * 0.5
                ),
            ])
    
    def load_image(self, image_path: str) -> np.ndarray:
        """Load image from path."""
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            # Assume it's already a numpy array or PIL Image
            if isinstance(image_path, Image.Image):
                image = np.array(image_path)
            else:
                image = image_path
        
        return image
    
    def preprocess(
        self, 
        images: Union[str, np.ndarray, List[Union[str, np.ndarray]]]
    ) -> torch.Tensor:
        """Preprocess image data."""
        if not isinstance(images, list):
            images = [images]
        
        processed_images = []
        
        for image in images:
            # Load image
            if isinstance(image, str):
                img_array = self.load_image(image)
            else:
                img_array = image
            
            # Apply augmentation if enabled
            if self.augment and hasattr(self, 'augment_pipeline'):
                augmented = self.augment_pipeline(image=img_array)
                img_array = augmented['image']
            
            # Convert to PIL for torchvision transforms
            if isinstance(img_array, np.ndarray):
                img_pil = Image.fromarray(img_array)
            else:
                img_pil = img_array
            
            # Apply transforms
            img_tensor = self.basic_transforms(img_pil)
            processed_images.append(img_tensor)
        
        return torch.stack(processed_images)


class TabularPreprocessor:
    """Tabular data preprocessing for multi-modal models."""
    
    def __init__(
        self,
        numerical_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
        target_column: Optional[str] = None,
        scale_numerical: bool = True,
        encode_categorical: bool = True
    ):
        self.numerical_features = numerical_features or []
        self.categorical_features = categorical_features or []
        self.target_column = target_column
        self.scale_numerical = scale_numerical
        self.encode_categorical = encode_categorical
        
        self.numerical_scaler = StandardScaler() if scale_numerical else None
        self.categorical_encoders = {}
        self.feature_names = []
        self.is_fitted = False
    
    def fit(self, data: pd.DataFrame) -> 'TabularPreprocessor':
        """Fit preprocessor on training data."""
        # Identify feature types if not provided
        if not self.numerical_features and not self.categorical_features:
            self.numerical_features = list(data.select_dtypes(
                include=[np.number]
            ).columns)
            self.categorical_features = list(data.select_dtypes(
                include=['object', 'category']
            ).columns)
            
            # Remove target column
            if self.target_column:
                if self.target_column in self.numerical_features:
                    self.numerical_features.remove(self.target_column)
                if self.target_column in self.categorical_features:
                    self.categorical_features.remove(self.target_column)
        
        # Fit numerical scaler
        if self.numerical_scaler and self.numerical_features:
            self.numerical_scaler.fit(data[self.numerical_features])
        
        # Fit categorical encoders
        if self.encode_categorical and self.categorical_features:
            for feature in self.categorical_features:
                encoder = LabelEncoder()
                encoder.fit(data[feature].astype(str))
                self.categorical_encoders[feature] = encoder
        
        # Store feature names
        self.feature_names = self.numerical_features + self.categorical_features
        self.is_fitted = True
        
        return self
    
    def transform(self, data: pd.DataFrame) -> torch.Tensor:
        """Transform data."""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transforming")
        
        processed_features = []
        
        # Process numerical features
        if self.numerical_features:
            if self.numerical_scaler:
                num_data = self.numerical_scaler.transform(
                    data[self.numerical_features]
                )
            else:
                num_data = data[self.numerical_features].values
            processed_features.append(num_data)
        
        # Process categorical features
        if self.categorical_features and self.encode_categorical:
            cat_data = []
            for feature in self.categorical_features:
                encoded = self.categorical_encoders[feature].transform(
                    data[feature].astype(str)
                )
                cat_data.append(encoded.reshape(-1, 1))
            
            if cat_data:
                cat_data = np.concatenate(cat_data, axis=1)
                processed_features.append(cat_data)
        
        # Combine all features
        if processed_features:
            combined_features = np.concatenate(processed_features, axis=1)
        else:
            combined_features = np.array([]).reshape(len(data), 0)
        
        return torch.FloatTensor(combined_features)
    
    def fit_transform(self, data: pd.DataFrame) -> torch.Tensor:
        """Fit and transform data."""
        return self.fit(data).transform(data)
    
    def get_feature_names(self) -> List[str]:
        """Get feature names after preprocessing."""
        return self.feature_names


class MultiModalPreprocessor:
    """Combined preprocessor for multi-modal data."""
    
    def __init__(
        self,
        text_config: Optional[Dict[str, Any]] = None,
        image_config: Optional[Dict[str, Any]] = None,
        tabular_config: Optional[Dict[str, Any]] = None
    ):
        # Initialize individual preprocessors
        self.text_preprocessor = None
        self.image_preprocessor = None
        self.tabular_preprocessor = None
        
        if text_config:
            self.text_preprocessor = TextPreprocessor(**text_config)
        
        if image_config:
            self.image_preprocessor = ImagePreprocessor(**image_config)
        
        if tabular_config:
            self.tabular_preprocessor = TabularPreprocessor(**tabular_config)
    
    def preprocess(
        self,
        data: Dict[str, Any],
        fit_tabular: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Preprocess multi-modal data."""
        processed_data = {}
        
        # Process text data
        if 'text' in data and self.text_preprocessor:
            processed_data['text'] = self.text_preprocessor.preprocess(data['text'])
        
        # Process image data
        if 'images' in data and self.image_preprocessor:
            processed_data['images'] = self.image_preprocessor.preprocess(data['images'])
        
        # Process tabular data
        if 'tabular' in data and self.tabular_preprocessor:
            if fit_tabular:
                processed_data['tabular'] = self.tabular_preprocessor.fit_transform(
                    data['tabular']
                )
            else:
                processed_data['tabular'] = self.tabular_preprocessor.transform(
                    data['tabular']
                )
        
        return processed_data
