"""Data loading utilities for multi-modal AI application."""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from typing import Dict, List, Optional, Tuple, Any
import json
from PIL import Image
import requests
from io import BytesIO

from ..config import config


class MultiModalDataset(Dataset):
    """Multi-modal dataset for text, image, and tabular data."""
    
    def __init__(
        self,
        data: pd.DataFrame,
        text_column: str = "text",
        image_column: str = "image_path",
        target_column: str = "label",
        text_transform=None,
        image_transform=None,
        tabular_columns: Optional[List[str]] = None,
    ):
        """
        Initialize the multi-modal dataset.
        
        Args:
            data: DataFrame containing the data
            text_column: Name of the text column
            image_column: Name of the image path column
            target_column: Name of the target label column
            text_transform: Text preprocessing transform
            image_transform: Image preprocessing transform
            tabular_columns: List of tabular feature columns
        """
        self.data = data
        self.text_column = text_column
        self.image_column = image_column
        self.target_column = target_column
        self.text_transform = text_transform
        self.image_transform = image_transform
        self.tabular_columns = tabular_columns or []
        
    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get an item from the dataset."""
        row = self.data.iloc[idx]
        
        # Get text data
        text = row[self.text_column] if pd.notna(row[self.text_column]) else ""
        if self.text_transform:
            text = self.text_transform(text)
        
        # Get image data
        image = None
        if self.image_column in row and pd.notna(row[self.image_column]):
            image_path = row[self.image_column]
            try:
                if image_path.startswith(('http://', 'https://')):
                    # Load image from URL
                    response = requests.get(image_path)
                    image = Image.open(BytesIO(response.content)).convert('RGB')
                else:
                    # Load image from local path
                    image = Image.open(image_path).convert('RGB')
                
                if self.image_transform:
                    image = self.image_transform(image)
            except (requests.RequestException, IOError, OSError) as e:
                print(f"Error loading image {image_path}: {e}")
                # Create a dummy image if loading fails
                image = Image.new('RGB', (224, 224), color='black')
                if self.image_transform:
                    image = self.image_transform(image)
        
        # Get tabular data
        tabular_features = []
        if self.tabular_columns:
            for col in self.tabular_columns:
                if col in row:
                    value = row[col]
                    if pd.isna(value):
                        value = 0.0
                    tabular_features.append(float(value))
                else:
                    tabular_features.append(0.0)
        
        # Get target
        target = row[self.target_column] if self.target_column in row else 0
        
        return {
            'text': text,
            'image': image,
            'tabular': torch.tensor(tabular_features, dtype=torch.float32),
            'target': torch.tensor(target, dtype=torch.long)
        }


class MultiModalDataLoader:
    """Data loader for multi-modal datasets."""
    
    def __init__(self):
        """Initialize the data loader."""
        self.config = config
        
    def load_csv(self, file_path: str) -> pd.DataFrame:
        """Load data from CSV file."""
        return pd.read_csv(file_path)
    
    def load_json(self, file_path: str) -> List[Dict]:
        """Load data from JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def load_jsonl(self, file_path: str) -> List[Dict]:
        """Load data from JSONL file."""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data
    
    def create_sample_data(self, num_samples: int = 1000) -> pd.DataFrame:
        """Create sample multi-modal data for testing."""
        np.random.seed(config.random_seed)
        
        # Sample text data
        sample_texts = [
            "This is a great product!",
            "I love this content, very informative.",
            "This is inappropriate and offensive.",
            "Spam message with links to suspicious sites.",
            "Normal everyday conversation about weather.",
            "Educational content about science and technology.",
            "Hate speech targeting specific groups.",
            "Advertisement for legitimate business services.",
            "Political discussion with respectful tone.",
            "Cyberbullying and harassment message.",
        ]
        
        # Generate random data
        data = {
            'text': np.random.choice(sample_texts, num_samples),
            'user_followers': np.random.randint(0, 100000, num_samples),
            'engagement_rate': np.random.uniform(0, 1, num_samples),
            'account_age_days': np.random.randint(0, 3650, num_samples),
            'is_verified': np.random.choice([0, 1], num_samples),
            'post_length': np.random.randint(10, 280, num_samples),
            'has_image': np.random.choice([0, 1], num_samples, p=[0.3, 0.7]),
            'time_of_day': np.random.randint(0, 24, num_samples),
            'day_of_week': np.random.randint(0, 7, num_samples),
        }
        
        # Generate labels (0: safe, 1: unsafe)
        # Create some correlation with text content
        labels = []
        for text in data['text']:
            if any(word in text.lower() for word in ['inappropriate', 'offensive', 'hate', 'spam', 'cyberbullying', 'harassment']):
                labels.append(1)  # Unsafe
            else:
                labels.append(0)  # Safe
        
        data['label'] = labels
        
        # Add some image paths (dummy)
        data['image_path'] = [f"data/images/sample_{i}.jpg" if has_img else None 
                             for i, has_img in enumerate(data['has_image'])]
        
        return pd.DataFrame(data)
    
    def split_data(
        self, 
        data: pd.DataFrame, 
        train_ratio: float = 0.8, 
        val_ratio: float = 0.1, 
        test_ratio: float = 0.1,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train, validation, and test sets."""
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
        
        # Shuffle data
        data = data.sample(frac=1, random_state=random_state).reset_index(drop=True)
        
        n_total = len(data)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_data = data[:n_train]
        val_data = data[n_train:n_train + n_val]
        test_data = data[n_train + n_val:]
        
        return train_data, val_data, test_data
    
    def get_dataloader(
        self,
        dataset: MultiModalDataset,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
        **kwargs
    ) -> TorchDataLoader:
        """Create a PyTorch DataLoader."""
        return TorchDataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            **kwargs
        )
