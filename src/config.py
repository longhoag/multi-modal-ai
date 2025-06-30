"""Configuration management for the multi-modal AI application."""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Config(BaseSettings):
    """Main configuration class using Pydantic for validation."""
    
    # Project paths
    project_root: Path = Field(default_factory=lambda: Path(__file__).parent.parent)
    data_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "data")
    models_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "models")
    logs_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "logs")
    
    # Model configurations
    text_model_name: str = "bert-base-uncased"
    image_model_name: str = "resnet50"
    max_sequence_length: int = 512
    image_size: int = 224
    batch_size: int = 32
    learning_rate: float = 2e-5
    num_epochs: int = 10
    
    # Data configurations
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    random_seed: int = 42
    
    # API configurations
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = False
    
    # Database configurations
    database_url: str = "sqlite:///./multimodal_ai.db"
    redis_url: str = "redis://localhost:6379"
    
    # Logging configurations
    log_level: str = "INFO"
    
    # MLflow configurations
    mlflow_tracking_uri: str = "file:./mlruns"
    mlflow_experiment_name: str = "multimodal_ai"
    
    # Wandb configurations
    wandb_project: str = "multimodal-ai"
    wandb_entity: Optional[str] = None
    
    class Config:
        env_file = ".env"
        case_sensitive = False


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)


def get_config() -> Config:
    """Get the global configuration instance."""
    return Config()


# Global config instance
config = get_config()


class ModelConfig:
    """Model-specific configurations."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        self.config = config_dict
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "ModelConfig":
        """Load model config from YAML file."""
        config_dict = load_yaml_config(config_path)
        return cls(config_dict)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.config.get(key, default)


class DataConfig:
    """Data-specific configurations."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        self.config = config_dict
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "DataConfig":
        """Load data config from YAML file."""
        config_dict = load_yaml_config(config_path)
        return cls(config_dict)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.config.get(key, default)


class TrainingConfig:
    """Training-specific configurations."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        self.config = config_dict
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "TrainingConfig":
        """Load training config from YAML file."""
        config_dict = load_yaml_config(config_path)
        return cls(config_dict)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.config.get(key, default)
