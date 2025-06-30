#!/bin/bash

# Multi-Modal AI Project Setup Script
echo "Setting up Multi-Modal AI Project..."

# Create conda environment
echo "Creating conda environment..."
conda create -n multimodal_ai python=3.11 -y
conda activate multimodal_ai

# Install PyTorch with CUDA support
echo "Installing PyTorch..."
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install system packages
echo "Installing system packages..."
conda install numpy pandas matplotlib seaborn jupyter scikit-learn -y

# Initialize Poetry if not already done
echo "Initializing Poetry..."
if [ ! -f "pyproject.toml" ]; then
    poetry init --no-interaction --name "multimodal-ai-app" --version "0.1.0"
fi

# Install ML dependencies
echo "Installing ML dependencies..."
poetry add transformers datasets evaluate accelerate
poetry add timm albumentations opencv-python pillow
poetry add pytorch-lightning torchmetrics
poetry add optuna wandb tensorboard mlflow
poetry add fastapi uvicorn pydantic streamlit gradio
poetry add requests beautifulsoup4 nltk spacy
poetry add shap lime captum

# Install development dependencies
echo "Installing development dependencies..."
poetry add --group dev jupyter ipython pytest black flake8 mypy
poetry add --group dev pre-commit pytest-cov pytest-asyncio

# Install all dependencies
poetry install

# Download language models
echo "Downloading language models..."
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

echo "Setup complete! Activate the environment with: conda activate multimodal_ai && poetry shell"
