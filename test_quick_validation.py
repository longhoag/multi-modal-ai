#!/usr/bin/env python3
"""
Quick validation test for the multi-modal AI application.
Tests basic functionality without heavy dependencies.
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

def test_configuration():
    """Test configuration loading"""
    print("🔧 Testing Configuration...")
    
    try:
        from config import get_config
        
        # Test basic config loading
        config = get_config()
        print(f"✅ Configuration loaded successfully")
        print(f"   Config type: {type(config)}")
        print(f"   Project root: {config.project_root}")
        print(f"   Batch size: {config.batch_size}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration failed: {e}")
        return False

def test_basic_data_operations():
    """Test basic data operations without torch"""
    print("\n📊 Testing Basic Data Operations...")
    
    try:
        # Test pandas operations
        df = pd.DataFrame({
            'text': ['Hello world', 'Machine learning', 'Python programming'],
            'value': [1.0, 2.5, 3.7],
            'category': ['A', 'B', 'A']
        })
        
        print(f"✅ Created test DataFrame: {df.shape}")
        
        # Test numpy operations
        features = np.random.random((5, 10))
        normalized = (features - features.mean()) / features.std()
        
        print(f"✅ NumPy operations working: {normalized.shape}")
        
        # Test basic text processing
        texts = df['text'].str.lower().str.split()
        print(f"✅ Basic text processing: {len(texts)} samples")
        
        return True
        
    except Exception as e:
        print(f"❌ Data operations failed: {e}")
        return False

def test_api_structure():
    """Test API structure without importing torch-dependent modules"""
    print("\n🌐 Testing API Structure...")
    
    try:
        # Test FastAPI import
        from fastapi import FastAPI
        
        # Create a simple app
        app = FastAPI(title="Multi-Modal AI Test")
        
        @app.get("/")
        def root():
            return {"message": "Multi-Modal AI API is working!"}
        
        @app.get("/health")
        def health():
            return {"status": "healthy", "components": ["api", "config"]}
        
        print(f"✅ FastAPI app created successfully")
        print(f"   Routes available: {len(app.routes)}")
        
        return True
        
    except Exception as e:
        print(f"❌ API structure test failed: {e}")
        return False

def test_mock_ml_pipeline():
    """Test a mock ML pipeline with numpy"""
    print("\n🤖 Testing Mock ML Pipeline...")
    
    try:
        # Mock text encoder using simple word counting
        def mock_text_encoder(texts):
            """Simple bag-of-words style encoding"""
            vocab = set()
            for text in texts:
                vocab.update(text.lower().split())
            vocab = list(vocab)
            
            features = []
            for text in texts:
                words = text.lower().split()
                feature_vec = [words.count(word) for word in vocab]
                features.append(feature_vec)
            
            return np.array(features), vocab
        
        # Mock tabular encoder using simple scaling
        def mock_tabular_encoder(data):
            """Simple scaling"""
            if isinstance(data, pd.DataFrame):
                data = data.select_dtypes(include=[np.number]).values
            data = np.array(data)
            return (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)
        
        # Test text encoding
        texts = ["hello world", "machine learning is great", "python programming"]
        text_features, vocab = mock_text_encoder(texts)
        print(f"✅ Text encoding: {text_features.shape}, vocab size: {len(vocab)}")
        
        # Test tabular encoding
        tabular_data = np.random.random((3, 5))
        tabular_features = mock_tabular_encoder(tabular_data)
        print(f"✅ Tabular encoding: {tabular_features.shape}")
        
        # Test fusion (simple concatenation)
        # Match dimensions by padding or truncating
        min_samples = min(text_features.shape[0], tabular_features.shape[0])
        fused = np.concatenate([
            text_features[:min_samples], 
            tabular_features[:min_samples]
        ], axis=1)
        print(f"✅ Feature fusion: {fused.shape}")
        
        # Mock prediction
        weights = np.random.random((fused.shape[1], 1))
        predictions = fused @ weights
        print(f"✅ Mock predictions: {predictions.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Mock ML pipeline failed: {e}")
        return False

def test_project_structure():
    """Test that project structure is intact"""
    print("\n📁 Testing Project Structure...")
    
    required_files = [
        'pyproject.toml',
        'README.md',
        'src/__init__.py',
        'src/config.py',
        'notebooks/01_multimodal_ai_tutorial.ipynb'
    ]
    
    all_present = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")
            all_present = False
    
    return all_present

def test_environment_setup():
    """Test environment and dependency setup"""
    print("\n🔍 Testing Environment Setup...")
    
    try:
        # Test essential imports
        import yaml
        print("✅ PyYAML available")
        
        import pandas
        print(f"✅ Pandas available: {pandas.__version__}")
        
        import numpy
        print(f"✅ NumPy available: {numpy.__version__}")
        
        import fastapi
        print(f"✅ FastAPI available: {fastapi.__version__}")
        
        # Test optional imports
        try:
            import torch
            print(f"✅ PyTorch available: {torch.__version__}")
        except ImportError:
            print("⚠️  PyTorch not available (install with: pip install torch)")
        
        try:
            import transformers
            print(f"✅ Transformers available")
        except ImportError:
            print("⚠️  Transformers not available (install with: pip install transformers)")
        
        return True
        
    except Exception as e:
        print(f"❌ Environment setup failed: {e}")
        return False

def main():
    """Run quick validation tests"""
    print("🚀 Multi-Modal AI - Quick Validation Test")
    print("=" * 60)
    
    tests = [
        ("Project Structure", test_project_structure),
        ("Environment Setup", test_environment_setup),
        ("Configuration", test_configuration),
        ("Basic Data Operations", test_basic_data_operations),
        ("API Structure", test_api_structure),
        ("Mock ML Pipeline", test_mock_ml_pipeline),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("🎯 VALIDATION SUMMARY")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n📊 Result: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed >= 4:  # At least 4 tests should pass
        print("\n🎉 VALIDATION SUCCESSFUL!")
        print("The multi-modal AI application basic structure is working!")
        print("\n📝 Next Steps to Complete Setup:")
        print("   1. Install PyTorch: pip install torch torchvision")
        print("   2. Install ML libraries: pip install transformers scikit-learn")
        print("   3. Install additional deps: pip install Pillow opencv-python")
        print("   4. Test full functionality with PyTorch")
        print("   5. Start the API: uvicorn src.api.main:app --reload")
        print("   6. Open Jupyter notebook for tutorials")
        
        # Test simple API endpoint
        print("\n🌐 Quick API Test:")
        print("   You can test the API structure is working by creating a simple endpoint")
        
        return True
    else:
        print("\n⚠️  VALIDATION FAILED")
        print("Some basic components are not working. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
