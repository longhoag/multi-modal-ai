#!/usr/bin/env python3
"""
Basic structure test for the multi-modal AI application.
Tests the package imports and basic functionality without heavy dependencies.
"""

import sys
import os
import importlib.util

def test_package_structure():
    """Test that all package modules can be imported"""
    print("🔍 Testing Package Structure...")
    
    # Add src to path
    src_path = os.path.join(os.path.dirname(__file__), 'src')
    sys.path.insert(0, src_path)
    
    try:
        # Test main package import
        import src
        print(f"✅ Main package imported successfully")
        print(f"   Version: {getattr(src, '__version__', 'unknown')}")
        print(f"   Author: {getattr(src, '__author__', 'unknown')}")
        
        # Test submodule imports
        modules_to_test = [
            'src.config',
            'src.api',
            'src.api.main',
            'src.api.schemas', 
            'src.data',
            'src.data.loaders',
            'src.data.preprocessors',
            'src.data.augmentations',
            'src.models',
            'src.models.text_encoder',
            'src.models.image_encoder', 
            'src.models.tabular_encoder',
            'src.models.fusion',
            'src.models.ensemble',
            'src.training'
        ]
        
        successful_imports = []
        failed_imports = []
        
        for module_name in modules_to_test:
            try:
                module = importlib.import_module(module_name)
                successful_imports.append(module_name)
                print(f"✅ {module_name}")
                
                # Check if module has expected attributes/classes
                if hasattr(module, '__all__'):
                    print(f"   Exports: {module.__all__}")
                    
            except ImportError as e:
                failed_imports.append((module_name, str(e)))
                print(f"❌ {module_name}: {e}")
        
        print(f"\n📊 Import Summary:")
        print(f"   Successful: {len(successful_imports)}/{len(modules_to_test)}")
        print(f"   Failed: {len(failed_imports)}")
        
        return len(failed_imports) == 0
        
    except Exception as e:
        print(f"❌ Failed to import main package: {e}")
        return False

def test_config_loading():
    """Test configuration loading"""
    print("\n🔧 Testing Configuration...")
    
    try:
        from src.config import load_config, save_config
        
        # Test default config
        config = load_config()
        print(f"✅ Configuration loaded successfully")
        print(f"   Keys: {list(config.keys()) if isinstance(config, dict) else 'Not a dict'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_api_structure():
    """Test API structure"""
    print("\n🌐 Testing API Structure...")
    
    try:
        from src.api.main import app
        from src.api.schemas import PredictionRequest, PredictionResponse
        
        print("✅ API components imported successfully")
        print(f"   FastAPI app: {type(app).__name__}")
        
        # Test schema structure
        print("✅ API schemas available")
        
        return True
        
    except Exception as e:
        print(f"❌ API structure test failed: {e}")
        return False

def test_model_structure():
    """Test model structure"""
    print("\n🤖 Testing Model Structure...")
    
    try:
        from src.models.text_encoder import TextEncoder
        from src.models.image_encoder import ImageEncoder
        from src.models.tabular_encoder import TabularEncoder
        from src.models.fusion import ModalityFusion
        from src.models.ensemble import EnsembleModel
        
        print("✅ Model classes imported successfully")
        
        # Test model instantiation (without dependencies)
        try:
            text_encoder = TextEncoder()
            print("✅ TextEncoder can be instantiated")
        except Exception as e:
            print(f"⚠️  TextEncoder instantiation failed: {e}")
        
        try:
            image_encoder = ImageEncoder()
            print("✅ ImageEncoder can be instantiated")
        except Exception as e:
            print(f"⚠️  ImageEncoder instantiation failed: {e}")
            
        try:
            tabular_encoder = TabularEncoder()
            print("✅ TabularEncoder can be instantiated")
        except Exception as e:
            print(f"⚠️  TabularEncoder instantiation failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model structure test failed: {e}")
        return False

def test_data_structure():
    """Test data handling structure"""
    print("\n📊 Testing Data Structure...")
    
    try:
        from src.data.loaders import DataLoader
        from src.data.preprocessors import TextPreprocessor, ImagePreprocessor, TabularPreprocessor
        from src.data.augmentations import DataAugmentation
        
        print("✅ Data handling classes imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Data structure test failed: {e}")
        return False

def test_file_structure():
    """Test file system structure"""
    print("\n📁 Testing File Structure...")
    
    required_files = [
        'pyproject.toml',
        'README.md',
        'src/__init__.py',
        'src/config.py',
        'src/api/__init__.py',
        'src/api/main.py',
        'src/api/schemas.py',
        'src/data/__init__.py',
        'src/data/loaders.py',
        'src/data/preprocessors.py',
        'src/data/augmentations.py',
        'src/models/__init__.py',
        'src/models/text_encoder.py',
        'src/models/image_encoder.py',
        'src/models/tabular_encoder.py',
        'src/models/fusion.py',
        'src/models/ensemble.py',
        'src/training/__init__.py',
        'notebooks/01_multimodal_ai_tutorial.ipynb'
    ]
    
    missing_files = []
    present_files = []
    
    for file_path in required_files:
        if os.path.exists(file_path):
            present_files.append(file_path)
            print(f"✅ {file_path}")
        else:
            missing_files.append(file_path)
            print(f"❌ {file_path}")
    
    print(f"\n📊 File Structure Summary:")
    print(f"   Present: {len(present_files)}/{len(required_files)}")
    print(f"   Missing: {len(missing_files)}")
    
    return len(missing_files) == 0

def main():
    """Run all tests"""
    print("🚀 Multi-Modal AI Application - Basic Structure Test")
    print("=" * 60)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Package Structure", test_package_structure),
        ("Configuration", test_config_loading),
        ("API Structure", test_api_structure),
        ("Model Structure", test_model_structure),
        ("Data Structure", test_data_structure),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Final summary
    print("\n" + "=" * 60)
    print("🎯 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n📊 Overall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The basic structure is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Check the details above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
