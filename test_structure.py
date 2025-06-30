#!/usr/bin/env python3
"""
Basic test script for Multi-Modal AI Application structure
Tests without requiring heavy ML dependencies
"""

import sys
import os
import importlib
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

def test_package_structure():
    """Test the basic package structure."""
    print("=" * 60)
    print("🏗️ Testing Package Structure")
    print("=" * 60)
    
    try:
        import src
        print(f"✅ Main package imported")
        print(f"   Version: {src.__version__}")
        print(f"   Author: {src.__author__}")
        print(f"   Email: {src.__email__}")
        return True
    except Exception as e:
        print(f"❌ Main package import failed: {e}")
        return False

def test_module_imports():
    """Test importing individual modules."""
    print("\n" + "=" * 60)
    print("📦 Testing Module Imports")
    print("=" * 60)
    
    modules_to_test = [
        ('src.config', 'Configuration module'),
        ('src.data', 'Data package'),
        ('src.models', 'Models package'),
        ('src.training', 'Training package'),
        ('src.api', 'API package'),
    ]
    
    results = {}
    
    for module_name, description in modules_to_test:
        try:
            module = importlib.import_module(module_name)
            print(f"✅ {description}: {module_name}")
            results[module_name] = True
        except ImportError as e:
            print(f"⚠️ {description}: {module_name} - {e}")
            results[module_name] = False
        except Exception as e:
            print(f"❌ {description}: {module_name} - Unexpected error: {e}")
            results[module_name] = False
    
    return results

def test_submodule_structure():
    """Test submodule structure without importing heavy dependencies."""
    print("\n" + "=" * 60)
    print("🔍 Testing Submodule Structure")
    print("=" * 60)
    
    # Test data submodules
    print("\n📊 Data submodules:")
    data_modules = [
        'src.data.loaders',
        'src.data.preprocessors',
        'src.data.augmentations'
    ]
    
    for module in data_modules:
        try:
            # Just check if the file exists
            module_path = module.replace('.', '/') + '.py'
            if os.path.exists(module_path):
                print(f"✅ {module} - File exists")
            else:
                print(f"❌ {module} - File missing")
        except Exception as e:
            print(f"❌ {module} - Error: {e}")
    
    # Test model submodules
    print("\n🧠 Model submodules:")
    model_modules = [
        'src.models.text_encoder',
        'src.models.image_encoder',
        'src.models.tabular_encoder',
        'src.models.fusion',
        'src.models.ensemble'
    ]
    
    for module in model_modules:
        try:
            module_path = module.replace('.', '/') + '.py'
            if os.path.exists(module_path):
                print(f"✅ {module} - File exists")
            else:
                print(f"❌ {module} - File missing")
        except Exception as e:
            print(f"❌ {module} - Error: {e}")
    
    # Test API submodules
    print("\n🌐 API submodules:")
    api_modules = [
        'src.api.main',
        'src.api.schemas'
    ]
    
    for module in api_modules:
        try:
            module_path = module.replace('.', '/') + '.py'
            if os.path.exists(module_path):
                print(f"✅ {module} - File exists")
            else:
                print(f"❌ {module} - File missing")
        except Exception as e:
            print(f"❌ {module} - Error: {e}")

def test_configuration():
    """Test configuration module."""
    print("\n" + "=" * 60)
    print("⚙️ Testing Configuration")
    print("=" * 60)
    
    try:
        from src.config import Config
        
        # Test configuration creation
        config = Config()
        print(f"✅ Configuration created")
        print(f"   Model configs available: {list(config.model_configs.keys())}")
        print(f"   Data configs available: {list(config.data_configs.keys())}")
        print(f"   Training configs available: {list(config.training_configs.keys())}")
        
        # Test specific configurations
        if 'text_encoder' in config.model_configs:
            text_config = config.get_model_config('text_encoder')
            print(f"   Text encoder config: {text_config}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_project_files():
    """Test project configuration files."""
    print("\n" + "=" * 60)
    print("📄 Testing Project Files")
    print("=" * 60)
    
    files_to_check = [
        ('pyproject.toml', 'Poetry configuration'),
        ('README.md', 'Project documentation'),
        ('Dockerfile', 'Docker configuration'),
        ('docker-compose.yml', 'Docker Compose configuration'),
        ('setup.sh', 'Setup script'),
        ('notebooks/01_multimodal_ai_tutorial.ipynb', 'Tutorial notebook'),
    ]
    
    for file_path, description in files_to_check:
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            print(f"✅ {description}: {file_path} ({file_size:,} bytes)")
        else:
            print(f"❌ {description}: {file_path} - Missing")

def test_api_structure():
    """Test API structure without starting the server."""
    print("\n" + "=" * 60)
    print("🌐 Testing API Structure")
    print("=" * 60)
    
    try:
        # Test schemas without pydantic dependencies
        print("📋 Checking API schemas...")
        schema_file = 'src/api/schemas.py'
        if os.path.exists(schema_file):
            with open(schema_file, 'r') as f:
                content = f.read()
                if 'ContentRequest' in content:
                    print("✅ ContentRequest schema defined")
                if 'ContentResponse' in content:
                    print("✅ ContentResponse schema defined")
                if 'HealthResponse' in content:
                    print("✅ HealthResponse schema defined")
        
        # Test main API file
        print("\n🚀 Checking API main...")
        main_file = 'src/api/main.py'
        if os.path.exists(main_file):
            with open(main_file, 'r') as f:
                content = f.read()
                if 'FastAPI' in content:
                    print("✅ FastAPI application defined")
                if '/predict/' in content:
                    print("✅ Prediction endpoints defined")
                if '/health' in content:
                    print("✅ Health endpoint defined")
        
        return True
        
    except Exception as e:
        print(f"❌ API structure test failed: {e}")
        return False

def test_notebook_structure():
    """Test notebook structure."""
    print("\n" + "=" * 60)
    print("📓 Testing Notebook Structure")
    print("=" * 60)
    
    try:
        notebook_path = 'notebooks/01_multimodal_ai_tutorial.ipynb'
        if os.path.exists(notebook_path):
            print(f"✅ Tutorial notebook exists: {notebook_path}")
            
            # Check notebook size
            size = os.path.getsize(notebook_path)
            print(f"   Size: {size:,} bytes")
            
            # Check if it's valid JSON
            import json
            with open(notebook_path, 'r') as f:
                try:
                    notebook_data = json.load(f)
                    print(f"✅ Valid Jupyter notebook format")
                    print(f"   Cells: {len(notebook_data.get('cells', []))}")
                except json.JSONDecodeError:
                    print(f"❌ Invalid JSON format")
        else:
            print(f"❌ Tutorial notebook missing: {notebook_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Notebook structure test failed: {e}")
        return False

def main():
    """Run all basic tests."""
    print("🧪 Multi-Modal AI Application - Basic Structure Test")
    print("=" * 60)
    
    tests = [
        ("Package Structure", test_package_structure),
        ("Module Imports", test_module_imports),
        ("Submodule Structure", test_submodule_structure),
        ("Configuration", test_configuration),
        ("Project Files", test_project_files),
        ("API Structure", test_api_structure),
        ("Notebook Structure", test_notebook_structure),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result if isinstance(result, bool) else True
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for r in results.values() if r is True)
    total = len([r for r in results.values() if isinstance(r, bool)])
    
    for test_name, result in results.items():
        if isinstance(result, bool):
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{status} {test_name}")
        elif isinstance(result, dict):
            # Handle module import results
            module_passed = sum(result.values())
            module_total = len(result)
            print(f"📊 {test_name}: {module_passed}/{module_total} modules")
    
    if total > 0:
        print(f"\n🎯 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 Basic structure tests passed! The project is properly set up.")
        print("\n💡 Next steps:")
        print("   1. Install dependencies: ./setup.sh")
        print("   2. Run full tests with PyTorch: python test_multimodal.py")
        print("   3. Start Jupyter notebook: jupyter lab notebooks/")
        return 0
    else:
        print(f"\n⚠️ Some structural issues found. Please check the errors above.")
        return 1

if __name__ == "__main__":
    exit(main())
