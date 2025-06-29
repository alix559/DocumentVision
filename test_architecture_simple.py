#!/usr/bin/env python3
"""
Simple test script to verify YOLOv10 architecture registration with MAX
"""

import sys
import os
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_architecture_structure():
    """Test if all required files are present"""
    print("Testing architecture file structure...")
    
    required_files = [
        "yolov10_model/__init__.py",
        "yolov10_model/arch.py", 
        "yolov10_model/model.py",
        "yolov10_model/model_config.py",
        "yolov10_model/weight_adapters.py"
    ]
    
    all_present = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - MISSING")
            all_present = False
    
    return all_present

def test_basic_imports():
    """Test basic imports without complex model implementation"""
    try:
        print("\nTesting basic imports...")
        
        # Test MAX imports
        import max
        print("✅ MAX library imported successfully")
        
        from max.pipelines.core import PipelineTask
        print("✅ PipelineTask imported successfully")
        
        from max.pipelines.lib import SupportedArchitecture, SupportedEncoding
        print("✅ SupportedArchitecture imported successfully")
        
        from max.graph.weights import WeightsFormat
        print("✅ WeightsFormat imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_architecture_registration():
    """Test if the architecture can be registered"""
    try:
        print("\nTesting architecture registration...")
        
        # Import the architecture
        from yolov10_model.arch import yolov10_arch
        
        print("✅ Architecture imported successfully")
        print(f"   Name: {yolov10_arch.name}")
        print(f"   Task: {yolov10_arch.task}")
        print(f"   Default encoding: {yolov10_arch.default_encoding}")
        print(f"   Supported encodings: {list(yolov10_arch.supported_encodings.keys())}")
        print(f"   Multi-GPU supported: {yolov10_arch.multi_gpu_supported}")
        
        # Test that it's a valid SupportedArchitecture
        from max.pipelines.lib import SupportedArchitecture
        if isinstance(yolov10_arch, SupportedArchitecture):
            print("✅ Architecture is valid SupportedArchitecture instance")
        else:
            print("❌ Architecture is not a valid SupportedArchitecture instance")
            return False
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import architecture: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error during registration test: {e}")
        return False

def test_config_import():
    """Test configuration import"""
    try:
        print("\nTesting configuration import...")
        
        from yolov10_model.model_config import YOLOv10ModelConfig
        
        # Create a basic configuration
        config = YOLOv10ModelConfig(
            num_classes=80,
            input_size=(640, 640)
        )
        
        print("✅ Configuration created successfully")
        print(f"   Classes: {config.num_classes}")
        print(f"   Input size: {config.input_size}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import configuration: {e}")
        return False
    except Exception as e:
        print(f"❌ Failed to create configuration: {e}")
        return False

def test_weight_adapters():
    """Test weight adapters import"""
    try:
        print("\nTesting weight adapters...")
        
        from yolov10_model import weight_adapters
        
        # Check if the main functions exist
        if hasattr(weight_adapters, 'convert_safetensor_state_dict'):
            print("✅ SafeTensors converter available")
        else:
            print("❌ SafeTensors converter missing")
            return False
            
        if hasattr(weight_adapters, 'convert_gguf_state_dict'):
            print("✅ GGUF converter available")
        else:
            print("❌ GGUF converter missing")
            return False
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import weight adapters: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error with weight adapters: {e}")
        return False

def test_max_serve_command():
    """Test if MAX serve command is available"""
    try:
        print("\nTesting MAX serve availability...")
        
        import subprocess
        result = subprocess.run(['max', '--help'], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ MAX CLI is available")
            print("   MAX command found in PATH")
            return True
        else:
            print("❌ MAX CLI not available")
            return False
            
    except FileNotFoundError:
        print("❌ MAX CLI not found in PATH")
        return False
    except Exception as e:
        print(f"❌ Error checking MAX CLI: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Simple YOLOv10 Architecture Test")
    print("=" * 50)
    
    tests = [
        ("File Structure", test_architecture_structure),
        ("Basic Imports", test_basic_imports),
        ("Configuration", test_config_import),
        ("Weight Adapters", test_weight_adapters),
        ("Architecture Registration", test_architecture_registration),
        ("MAX CLI", test_max_serve_command),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name} test...")
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! YOLOv10 architecture is ready for basic use.")
        print("\nNext steps:")
        print("1. Use 'max serve --custom-architectures yolov10_model' to serve the model")
        print("2. Note: Model implementation may need API updates for full functionality")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 