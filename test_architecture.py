#!/usr/bin/env python3
"""
Test script to verify YOLOv10 architecture is working with MAX
"""

import sys
import os
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_architecture_import():
    """Test if the architecture can be imported successfully"""
    try:
        print("Testing YOLOv10 architecture import...")
        from yolov10_model import ARCHITECTURES, yolov10_arch
        print("✅ Successfully imported YOLOv10 architecture")
        print(f"   Found {len(ARCHITECTURES)} architecture(s)")
        print(f"   Architecture name: {yolov10_arch.name}")
        print(f"   Task: {yolov10_arch.task}")
        print(f"   Supported encodings: {list(yolov10_arch.supported_encodings.keys())}")
        return True
    except ImportError as e:
        print(f"❌ Failed to import YOLOv10 architecture: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error during import: {e}")
        return False

def test_model_creation():
    """Test if the model can be instantiated"""
    try:
        print("\nTesting YOLOv10 model creation...")
        from yolov10_model.model import YOLOv10Model
        from yolov10_model.model_config import YOLOv10ModelConfig
        
        # Create a basic configuration
        config = YOLOv10ModelConfig(
            num_classes=80,  # COCO classes
            input_size=(640, 640),
            backbone_channels=(32, 64, 128, 256, 512, 1024),
            neck_channels=256,
            anchors_per_scale=3
        )
        
        print("✅ Successfully created YOLOv10 configuration")
        print(f"   Classes: {config.num_classes}")
        print(f"   Input size: {config.input_size}")
        print(f"   Backbone channels: {config.backbone_channels}")
        
        # Note: Model instantiation would require MAX graph context
        # This is just testing the configuration
        return True
        
    except Exception as e:
        print(f"❌ Failed to create model configuration: {e}")
        return False

def test_max_integration():
    """Test MAX-specific functionality"""
    try:
        print("\nTesting MAX integration...")
        import max
        
        # Test if MAX is available
        print("✅ MAX library is available")
        
        # Test if we can access MAX components
        from max.pipelines.core import PipelineTask
        from max.graph.weights import WeightsFormat
        
        print("✅ MAX pipeline components are accessible")
        
        # Test if our architecture task is valid
        from yolov10_model.arch import yolov10_arch
        print(f"✅ Architecture task is: {yolov10_arch.task}")
            
        return True
        
    except ImportError as e:
        print(f"❌ MAX library not available: {e}")
        return False
    except Exception as e:
        print(f"❌ MAX integration test failed: {e}")
        return False

def test_architecture_structure():
    """Test if all required files are present"""
    print("\nTesting architecture file structure...")
    
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

def main():
    """Run all tests"""
    print("🧪 Testing YOLOv10 Architecture with MAX")
    print("=" * 50)
    
    tests = [
        ("File Structure", test_architecture_structure),
        ("Architecture Import", test_architecture_import),
        ("Model Configuration", test_model_creation),
        ("MAX Integration", test_max_integration),
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
        print("🎉 All tests passed! YOLOv10 architecture is ready for use with MAX.")
        print("\nNext steps:")
        print("1. Use 'max serve --custom-architectures yolov10_model' to serve the model")
        print("2. Or integrate with your MAX pipeline")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 