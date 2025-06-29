#!/usr/bin/env python3
"""
Test script for YOLOv10 deployment
"""

import requests
import base64
import json
import time

def test_health():
    """Test health endpoint"""
    try:
        response = requests.get("http://localhost:8000/health")
        print(f"✅ Health check: {response.status_code}")
        print(f"Response: {response.json()}")
        return True
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_model_info():
    """Test model info endpoint"""
    try:
        response = requests.get("http://localhost:8000/model/info")
        print(f"✅ Model info: {response.status_code}")
        print(f"Response: {response.json()}")
        return True
    except Exception as e:
        print(f"❌ Model info failed: {e}")
        return False

def test_object_detection():
    """Test object detection endpoint"""
    try:
        # Create test image
        test_image = base64.b64encode(b"test_image_data").decode('utf-8')
        
        response = requests.post(
            f"http://localhost:8000/v1/vision/detect",
            json={"image": test_image, "confidence_threshold": 0.25}
        )
        print(f"✅ Object detection: {response.status_code}")
        print(f"Response: {response.json()}")
        return True
    except Exception as e:
        print(f"❌ Object detection failed: {e}")
        return False

def test_vision_chat():
    """Test vision chat endpoint"""
    try:
        test_image = base64.b64encode(b"test_image_data").decode('utf-8')
        
        response = requests.post(
            f"http://localhost:8000/v1/chat/completions",
            json={
                "model": "yolov10-vision",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What is in this image?"},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{test_image}"}}
                        ]
                    }
                ]
            }
        )
        print(f"✅ Vision chat: {response.status_code}")
        print(f"Response: {response.json()}")
        return True
    except Exception as e:
        print(f"❌ Vision chat failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing YOLOv10 deployment...")
    print("=" * 50)
    
    tests = [
        test_health,
        test_model_info,
        test_object_detection,
        test_vision_chat
    ]
    
    results = []
    for test in tests:
        results.append(test())
        time.sleep(1)  # Wait between tests
    
    print("=" * 50)
    passed = sum(results)
    total = len(results)
    print(f"📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed. Check the server logs.")

if __name__ == "__main__":
    main()
