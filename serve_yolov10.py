#!/usr/bin/env python3
"""
Serve YOLOv10 Model with MAX
This script demonstrates how to serve the YOLOv10 model using MAX serve
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def setup_environment():
    """Setup the environment for serving YOLOv10"""
    print("🔧 Setting up YOLOv10 serving environment...")
    
    # Check if we're in the right directory
    current_dir = Path.cwd()
    if not (current_dir / "yolov10_model").exists():
        print("❌ yolov10_model directory not found!")
        print("Please run this script from the DOCVISION directory")
        return False
    
    print("✅ Environment setup complete")
    return True


def create_model_config():
    """Create a sample model configuration"""
    config_content = '''{
    "model_type": "YOLOv10ForObjectDetection",
    "input_size": [640, 640],
    "num_classes": 80,
    "backbone_channels": [32, 64, 128, 256, 512, 1024],
    "neck_channels": 256,
    "anchors_per_scale": 3,
    "confidence_threshold": 0.5,
    "nms_threshold": 0.4
}'''
    
    config_path = Path("yolov10_config.json")
    with open(config_path, "w") as f:
        f.write(config_content)
    
    print(f"✅ Created model config: {config_path}")
    return config_path


def serve_model(port: int = 8000, host: str = "0.0.0.0"):
    """Serve the YOLOv10 model using MAX serve"""
    
    print(f"🚀 Starting YOLOv10 model server on {host}:{port}")
    
    # Create the serve command
    cmd = [
        "max", "serve",
        "--model-path", "yolov10_model",
        "--custom-architectures", "yolov10_model",
        "--port", str(port),
        "--host", host
    ]
    
    print(f"Command: {' '.join(cmd)}")
    
    try:
        # Run the serve command
        result = subprocess.run(cmd, check=True)
        print("✅ Model server started successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start model server: {e}")
        return False
    except FileNotFoundError:
        print("❌ 'max' command not found. Please ensure MAX is installed and in PATH")
        return False


def test_model_endpoint(port: int = 8000):
    """Test the model endpoint"""
    print("🧪 Testing model endpoint...")
    
    test_script = f'''
import requests
import json
import numpy as np

# Test endpoint
url = "http://localhost:{port}/v1/chat/completions"

# Create a dummy image (in practice, you'd load a real image)
dummy_image = np.random.randn(640, 640, 3).astype(np.float32)

# Test payload
payload = {{
    "model": "YOLOv10ForObjectDetection",
    "messages": [
        {{
            "role": "user",
            "content": "Detect objects in this image"
        }}
    ],
    "max_tokens": 100
}}

try:
    response = requests.post(url, json=payload)
    print(f"Status Code: {{response.status_code}}")
    print(f"Response: {{response.text}}")
except Exception as e:
    print(f"Error: {{e}}")
'''
    
    # Write test script
    test_file = Path("test_endpoint.py")
    with open(test_file, "w") as f:
        f.write(test_script)
    
    print(f"✅ Created test script: {test_file}")
    print("Run 'python test_endpoint.py' to test the endpoint")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Serve YOLOv10 model with MAX")
    parser.add_argument("--port", type=int, default=8000, help="Port to serve on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to serve on")
    parser.add_argument("--test", action="store_true", help="Test the endpoint after starting")
    parser.add_argument("--config-only", action="store_true", help="Only create config, don't serve")
    
    args = parser.parse_args()
    
    print("🎯 YOLOv10 Model Serving with MAX")
    print("=" * 50)
    
    # Setup environment
    if not setup_environment():
        sys.exit(1)
    
    # Create model config
    config_path = create_model_config()
    
    if args.config_only:
        print("✅ Configuration created. Use --config-only=false to serve the model")
        return
    
    # Serve the model
    if serve_model(args.port, args.host):
        print(f"\n🎉 YOLOv10 model is now serving on http://{args.host}:{args.port}")
        print("\n📋 Available endpoints:")
        print(f"  - Chat completions: http://{args.host}:{args.port}/v1/chat/completions")
        print(f"  - Model info: http://{args.host}:{args.port}/v1/models")
        
        if args.test:
            test_model_endpoint(args.port)
        
        print("\n🔗 Example usage:")
        print("curl -X POST http://localhost:8000/v1/chat/completions \\")
        print("  -H 'Content-Type: application/json' \\")
        print("  -d '{\"model\": \"YOLOv10ForObjectDetection\", \"messages\": [{\"role\": \"user\", \"content\": \"Detect objects\"}]}'")
        
        print("\n⏹️  Press Ctrl+C to stop the server")
    else:
        print("❌ Failed to start the model server")
        sys.exit(1)


if __name__ == "__main__":
    main() 