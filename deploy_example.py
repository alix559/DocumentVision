#!/usr/bin/env python3
"""
YOLOv10 Deployment Example
Demonstrates deployment following MAX documentation patterns
"""

import os
import sys
import json
import base64
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class YOLOv10DeploymentExample:
    """
    YOLOv10 deployment example following MAX patterns
    """
    
    def __init__(self):
        self.model_name = "yolov10-vision"
        self.version = "1.0.0"
        
        logger.info(f"Initialized YOLOv10 deployment example: {self.model_name} v{self.version}")
    
    def create_deployment_files(self, output_dir: str = "yolov10_deployment"):
        """Create deployment files following MAX patterns"""
        logger.info(f"Creating deployment files in {output_dir}")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 1. Create model configuration
        model_config = {
            "model": {
                "name": self.model_name,
                "version": self.version,
                "architecture": "YOLOv10",
                "input_size": [640, 640],
                "num_classes": 80,
                "backbone_channels": [32, 64, 128, 256, 512, 1024],
                "neck_channels": 256,
                "anchors_per_scale": 3,
                "confidence_threshold": 0.25,
                "nms_threshold": 0.45
            },
            "inference": {
                "batch_size": 1,
                "device": "auto",
                "precision": "fp16",
                "max_concurrent_requests": 10
            },
            "serving": {
                "port": 8000,
                "host": "0.0.0.0",
                "endpoints": [
                    "/v1/vision/detect",
                    "/v1/chat/completions",
                    "/health",
                    "/model/info"
                ]
            }
        }
        
        config_file = output_path / "model_config.json"
        with open(config_file, 'w') as f:
            json.dump(model_config, f, indent=2)
        
        # 2. Create deployment script
        deploy_script = output_path / "deploy.sh"
        deploy_content = f"""#!/bin/bash
# YOLOv10 Deployment Script
# Following MAX documentation patterns

echo "🚀 Deploying YOLOv10 Vision Model"

# Set environment variables
export MAX_DEVICE="auto"
export MAX_PRECISION="fp16"
export MODEL_NAME="{self.model_name}"
export MODEL_VERSION="{self.version}"

# Check if MAX is available
if ! command -v max &> /dev/null; then
    echo "❌ MAX CLI not found. Please install MAX first."
    exit 1
fi

# Start MAX serve with custom model
echo "📡 Starting MAX serve..."
max serve \\
    --model-path="./model" \\
    --port={model_config['serving']['port']} \\
    --host={model_config['serving']['host']} \\
    --max-concurrent-requests={model_config['inference']['max_concurrent_requests']}

echo "✅ YOLOv10 deployment completed!"
"""
        deploy_script.write_text(deploy_content)
        deploy_script.chmod(0o755)
        
        # 3. Create Dockerfile
        dockerfile = output_path / "Dockerfile"
        docker_content = f"""# YOLOv10 MAX Container
# Based on MAX documentation patterns

FROM modular/max:latest

# Set environment variables
ENV MAX_DEVICE=auto
ENV MAX_PRECISION=fp16
ENV MODEL_NAME={self.model_name}
ENV MODEL_VERSION={self.version}

# Copy model files
COPY model/ /app/model/
COPY model_config.json /app/config.json

# Expose port
EXPOSE {model_config['serving']['port']}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:{model_config['serving']['port']}/health || exit 1

# Start server
CMD ["max", "serve", "--model-path=/app/model", "--port={model_config['serving']['port']}", "--host=0.0.0.0"]
"""
        dockerfile.write_text(docker_content)
        
        # 4. Create docker-compose.yml
        compose_file = output_path / "docker-compose.yml"
        compose_content = f"""version: '3.8'

services:
  yolov10-vision:
    build: .
    ports:
      - "{model_config['serving']['port']}:{model_config['serving']['port']}"
    environment:
      - MAX_DEVICE=auto
      - MAX_PRECISION=fp16
      - MODEL_NAME={self.model_name}
      - MODEL_VERSION={self.version}
    volumes:
      - ./model:/app/model
      - ./config.json:/app/config.json
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:{model_config['serving']['port']}/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
"""
        compose_file.write_text(compose_content)
        
        # 5. Create Kubernetes deployment
        k8s_file = output_path / "k8s-deployment.yaml"
        k8s_content = f"""apiVersion: apps/v1
kind: Deployment
metadata:
  name: yolov10-vision
  labels:
    app: yolov10-vision
spec:
  replicas: 1
  selector:
    matchLabels:
      app: yolov10-vision
  template:
    metadata:
      labels:
        app: yolov10-vision
    spec:
      containers:
      - name: yolov10-vision
        image: yolov10-vision:latest
        ports:
        - containerPort: {model_config['serving']['port']}
        env:
        - name: MAX_DEVICE
          value: "auto"
        - name: MAX_PRECISION
          value: "fp16"
        - name: MODEL_NAME
          value: "{self.model_name}"
        - name: MODEL_VERSION
          value: "{self.version}"
        resources:
          limits:
            nvidia.com/gpu: 1
          requests:
            memory: "4Gi"
            cpu: "2"
        livenessProbe:
          httpGet:
            path: /health
            port: {model_config['serving']['port']}
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: {model_config['serving']['port']}
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: yolov10-vision-service
spec:
  selector:
    app: yolov10-vision
  ports:
    - protocol: TCP
      port: 80
      targetPort: {model_config['serving']['port']}
  type: LoadBalancer
"""
        k8s_file.write_text(k8s_content)
        
        # 6. Create API documentation
        api_docs = output_path / "API.md"
        api_content = f"""# YOLOv10 Vision API

This document describes the API endpoints for the YOLOv10 vision model deployed with MAX.

## Base URL
```
http://localhost:{model_config['serving']['port']}
```

## Endpoints

### 1. Object Detection
**POST** `/v1/vision/detect`

Detect objects in an image.

**Request Body:**
```json
{{
  "image": "base64_encoded_image_data",
  "confidence_threshold": 0.25
}}
```

**Response:**
```json
{{
  "detections": [
    {{
      "bbox": [0.1, 0.2, 0.3, 0.4],
      "confidence": 0.85,
      "class_id": 0,
      "class_name": "person"
    }}
  ],
  "num_detections": 1,
  "model_info": {{
    "architecture": "YOLOv10",
    "input_size": [640, 640],
    "num_classes": 80
  }}
}}
```

### 2. Vision Chat (MAX Compatible)
**POST** `/v1/chat/completions`

Vision chat endpoint compatible with MAX vision models.

**Request Body:**
```json
{{
  "model": "{self.model_name}",
  "messages": [
    {{
      "role": "user",
      "content": [
        {{
          "type": "text",
          "text": "What is in this image?"
        }},
        {{
          "type": "image_url",
          "image_url": {{
            "url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ..."
          }}
        }}
      ]
    }}
  ],
  "max_tokens": 300
}}
```

**Response:**
```json
{{
  "id": "yolov10-vision-response",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "{self.model_name}",
  "choices": [
    {{
      "index": 0,
      "message": {{
        "role": "assistant",
        "content": "I can see 3 objects in this image:\\n1. person (confidence: 0.85)\\n2. car (confidence: 0.72)"
      }},
      "finish_reason": "stop"
    }}
  ],
  "usage": {{
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0
  }}
}}
```

### 3. Health Check
**GET** `/health`

Check server health.

**Response:**
```json
{{
  "status": "healthy",
  "model": "YOLOv10",
  "version": "{self.version}"
}}
```

### 4. Model Information
**GET** `/model/info`

Get model information.

**Response:**
```json
{{
  "architecture": "YOLOv10",
  "input_size": [640, 640],
  "num_classes": 80,
  "confidence_threshold": 0.25,
  "nms_threshold": 0.45
}}
```

## Usage Examples

### Python Client
```python
import requests
import base64

# Object detection
with open("image.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode('utf-8')

response = requests.post(
    "http://localhost:{model_config['serving']['port']}/v1/vision/detect",
    json={{"image": image_data, "confidence_threshold": 0.25}}
)

print(response.json())
```

### cURL Examples
```bash
# Health check
curl http://localhost:{model_config['serving']['port']}/health

# Model info
curl http://localhost:{model_config['serving']['port']}/model/info

# Object detection
curl -X POST http://localhost:{model_config['serving']['port']}/v1/vision/detect \\
  -H "Content-Type: application/json" \\
  -d '{{"image": "base64_encoded_image", "confidence_threshold": 0.25}}'
```

## Model Details

- **Architecture**: YOLOv10 with CSPDarknet backbone
- **Input Size**: {model_config['model']['input_size']}
- **Classes**: {model_config['model']['num_classes']}
- **Device**: {model_config['inference']['device']}
- **Precision**: {model_config['inference']['precision']}
"""
        api_docs.write_text(api_content)
        
        # 7. Create README
        readme_file = output_path / "README.md"
        readme_content = f"""# YOLOv10 Vision Model Deployment

This directory contains the deployment files for the YOLOv10 vision model using MAX.

## Quick Start

### Local Deployment
```bash
# Start the server
./deploy.sh

# Or use MAX CLI directly
max serve --model-path=./model --port={model_config['serving']['port']}
```

### Docker Deployment
```bash
# Build and run with Docker
docker-compose up -d

# Or build manually
docker build -t yolov10-vision .
docker run -p {model_config['serving']['port']}:{model_config['serving']['port']} --gpus all yolov10-vision
```

### Kubernetes Deployment
```bash
# Deploy to Kubernetes
kubectl apply -f k8s-deployment.yaml

# Check deployment
kubectl get pods -l app=yolov10-vision
```

## Testing

### Health Check
```bash
curl http://localhost:{model_config['serving']['port']}/health
```

### Object Detection
```bash
curl -X POST http://localhost:{model_config['serving']['port']}/v1/vision/detect \\
  -H "Content-Type: application/json" \\
  -d '{{"image": "base64_encoded_image", "confidence_threshold": 0.25}}'
```

### Vision Chat
```bash
curl -N http://localhost:{model_config['serving']['port']}/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{{"model": "{self.model_name}", "messages": [{{"role": "user", "content": [{{"type": "text", "text": "What is in this image?"}}, {{"type": "image_url", "image_url": {{"url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ..."}}}}]}}]}}'
```

## Configuration

Model configuration is stored in `model_config.json`:

```json
{json.dumps(model_config, indent=2)}
```

## Architecture

The YOLOv10 model consists of:

1. **CSPDarknet Backbone**: Feature extraction network
2. **PANet Neck**: Feature fusion network
3. **Detection Heads**: Multi-scale object detection

## Performance

- **Input Size**: {model_config['model']['input_size']}
- **Classes**: {model_config['model']['num_classes']}
- **Device**: {model_config['inference']['device']}
- **Precision**: {model_config['inference']['precision']}
- **Concurrent Requests**: {model_config['inference']['max_concurrent_requests']}

## Monitoring

- Health check endpoint: `/health`
- Model information: `/model/info`
- Metrics: Available through MAX monitoring

## Troubleshooting

1. **GPU Not Available**: Ensure NVIDIA drivers and Docker GPU support are installed
2. **Memory Issues**: Reduce batch size or use model quantization
3. **API Errors**: Check request format and image encoding

For more information, see the [API Documentation](API.md).
"""
        readme_file.write_text(readme_content)
        
        # 8. Create test script
        test_script = output_path / "test_deployment.py"
        test_content = f"""#!/usr/bin/env python3
\"\"\"
Test script for YOLOv10 deployment
\"\"\"

import requests
import base64
import json
import time

def test_health():
    \"\"\"Test health endpoint\"\"\"
    try:
        response = requests.get("http://localhost:{model_config['serving']['port']}/health")
        print(f"✅ Health check: {{response.status_code}}")
        print(f"Response: {{response.json()}}")
        return True
    except Exception as e:
        print(f"❌ Health check failed: {{e}}")
        return False

def test_model_info():
    \"\"\"Test model info endpoint\"\"\"
    try:
        response = requests.get("http://localhost:{model_config['serving']['port']}/model/info")
        print(f"✅ Model info: {{response.status_code}}")
        print(f"Response: {{response.json()}}")
        return True
    except Exception as e:
        print(f"❌ Model info failed: {{e}}")
        return False

def test_object_detection():
    \"\"\"Test object detection endpoint\"\"\"
    try:
        # Create test image
        test_image = base64.b64encode(b"test_image_data").decode('utf-8')
        
        response = requests.post(
            f"http://localhost:{model_config['serving']['port']}/v1/vision/detect",
            json={{"image": test_image, "confidence_threshold": 0.25}}
        )
        print(f"✅ Object detection: {{response.status_code}}")
        print(f"Response: {{response.json()}}")
        return True
    except Exception as e:
        print(f"❌ Object detection failed: {{e}}")
        return False

def test_vision_chat():
    \"\"\"Test vision chat endpoint\"\"\"
    try:
        test_image = base64.b64encode(b"test_image_data").decode('utf-8')
        
        response = requests.post(
            f"http://localhost:{model_config['serving']['port']}/v1/chat/completions",
            json={{
                "model": "{self.model_name}",
                "messages": [
                    {{
                        "role": "user",
                        "content": [
                            {{"type": "text", "text": "What is in this image?"}},
                            {{"type": "image_url", "image_url": {{"url": f"data:image/jpeg;base64,{{test_image}}"}}}}
                        ]
                    }}
                ]
            }}
        )
        print(f"✅ Vision chat: {{response.status_code}}")
        print(f"Response: {{response.json()}}")
        return True
    except Exception as e:
        print(f"❌ Vision chat failed: {{e}}")
        return False

def main():
    \"\"\"Run all tests\"\"\"
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
    print(f"📊 Test Results: {{passed}}/{{total}} passed")
    
    if passed == total:
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed. Check the server logs.")

if __name__ == "__main__":
    main()
"""
        test_script.write_text(test_content)
        test_script.chmod(0o755)
        
        # 9. Create model directory placeholder
        model_dir = output_path / "model"
        model_dir.mkdir(exist_ok=True)
        
        # Create placeholder model file
        placeholder = model_dir / "README.md"
        placeholder.write_text(f"""# YOLOv10 Model Directory

This directory should contain your trained YOLOv10 model files.

## Expected Files

- Model weights (e.g., `yolov10_weights.pt`)
- Model configuration (e.g., `yolov10_config.json`)
- Architecture definition (e.g., `yolov10_arch.py`)

## Model Format

The model should be compatible with MAX serving. You can:

1. Train your model using the training scripts
2. Export the model to this directory
3. Update the deployment configuration as needed

## Example

```bash
# Copy your trained model here
cp /path/to/trained/model/* ./model/

# Start deployment
./deploy.sh
```
""")
        
        logger.info("✅ Deployment files created successfully")
        logger.info(f"📁 Files created in {output_path}:")
        logger.info(f"   - model_config.json - Model configuration")
        logger.info(f"   - deploy.sh - Local deployment script")
        logger.info(f"   - Dockerfile - Docker container definition")
        logger.info(f"   - docker-compose.yml - Docker Compose setup")
        logger.info(f"   - k8s-deployment.yaml - Kubernetes deployment")
        logger.info(f"   - API.md - API documentation")
        logger.info(f"   - README.md - Deployment instructions")
        logger.info(f"   - test_deployment.py - Test script")
        logger.info(f"   - model/ - Model directory (placeholder)")
        
        return output_path
    
    def demonstrate_deployment(self):
        """Demonstrate the deployment process"""
        logger.info("🎯 YOLOv10 Deployment Demonstration")
        logger.info("=" * 50)
        
        # Create deployment files
        deployment_dir = self.create_deployment_files()
        
        logger.info("\n📋 Deployment Steps:")
        logger.info("1. ✅ Created deployment files")
        logger.info("2. 🔧 Configure your model in the 'model' directory")
        logger.info("3. 🚀 Run deployment:")
        logger.info(f"   cd {deployment_dir}")
        logger.info("   ./deploy.sh")
        logger.info("4. 🧪 Test the deployment:")
        logger.info("   python test_deployment.py")
        logger.info("5. 📊 Monitor and scale as needed")
        
        logger.info("\n🌐 Available Deployment Options:")
        logger.info("   - Local: ./deploy.sh")
        logger.info("   - Docker: docker-compose up -d")
        logger.info("   - Kubernetes: kubectl apply -f k8s-deployment.yaml")
        logger.info("   - Cloud: Follow cloud provider guides")
        
        logger.info("\n📚 Next Steps:")
        logger.info("1. Train your YOLOv10 model using the training scripts")
        logger.info("2. Export the trained model to the deployment directory")
        logger.info("3. Customize the configuration for your use case")
        logger.info("4. Deploy to your preferred environment")
        logger.info("5. Monitor performance and scale as needed")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="YOLOv10 Deployment Example")
    parser.add_argument("--output-dir", type=str, default="yolov10_deployment", help="Output directory")
    parser.add_argument("--demo", action="store_true", help="Run deployment demonstration")
    
    args = parser.parse_args()
    
    # Create deployment example
    deployer = YOLOv10DeploymentExample()
    
    if args.demo:
        deployer.demonstrate_deployment()
    else:
        deployer.create_deployment_files(args.output_dir)


if __name__ == "__main__":
    main() 