#!/usr/bin/env python3
"""
YOLOv10 Deployment Script for MAX
Deploy the custom YOLOv10 architecture for inference
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import max
    from max import serve
    from max.architectures import register_architecture
    from max.models import Model
    from max.serve import ModelServer
except ImportError as e:
    logger.error(f"MAX not available: {e}")
    logger.info("Please install MAX: pip install max")
    sys.exit(1)

# Import our YOLOv10 model
from yolov10_model.arch import YOLOv10Architecture
from yolov10_model.model import YOLOv10Model
from yolov10_model.model_config import YOLOv10Config


class YOLOv10Deployer:
    """
    Deploy YOLOv10 model for inference using MAX
    """
    
    def __init__(self, model_path: str, config_path: Optional[str] = None):
        self.model_path = Path(model_path)
        self.config_path = Path(config_path) if config_path else None
        
        # Load configuration
        self.config = self._load_config()
        
        logger.info(f"Initialized YOLOv10 deployer for {model_path}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load model configuration"""
        if self.config_path and self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return json.load(f)
        else:
            # Default configuration
            return {
                "model": {
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
                    "precision": "fp16"
                },
                "serving": {
                    "port": 8000,
                    "host": "0.0.0.0",
                    "max_concurrent_requests": 10
                }
            }
    
    def register_architecture(self):
        """Register YOLOv10 architecture with MAX"""
        logger.info("Registering YOLOv10 architecture with MAX...")
        
        try:
            # Register the architecture
            register_architecture("yolov10", YOLOv10Architecture)
            logger.info("✅ YOLOv10 architecture registered successfully")
            
            return True
        except Exception as e:
            logger.error(f"❌ Failed to register architecture: {e}")
            return False
    
    def create_model(self) -> Optional[Model]:
        """Create YOLOv10 model instance"""
        logger.info("Creating YOLOv10 model...")
        
        try:
            # Create model configuration
            model_config = YOLOv10Config(
                input_size=self.config["model"]["input_size"],
                num_classes=self.config["model"]["num_classes"],
                backbone_channels=self.config["model"]["backbone_channels"],
                neck_channels=self.config["model"]["neck_channels"],
                anchors_per_scale=self.config["model"]["anchors_per_scale"]
            )
            
            # Create model
            model = YOLOv10Model(model_config)
            
            # Load weights if available
            if self.model_path.exists():
                logger.info(f"Loading model weights from {self.model_path}")
                model.load_weights(self.model_path)
            
            logger.info("✅ YOLOv10 model created successfully")
            return model
            
        except Exception as e:
            logger.error(f"❌ Failed to create model: {e}")
            return None
    
    def deploy_local(self, port: int = 8000, host: str = "0.0.0.0"):
        """Deploy model locally using MAX Serve"""
        logger.info(f"Deploying YOLOv10 model locally on {host}:{port}")
        
        # Register architecture
        if not self.register_architecture():
            return False
        
        # Create model
        model = self.create_model()
        if not model:
            return False
        
        try:
            # Create model server
            server = ModelServer(
                model=model,
                port=port,
                host=host,
                max_concurrent_requests=self.config["serving"]["max_concurrent_requests"]
            )
            
            logger.info("✅ YOLOv10 model server created")
            logger.info(f"🚀 Starting server on http://{host}:{port}")
            logger.info("📝 API endpoints:")
            logger.info(f"   - POST /v1/vision/detect - Object detection")
            logger.info(f"   - GET  /health - Health check")
            logger.info(f"   - GET  /model/info - Model information")
            
            # Start server
            server.start()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start server: {e}")
            return False
    
    def export_model(self, export_path: str):
        """Export model for deployment"""
        logger.info(f"Exporting YOLOv10 model to {export_path}")
        
        # Register architecture
        if not self.register_architecture():
            return False
        
        # Create model
        model = self.create_model()
        if not model:
            return False
        
        try:
            # Export model
            export_dir = Path(export_path)
            export_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model
            model.save(export_dir / "model")
            
            # Save configuration
            config_file = export_dir / "config.json"
            with open(config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            
            # Create deployment script
            deploy_script = export_dir / "deploy.sh"
            deploy_content = f"""#!/bin/bash
# YOLOv10 Deployment Script
# Generated by YOLOv10 Deployer

echo "Starting YOLOv10 inference server..."

# Set environment variables
export MAX_DEVICE="auto"
export MAX_PRECISION="fp16"

# Start MAX Serve
max serve \\
    --model-path="{export_dir}/model" \\
    --port={self.config['serving']['port']} \\
    --host={self.config['serving']['host']} \\
    --max-concurrent-requests={self.config['serving']['max_concurrent_requests']}

echo "YOLOv10 server started successfully!"
"""
            deploy_script.write_text(deploy_content)
            deploy_script.chmod(0o755)
            
            # Create Dockerfile
            dockerfile = export_dir / "Dockerfile"
            docker_content = f"""# YOLOv10 MAX Container
FROM modular/max:latest

# Copy model files
COPY model/ /app/model/
COPY config.json /app/config.json

# Set environment variables
ENV MAX_DEVICE=auto
ENV MAX_PRECISION=fp16

# Expose port
EXPOSE {self.config['serving']['port']}

# Start server
CMD ["max", "serve", "--model-path=/app/model", "--port={self.config['serving']['port']}", "--host=0.0.0.0"]
"""
            dockerfile.write_text(docker_content)
            
            # Create docker-compose.yml
            compose_file = export_dir / "docker-compose.yml"
            compose_content = f"""version: '3.8'

services:
  yolov10-inference:
    build: .
    ports:
      - "{self.config['serving']['port']}:{self.config['serving']['port']}"
    environment:
      - MAX_DEVICE=auto
      - MAX_PRECISION=fp16
    volumes:
      - ./model:/app/model
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
"""
            compose_file.write_text(compose_content)
            
            # Create README
            readme_file = export_dir / "README.md"
            readme_content = f"""# YOLOv10 Inference Deployment

This directory contains the exported YOLOv10 model for inference deployment.

## Quick Start

### Local Deployment

```bash
# Start the server
./deploy.sh

# Or use MAX CLI directly
max serve --model-path=./model --port={self.config['serving']['port']}
```

### Docker Deployment

```bash
# Build and run with Docker
docker-compose up -d

# Or build manually
docker build -t yolov10-inference .
docker run -p {self.config['serving']['port']}:{self.config['serving']['port']} --gpus all yolov10-inference
```

## API Usage

### Object Detection

```bash
curl -X POST http://localhost:{self.config['serving']['port']}/v1/vision/detect \\
  -H "Content-Type: application/json" \\
  -d '{{"image": "base64_encoded_image", "confidence_threshold": 0.25}}'
```

### Health Check

```bash
curl http://localhost:{self.config['serving']['port']}/health
```

### Model Information

```bash
curl http://localhost:{self.config['serving']['port']}/model/info
```

## Configuration

Model configuration is stored in `config.json`:

```json
{json.dumps(self.config, indent=2)}
```

## Model Details

- **Architecture**: YOLOv10 with CSPDarknet backbone
- **Input Size**: {self.config['model']['input_size']}
- **Classes**: {self.config['model']['num_classes']}
- **Device**: {self.config['inference']['device']}
- **Precision**: {self.config['inference']['precision']}
"""
            readme_file.write_text(readme_content)
            
            logger.info("✅ Model exported successfully")
            logger.info(f"📁 Export directory: {export_dir}")
            logger.info(f"📄 Files created:")
            logger.info(f"   - model/ - Model weights and architecture")
            logger.info(f"   - config.json - Model configuration")
            logger.info(f"   - deploy.sh - Local deployment script")
            logger.info(f"   - Dockerfile - Docker container definition")
            logger.info(f"   - docker-compose.yml - Docker Compose setup")
            logger.info(f"   - README.md - Deployment instructions")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to export model: {e}")
            return False
    
    def test_inference(self, test_image_path: Optional[str] = None):
        """Test model inference"""
        logger.info("Testing YOLOv10 inference...")
        
        # Create model
        model = self.create_model()
        if not model:
            return False
        
        try:
            # Create test input
            import numpy as np
            
            input_size = self.config["model"]["input_size"]
            test_input = np.random.randn(1, *input_size, 3).astype(np.float32)
            
            # Run inference
            logger.info("Running inference test...")
            predictions = model.predict(test_input)
            
            logger.info("✅ Inference test successful")
            logger.info(f"📊 Output shape: {predictions.shape}")
            logger.info(f"📊 Output dtype: {predictions.dtype}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Inference test failed: {e}")
            return False


def main():
    """Main deployment function"""
    parser = argparse.ArgumentParser(description="Deploy YOLOv10 model for inference")
    parser.add_argument("--model-path", type=str, default="trained_model", help="Path to trained model")
    parser.add_argument("--config", type=str, help="Model configuration file")
    parser.add_argument("--export", type=str, help="Export model to directory")
    parser.add_argument("--deploy-local", action="store_true", help="Deploy locally")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--test", action="store_true", help="Test inference")
    
    args = parser.parse_args()
    
    # Create deployer
    deployer = YOLOv10Deployer(args.model_path, args.config)
    
    # Register architecture
    if not deployer.register_architecture():
        logger.error("Failed to register architecture")
        return
    
    # Test inference
    if args.test:
        if not deployer.test_inference():
            logger.error("Inference test failed")
            return
    
    # Export model
    if args.export:
        if not deployer.export_model(args.export):
            logger.error("Model export failed")
            return
    
    # Deploy locally
    if args.deploy_local:
        if not deployer.deploy_local(args.port, args.host):
            logger.error("Local deployment failed")
            return


if __name__ == "__main__":
    main() 