#!/usr/bin/env python3
"""
YOLOv10 MAX Inference Server
Custom inference server for YOLOv10 model using MAX
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

try:
    import max
    from max import serve
    from max.graph import Graph, ops
    from max.dtype import DType
except ImportError as e:
    logger.error(f"MAX not available: {e}")
    logger.info("Please install MAX: pip install max")
    sys.exit(1)

# Import our YOLOv10 model
from yolov10_model.model import YOLOv10Model
from yolov10_model.model_config import YOLOv10Config


class YOLOv10InferenceServer:
    """
    YOLOv10 inference server using MAX
    """
    
    def __init__(self, model_path: str, config_path: Optional[str] = None):
        self.model_path = Path(model_path)
        self.config_path = Path(config_path) if config_path else None
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize model
        self.model = self._load_model()
        
        logger.info(f"Initialized YOLOv10 inference server")
    
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
                }
            }
    
    def _load_model(self) -> Optional[YOLOv10Model]:
        """Load YOLOv10 model"""
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
            else:
                logger.warning(f"Model weights not found at {self.model_path}, using random weights")
            
            logger.info("✅ YOLOv10 model loaded successfully")
            return model
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            return None
    
    def preprocess_image(self, image_data: str) -> np.ndarray:
        """Preprocess image for inference"""
        try:
            # Decode base64 image
            if image_data.startswith('data:image'):
                # Remove data URL prefix
                image_data = image_data.split(',')[1]
            
            image_bytes = base64.b64decode(image_data)
            
            # In a real implementation, you'd use PIL or OpenCV
            # For now, create a dummy image
            input_size = self.config["model"]["input_size"]
            image = np.random.randn(*input_size, 3).astype(np.float32)
            
            # Normalize to [0, 1]
            image = (image - image.min()) / (image.max() - image.min())
            
            # Add batch dimension
            image = np.expand_dims(image, axis=0)
            
            return image
            
        except Exception as e:
            logger.error(f"Failed to preprocess image: {e}")
            raise
    
    def postprocess_predictions(self, predictions: np.ndarray) -> List[Dict[str, Any]]:
        """Postprocess model predictions"""
        try:
            # In a real implementation, you'd:
            # 1. Apply confidence threshold
            # 2. Perform non-maximum suppression
            # 3. Convert to bounding boxes
            # 4. Map class IDs to class names
            
            # For now, return dummy detections
            detections = []
            
            # Simulate some detections
            for i in range(min(5, predictions.shape[0])):
                class_id = i % self.config["model"]["num_classes"]
                detection = {
                    "bbox": [0.1 + i * 0.1, 0.1 + i * 0.1, 0.2, 0.2],  # [x, y, w, h]
                    "confidence": 0.8 - i * 0.1,
                    "class_id": class_id,
                    "class_name": f"class_{class_id}"
                }
                detections.append(detection)
            
            return detections
            
        except Exception as e:
            logger.error(f"Failed to postprocess predictions: {e}")
            return []
    
    def predict(self, image_data: str, confidence_threshold: float = 0.25) -> Dict[str, Any]:
        """Run inference on image"""
        try:
            if not self.model:
                return {
                    "error": "Model not loaded",
                    "detections": [],
                    "num_detections": 0
                }
            
            # Preprocess image
            image = self.preprocess_image(image_data)
            
            # Run inference
            predictions = self.model.predict(image)
            
            # Postprocess predictions
            detections = self.postprocess_predictions(predictions)
            
            # Filter by confidence threshold
            detections = [d for d in detections if d["confidence"] >= confidence_threshold]
            
            return {
                "detections": detections,
                "num_detections": len(detections),
                "model_info": {
                    "architecture": "YOLOv10",
                    "input_size": self.config["model"]["input_size"],
                    "num_classes": self.config["model"]["num_classes"]
                }
            }
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return {
                "error": str(e),
                "detections": [],
                "num_detections": 0
            }


def create_max_app():
    """Create MAX application for serving"""
    try:
        from flask import Flask, request, jsonify
    except ImportError:
        logger.error("Flask not available. Please install: pip install flask")
        return None
    
    app = Flask(__name__)
    
    # Initialize server
    model_path = os.getenv("MODEL_PATH", "trained_model")
    config_path = os.getenv("CONFIG_PATH")
    
    server = YOLOv10InferenceServer(model_path, config_path)
    
    if not server.model:
        logger.error("Failed to initialize model")
        return None
    
    @app.route('/health', methods=['GET'])
    def health_check():
        """Health check endpoint"""
        return jsonify({
            "status": "healthy",
            "model": "YOLOv10",
            "version": "1.0.0"
        })
    
    @app.route('/model/info', methods=['GET'])
    def model_info():
        """Model information endpoint"""
        return jsonify({
            "architecture": "YOLOv10",
            "input_size": server.config["model"]["input_size"],
            "num_classes": server.config["model"]["num_classes"],
            "confidence_threshold": server.config["model"]["confidence_threshold"],
            "nms_threshold": server.config["model"]["nms_threshold"]
        })
    
    @app.route('/v1/vision/detect', methods=['POST'])
    def detect_objects():
        """Object detection endpoint"""
        try:
            data = request.get_json()
            
            if not data or 'image' not in data:
                return jsonify({"error": "Missing image data"}), 400
            
            image_data = data['image']
            confidence_threshold = data.get('confidence_threshold', 0.25)
            
            # Run inference
            result = server.predict(image_data, confidence_threshold)
            
            return jsonify(result)
            
        except Exception as e:
            logger.error(f"Detection request failed: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route('/v1/chat/completions', methods=['POST'])
    def vision_chat():
        """Vision chat endpoint (compatible with MAX vision models)"""
        try:
            data = request.get_json()
            
            if not data or 'messages' not in data:
                return jsonify({"error": "Missing messages"}), 400
            
            messages = data['messages']
            
            # Find image in messages
            image_data = None
            text_prompt = "What do you see in this image?"
            
            for message in messages:
                if message.get('role') == 'user' and 'content' in message:
                    content = message['content']
                    if isinstance(content, list):
                        for item in content:
                            if item.get('type') == 'image_url':
                                image_data = item['image_url']['url']
                            elif item.get('type') == 'text':
                                text_prompt = item['text']
                    elif isinstance(content, str):
                        text_prompt = content
            
            if not image_data:
                return jsonify({"error": "No image found in request"}), 400
            
            # Run object detection
            detection_result = server.predict(image_data)
            
            # Generate response based on detections
            if detection_result.get('detections'):
                detections = detection_result['detections']
                response_text = f"I can see {len(detections)} objects in this image:\n"
                
                for i, detection in enumerate(detections[:5]):  # Limit to 5 detections
                    response_text += f"{i+1}. {detection['class_name']} (confidence: {detection['confidence']:.2f})\n"
            else:
                response_text = "I don't see any objects in this image."
            
            # Format response like MAX vision models
            response = {
                "id": "yolov10-vision-response",
                "object": "chat.completion",
                "created": 1234567890,
                "model": "yolov10-vision",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": response_text
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
            }
            
            return jsonify(response)
            
        except Exception as e:
            logger.error(f"Vision chat request failed: {e}")
            return jsonify({"error": str(e)}), 500
    
    return app


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="YOLOv10 MAX Inference Server")
    parser.add_argument("--model-path", type=str, default="trained_model", help="Path to trained model")
    parser.add_argument("--config", type=str, help="Model configuration file")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--test", action="store_true", help="Test inference")
    
    args = parser.parse_args()
    
    # Set environment variables
    os.environ["MODEL_PATH"] = args.model_path
    if args.config:
        os.environ["CONFIG_PATH"] = args.config
    
    # Test inference
    if args.test:
        logger.info("Testing YOLOv10 inference...")
        server = YOLOv10InferenceServer(args.model_path, args.config)
        
        if server.model:
            # Create test image (base64 encoded dummy data)
            test_image = base64.b64encode(b"dummy_image_data").decode('utf-8')
            
            result = server.predict(test_image)
            logger.info(f"Test result: {result}")
        else:
            logger.error("Model initialization failed")
            return
    
    # Create and run Flask app
    app = create_max_app()
    if app:
        logger.info(f"🚀 Starting YOLOv10 inference server on {args.host}:{args.port}")
        logger.info("📝 Available endpoints:")
        logger.info("   - POST /v1/vision/detect - Object detection")
        logger.info("   - POST /v1/chat/completions - Vision chat (MAX compatible)")
        logger.info("   - GET  /health - Health check")
        logger.info("   - GET  /model/info - Model information")
        
        app.run(host=args.host, port=args.port, debug=False)
    else:
        logger.error("Failed to create application")


if __name__ == "__main__":
    main() 