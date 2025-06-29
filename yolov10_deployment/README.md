# YOLOv10 Vision Model Deployment

This directory contains the deployment files for the YOLOv10 vision model using MAX.

## Quick Start

### Local Deployment
```bash
# Start the server
./deploy.sh

# Or use MAX CLI directly
max serve --model-path=./model --port=8000
```

### Docker Deployment
```bash
# Build and run with Docker
docker-compose up -d

# Or build manually
docker build -t yolov10-vision .
docker run -p 8000:8000 --gpus all yolov10-vision
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
curl http://localhost:8000/health
```

### Object Detection
```bash
curl -X POST http://localhost:8000/v1/vision/detect \
  -H "Content-Type: application/json" \
  -d '{"image": "base64_encoded_image", "confidence_threshold": 0.25}'
```

### Vision Chat
```bash
curl -N http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "yolov10-vision", "messages": [{"role": "user", "content": [{"type": "text", "text": "What is in this image?"}, {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ..."}}]}]}'
```

## Configuration

Model configuration is stored in `model_config.json`:

```json
{
  "model": {
    "name": "yolov10-vision",
    "version": "1.0.0",
    "architecture": "YOLOv10",
    "input_size": [
      640,
      640
    ],
    "num_classes": 80,
    "backbone_channels": [
      32,
      64,
      128,
      256,
      512,
      1024
    ],
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
```

## Architecture

The YOLOv10 model consists of:

1. **CSPDarknet Backbone**: Feature extraction network
2. **PANet Neck**: Feature fusion network
3. **Detection Heads**: Multi-scale object detection

## Performance

- **Input Size**: [640, 640]
- **Classes**: 80
- **Device**: auto
- **Precision**: fp16
- **Concurrent Requests**: 10

## Monitoring

- Health check endpoint: `/health`
- Model information: `/model/info`
- Metrics: Available through MAX monitoring

## Troubleshooting

1. **GPU Not Available**: Ensure NVIDIA drivers and Docker GPU support are installed
2. **Memory Issues**: Reduce batch size or use model quantization
3. **API Errors**: Check request format and image encoding

For more information, see the [API Documentation](API.md).
