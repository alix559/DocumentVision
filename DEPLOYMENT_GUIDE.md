# YOLOv10 Deployment Guide

This guide explains how to deploy your custom YOLOv10 architecture for inference using MAX.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Deployment](#local-deployment)
3. [Docker Deployment](#docker-deployment)
4. [Cloud Deployment](#cloud-deployment)
5. [API Usage](#api-usage)
6. [Testing](#testing)
7. [Troubleshooting](#troubleshooting)

## Prerequisites

Before deploying, ensure you have:

- MAX environment set up with pixi
- Trained YOLOv10 model (or use the architecture for inference)
- Python dependencies installed
- Docker (for containerized deployment)

```bash
# Activate the environment
pixi shell

# Install additional dependencies
pip install flask
```

## Local Deployment

### 1. Quick Start

```bash
# Start the inference server
python max_inference_server.py --model-path trained_model --port 8000

# Test the server
curl http://localhost:8000/health
```

### 2. Using MAX CLI

```bash
# Export your model for MAX serving
python deploy_yolov10.py --export ./yolov10_deployment

# Start MAX serve
max serve --model-path ./yolov10_deployment/model --port 8000
```

### 3. Custom Configuration

```bash
# Create custom deployment configuration
python deploy_yolov10.py \
    --model-path trained_model \
    --config custom_config.json \
    --export ./custom_deployment
```

## Docker Deployment

### 1. Build and Run

```bash
# Export model with Docker files
python deploy_yolov10.py --export ./yolov10_docker

# Navigate to deployment directory
cd yolov10_docker

# Build Docker image
docker build -t yolov10-inference .

# Run container
docker run -p 8000:8000 --gpus all yolov10-inference
```

### 2. Docker Compose

```bash
# Start with Docker Compose
docker-compose up -d

# Check logs
docker-compose logs -f

# Stop services
docker-compose down
```

### 3. Multi-GPU Deployment

```yaml
# docker-compose.yml
version: '3.8'

services:
  yolov10-inference:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MAX_DEVICE=auto
      - MAX_PRECISION=fp16
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    restart: unless-stopped
```

## Cloud Deployment

### 1. Google Cloud Run

```bash
# Build and push to Google Container Registry
gcloud builds submit --tag gcr.io/PROJECT_ID/yolov10-inference

# Deploy to Cloud Run
gcloud run deploy yolov10-inference \
    --image gcr.io/PROJECT_ID/yolov10-inference \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --port 8000
```

### 2. AWS ECS

```bash
# Build and push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin ACCOUNT.dkr.ecr.us-east-1.amazonaws.com

docker build -t yolov10-inference .
docker tag yolov10-inference:latest ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/yolov10-inference:latest
docker push ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/yolov10-inference:latest

# Deploy to ECS
aws ecs create-service \
    --cluster your-cluster \
    --service-name yolov10-inference \
    --task-definition yolov10-task \
    --desired-count 1
```

### 3. Azure Container Instances

```bash
# Build and push to Azure Container Registry
az acr build --registry your-registry --image yolov10-inference .

# Deploy to Container Instances
az container create \
    --resource-group your-rg \
    --name yolov10-inference \
    --image your-registry.azurecr.io/yolov10-inference:latest \
    --ports 8000 \
    --dns-name-label yolov10-inference
```

## API Usage

### 1. Object Detection

```bash
# Detect objects in an image
curl -X POST http://localhost:8000/v1/vision/detect \
  -H "Content-Type: application/json" \
  -d '{
    "image": "base64_encoded_image_data",
    "confidence_threshold": 0.25
  }'
```

**Response:**
```json
{
  "detections": [
    {
      "bbox": [0.1, 0.2, 0.3, 0.4],
      "confidence": 0.85,
      "class_id": 0,
      "class_name": "person"
    }
  ],
  "num_detections": 1,
  "model_info": {
    "architecture": "YOLOv10",
    "input_size": [640, 640],
    "num_classes": 80
  }
}
```

### 2. Vision Chat (MAX Compatible)

```bash
# Vision chat endpoint (compatible with MAX vision models)
curl -N http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "yolov10-vision",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "What is in this image?"
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ..."
            }
          }
        ]
      }
    ],
    "max_tokens": 300
  }'
```

**Response:**
```json
{
  "id": "yolov10-vision-response",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "yolov10-vision",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "I can see 3 objects in this image:\n1. person (confidence: 0.85)\n2. car (confidence: 0.72)\n3. bicycle (confidence: 0.68)"
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
```

### 3. Health Check

```bash
# Check server health
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "model": "YOLOv10",
  "version": "1.0.0"
}
```

### 4. Model Information

```bash
# Get model information
curl http://localhost:8000/model/info
```

**Response:**
```json
{
  "architecture": "YOLOv10",
  "input_size": [640, 640],
  "num_classes": 80,
  "confidence_threshold": 0.25,
  "nms_threshold": 0.45
}
```

## Testing

### 1. Test Inference

```bash
# Test model inference
python max_inference_server.py --test --model-path trained_model
```

### 2. Test API Endpoints

```bash
# Test object detection
python -c "
import requests
import base64

# Create test image
test_image = base64.b64encode(b'test_image_data').decode('utf-8')

# Send request
response = requests.post('http://localhost:8000/v1/vision/detect', 
    json={'image': test_image, 'confidence_threshold': 0.25})

print('Response:', response.json())
"
```

### 3. Load Testing

```bash
# Install load testing tool
pip install locust

# Create load test file
cat > load_test.py << 'EOF'
from locust import HttpUser, task, between

class YOLOv10User(HttpUser):
    wait_time = between(1, 3)
    
    @task
    def detect_objects(self):
        test_image = "base64_encoded_test_image"
        self.client.post("/v1/vision/detect", 
            json={"image": test_image, "confidence_threshold": 0.25})
    
    @task
    def health_check(self):
        self.client.get("/health")
EOF

# Run load test
locust -f load_test.py --host=http://localhost:8000
```

## Performance Optimization

### 1. Model Optimization

```python
# Enable mixed precision
os.environ["MAX_PRECISION"] = "fp16"

# Use GPU acceleration
os.environ["MAX_DEVICE"] = "cuda"

# Optimize batch size
config = {
    "inference": {
        "batch_size": 4,  # Adjust based on GPU memory
        "device": "cuda",
        "precision": "fp16"
    }
}
```

### 2. Server Optimization

```python
# Increase concurrent requests
server = ModelServer(
    model=model,
    port=8000,
    max_concurrent_requests=20  # Adjust based on hardware
)

# Enable request batching
config = {
    "serving": {
        "max_concurrent_requests": 20,
        "batch_timeout": 0.1,
        "max_batch_size": 8
    }
}
```

### 3. Monitoring

```bash
# Monitor server performance
curl http://localhost:8000/metrics

# Check GPU utilization
nvidia-smi

# Monitor memory usage
docker stats yolov10-inference
```

## Troubleshooting

### Common Issues

1. **Model Loading Failed**
   ```
   Solution: Check model path and ensure weights are available
   ```

2. **GPU Not Available**
   ```
   Solution: Install NVIDIA drivers and Docker GPU support
   ```

3. **Memory Issues**
   ```
   Solution: Reduce batch size or use model quantization
   ```

4. **API Errors**
   ```
   Solution: Check request format and image encoding
   ```

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Start server with debug
python max_inference_server.py --port 8000 --debug

# Check logs
docker logs yolov10-inference
```

### Performance Tuning

1. **Batch Processing**
   - Increase batch size for better throughput
   - Use request batching for multiple images

2. **Model Quantization**
   - Use INT8 quantization for faster inference
   - Enable TensorRT optimization

3. **Caching**
   - Cache model weights in memory
   - Use Redis for request caching

## Production Deployment

### 1. Security

```bash
# Enable authentication
export API_KEY="your-secret-key"

# Use HTTPS
docker run -p 443:8000 -e SSL_CERT=/path/to/cert yolov10-inference
```

### 2. Monitoring

```bash
# Set up monitoring
docker run -d \
  --name prometheus \
  -p 9090:9090 \
  prom/prometheus

# Add Grafana dashboard
docker run -d \
  --name grafana \
  -p 3000:3000 \
  grafana/grafana
```

### 3. Scaling

```bash
# Scale horizontally
docker-compose up --scale yolov10-inference=3

# Use Kubernetes
kubectl apply -f k8s-deployment.yaml
kubectl scale deployment yolov10-inference --replicas=5
```

## Next Steps

1. **Customize Model**: Adapt the architecture for your specific use case
2. **Optimize Performance**: Fine-tune for your hardware and requirements
3. **Add Features**: Implement additional endpoints and functionality
4. **Monitor**: Set up comprehensive monitoring and alerting
5. **Scale**: Deploy to production with proper scaling and load balancing

For more advanced deployment options, refer to the MAX documentation and cloud provider guides. 