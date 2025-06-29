# YOLOv10 Vision API

This document describes the API endpoints for the YOLOv10 vision model deployed with MAX.

## Base URL
```
http://localhost:8000
```

## Endpoints

### 1. Object Detection
**POST** `/v1/vision/detect`

Detect objects in an image.

**Request Body:**
```json
{
  "image": "base64_encoded_image_data",
  "confidence_threshold": 0.25
}
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
**POST** `/v1/chat/completions`

Vision chat endpoint compatible with MAX vision models.

**Request Body:**
```json
{
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
}
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
        "content": "I can see 3 objects in this image:\n1. person (confidence: 0.85)\n2. car (confidence: 0.72)"
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
**GET** `/health`

Check server health.

**Response:**
```json
{
  "status": "healthy",
  "model": "YOLOv10",
  "version": "1.0.0"
}
```

### 4. Model Information
**GET** `/model/info`

Get model information.

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

## Usage Examples

### Python Client
```python
import requests
import base64

# Object detection
with open("image.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode('utf-8')

response = requests.post(
    "http://localhost:8000/v1/vision/detect",
    json={"image": image_data, "confidence_threshold": 0.25}
)

print(response.json())
```

### cURL Examples
```bash
# Health check
curl http://localhost:8000/health

# Model info
curl http://localhost:8000/model/info

# Object detection
curl -X POST http://localhost:8000/v1/vision/detect \
  -H "Content-Type: application/json" \
  -d '{"image": "base64_encoded_image", "confidence_threshold": 0.25}'
```

## Model Details

- **Architecture**: YOLOv10 with CSPDarknet backbone
- **Input Size**: [640, 640]
- **Classes**: 80
- **Device**: auto
- **Precision**: fp16
