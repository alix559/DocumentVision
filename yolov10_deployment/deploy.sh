#!/bin/bash
# YOLOv10 Deployment Script
# Following MAX documentation patterns

echo "🚀 Deploying YOLOv10 Vision Model"

# Set environment variables
export MAX_DEVICE="auto"
export MAX_PRECISION="fp16"
export MODEL_NAME="yolov10-vision"
export MODEL_VERSION="1.0.0"

# Check if MAX is available
if ! command -v max &> /dev/null; then
    echo "❌ MAX CLI not found. Please install MAX first."
    exit 1
fi

# Start MAX serve with custom model
echo "📡 Starting MAX serve..."
max serve \
    --model-path="./model" \
    --port=8000 \
    --host=0.0.0.0 \
    --max-concurrent-requests=10

echo "✅ YOLOv10 deployment completed!"
