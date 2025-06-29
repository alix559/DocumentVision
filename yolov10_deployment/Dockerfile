# YOLOv10 MAX Container
# Based on MAX documentation patterns

FROM modular/max:latest

# Set environment variables
ENV MAX_DEVICE=auto
ENV MAX_PRECISION=fp16
ENV MODEL_NAME=yolov10-vision
ENV MODEL_VERSION=1.0.0

# Copy model files
COPY model/ /app/model/
COPY model_config.json /app/config.json

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start server
CMD ["max", "serve", "--model-path=/app/model", "--port=8000", "--host=0.0.0.0"]
