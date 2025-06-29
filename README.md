# YOLOv10 Model Architecture for MAX

A clean, minimal implementation of YOLOv10 model architecture for MAX (Modular AI Execution) pipelines.

## Overview

This repository contains a custom YOLOv10 model architecture implementation that follows MAX's standard architecture structure. The model is designed for object detection and can be integrated into MAX pipelines for high-performance inference.

## Architecture Structure

Following MAX's standard architecture pattern:

```
yolov10_model/
├── __init__.py          # Makes architecture discoverable
├── arch.py              # Registers model with MAX
├── model.py             # Core model implementation
├── model_config.py      # Configuration parsing
└── weight_adapters.py   # Weight format conversion
```

## Files Description

### Core Architecture Files

- **`__init__.py`**: Package initialization and architecture discovery
- **`arch.py`**: Model registration with MAX, specifying supported encodings and capabilities
- **`model.py`**: Core YOLOv10 model implementation using MAX graphs
- **`model_config.py`**: Configuration parsing and validation
- **`weight_adapters.py`**: Converts model weights from formats like PyTorch, ONNX, or SafeTensors

### Model Features

- **CSPDarknet Backbone**: Efficient feature extraction with Cross Stage Partial connections
- **PANet Neck**: Path Aggregation Network for multi-scale feature fusion
- **Detection Heads**: Multi-scale object detection with anchor-based predictions
- **MAX Graph Integration**: Hardware-optimized computation graphs
- **Flexible Input Sizes**: Supports various input resolutions (default: 640x640)
- **Multi-class Support**: Configurable number of classes

## Quick Start

### Prerequisites

This project uses Pixi for dependency management:

```bash
# Install dependencies
pixi install
```

### Using the Architecture

```python
from yolov10_model import YOLOv10Model
from yolov10_model.model_config import YOLOv10Config

# Create configuration
config = YOLOv10Config(
    input_size=(640, 640),
    num_classes=80,
    anchors=[[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
)

# Initialize model
model = YOLOv10Model(config)

# Load weights (if available)
# model.load_weights("path/to/weights.pt")

# Run inference
# predictions = model.predict(input_tensor)
```

## Model Architecture Details

### CSPDarknet Backbone
- **Stages**: 5 stages with increasing channel dimensions (64, 128, 256, 512, 1024)
- **Features**: Cross Stage Partial connections for efficient gradient flow
- **Output**: Multi-scale feature maps for different detection scales

### PANet Neck
- **Top-down Path**: High-level semantic information flows down
- **Bottom-up Path**: Low-level spatial information flows up
- **Lateral Connections**: Direct information flow between scales
- **Output**: Enhanced multi-scale feature representation

### Detection Heads
- **Multi-scale**: 5 detection heads for different object sizes
- **Anchors**: Pre-defined anchor boxes for each scale
- **Output**: Bounding boxes, confidence scores, and class predictions

## Configuration

The model supports various configuration options:

```python
config = YOLOv10Config(
    input_size=(640, 640),        # Input image size
    num_classes=80,               # Number of object classes
    anchors=[[...], [...], [...]], # Anchor boxes for each scale
    backbone_depth=1.0,           # Backbone depth multiplier
    backbone_width=1.0,           # Backbone width multiplier
    neck_depth=1.0,               # Neck depth multiplier
    neck_width=1.0                # Neck width multiplier
)
```

## MAX Integration

This architecture is designed to work seamlessly with MAX:

- **Graph Compilation**: High-performance graph optimization
- **Hardware Support**: CPU and GPU acceleration
- **Memory Efficiency**: Optimized memory layout and operations
- **Scalability**: Support for batch processing and real-time inference

## Development

### Adding Custom Operations

To add custom operations to the model:

1. Define the operation in `model.py`
2. Register it in `arch.py` if needed
3. Update `weight_adapters.py` for any new weight formats

### Extending the Architecture

The modular design allows easy extension:

- **New Backbones**: Implement in `model.py` and register in `arch.py`
- **New Neck Networks**: Add to the neck module in `model.py`
- **Custom Heads**: Extend the detection head module

## License

This project follows the same license as the original YOLOv10 implementation.

## Contributing

When contributing to this architecture:

1. Follow MAX's architecture patterns
2. Maintain compatibility with existing weight formats
3. Add appropriate tests for new features
4. Update documentation for any API changes

---

**Note**: This is a clean, architecture-only implementation. For training, deployment, or inference servers, refer to the MAX documentation or create separate implementations as needed. 