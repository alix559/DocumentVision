# YOLOv10 Training Guide

This guide explains how to train the YOLOv10 model using MAX graphs with your own datasets.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Dataset Preparation](#dataset-preparation)
3. [Training Configuration](#training-configuration)
4. [Training Process](#training-process)
5. [Monitoring and Evaluation](#monitoring-and-evaluation)
6. [Model Export](#model-export)
7. [Troubleshooting](#troubleshooting)

## Prerequisites

Before starting training, ensure you have:

- MAX environment set up with pixi
- Sufficient GPU memory (recommended: 8GB+ for batch size 16)
- Dataset in a supported format (COCO, YOLO, or custom)
- Python dependencies installed

```bash
# Activate the environment
pixi shell

# Verify MAX installation
python -c "import max; print(max.__version__)"
```

## Dataset Preparation

### Supported Formats

The training system supports multiple dataset formats:

1. **YOLO Format** (Recommended)
   - Images: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`
   - Labels: `.txt` files with YOLO format annotations
   - Structure: `class_id x_center y_center width height`

2. **COCO Format**
   - Images: Any common format
   - Annotations: JSON file with COCO format
   - Will be converted to YOLO format automatically

3. **Custom Format**
   - Can be extended by modifying the dataset loader

### Dataset Structure

Your dataset should follow this structure:

```
dataset/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── labels/
│   ├── image1.txt
│   ├── image2.txt
│   └── ...
└── dataset.yaml
```

### Preparing Your Dataset

Use the dataset preparation script:

```bash
# Create a sample dataset for testing
python prepare_dataset.py --create-sample

# Prepare your own dataset
python prepare_dataset.py \
    --input /path/to/your/raw/dataset \
    --output ./prepared_dataset \
    --format yolo \
    --split-ratio 0.8 0.1 0.1 \
    --validate
```

For COCO format conversion:

```bash
python prepare_dataset.py \
    --input /path/to/coco/images \
    --output ./prepared_dataset \
    --format yolo \
    --coco-annotations /path/to/annotations.json \
    --validate
```

### Dataset Configuration

Create a `dataset.yaml` file:

```yaml
path: ./prepared_dataset
train: train/images
val: val/images
test: test/images
nc: 80  # number of classes
names: ['person', 'bicycle', 'car', ...]  # class names
```

## Training Configuration

### Default Configuration

The training system uses sensible defaults, but you can customize them:

```json
{
  "input_size": [640, 640],
  "num_classes": 80,
  "backbone_channels": [32, 64, 128, 256, 512, 1024],
  "neck_channels": 256,
  "anchors_per_scale": 3,
  "learning_rate": 0.001,
  "batch_size": 16,
  "num_epochs": 100,
  "save_interval": 10,
  "lambda_coord": 5.0,
  "lambda_noobj": 0.5,
  "lambda_class": 1.0,
  "data_path": "dataset",
  "config_path": "dataset.yaml",
  "augment": true
}
```

### Creating Custom Configuration

```bash
# Create default configuration
python train_yolov10.py --create-config --config my_config.json

# Edit the configuration file
nano my_config.json
```

### Key Configuration Parameters

| Parameter | Description | Default | Recommended Range |
|-----------|-------------|---------|-------------------|
| `input_size` | Input image dimensions | [640, 640] | [416, 416] to [1024, 1024] |
| `num_classes` | Number of object classes | 80 | 1 to 1000+ |
| `learning_rate` | Learning rate | 0.001 | 0.0001 to 0.01 |
| `batch_size` | Batch size | 16 | 8 to 64 (based on GPU memory) |
| `num_epochs` | Training epochs | 100 | 50 to 500+ |
| `lambda_coord` | Coordinate loss weight | 5.0 | 1.0 to 10.0 |
| `lambda_noobj` | No-object confidence weight | 0.5 | 0.1 to 1.0 |

## Training Process

### Starting Training

```bash
# Basic training with default settings
python train_yolov10.py \
    --data-path ./prepared_dataset \
    --config train_config.json \
    --epochs 100 \
    --batch-size 16 \
    --learning-rate 0.001
```

### Training with Custom Settings

```bash
# Custom training configuration
python train_yolov10.py \
    --data-path ./my_dataset \
    --config custom_config.json \
    --epochs 200 \
    --batch-size 32 \
    --learning-rate 0.0005
```

### Training Output

The training script will output:

```
INFO: Loading dataset...
INFO: Loaded 1000 images and 1000 labels
INFO: YOLOv10 Training Configuration:
INFO:   input_size: [640, 640]
INFO:   num_classes: 80
INFO:   learning_rate: 0.001
INFO:   batch_size: 16
INFO:   num_epochs: 100
INFO: Starting YOLOv10 training...
INFO: Epoch 1/100
INFO: Train Loss: 15.2341
INFO: Val Loss: 14.9876
INFO: Saved best model
INFO: Epoch 2/100
INFO: Train Loss: 12.4567
INFO: Val Loss: 12.1234
...
```

### Checkpoints and Model Saving

The training system automatically saves:

- **Best Model**: `checkpoints/best_model.json` (lowest validation loss)
- **Regular Checkpoints**: `checkpoints/checkpoint_epoch_X.json` (every `save_interval` epochs)
- **Training Logs**: Console output with loss metrics

## Monitoring and Evaluation

### Loss Components

The training tracks four loss components:

1. **Total Loss**: Combined loss from all components
2. **Coordinate Loss**: Bounding box regression loss
3. **Confidence Loss**: Object confidence prediction loss
4. **Classification Loss**: Class prediction loss

### Expected Loss Progression

- **Epochs 1-10**: High loss (15-20), rapid improvement
- **Epochs 10-50**: Steady decrease (10-5)
- **Epochs 50-100**: Slower improvement (5-2)
- **Epochs 100+**: Fine-tuning (2-1)

### Monitoring Training

Watch for these indicators:

- **Loss not decreasing**: Learning rate too high/low
- **Overfitting**: Validation loss increases while training loss decreases
- **Underfitting**: Both losses remain high
- **Convergence**: Loss plateaus with minimal improvement

## Model Export

### Exporting Trained Model

After training, export your model for inference:

```python
from yolov10_model.model import YOLOv10Model
from yolov10_model.model_config import YOLOv10Config

# Load trained model
config = YOLOv10Config.from_checkpoint('checkpoints/best_model.json')
model = YOLOv10Model(config)

# Export for inference
model.export('trained_yolov10_model')
```

### Serving Trained Model

```bash
# Serve the trained model
python serve_yolov10.py \
    --model-path trained_yolov10_model \
    --port 8000
```

## Advanced Training Features

### Data Augmentation

The training system includes data augmentation:

- Random horizontal flip
- Random brightness/contrast adjustment
- Random scaling and cropping
- Color jittering

### Learning Rate Scheduling

Implement learning rate scheduling:

```python
# In your training configuration
{
  "learning_rate": 0.001,
  "lr_schedule": {
    "type": "step",
    "step_size": 30,
    "gamma": 0.1
  }
}
```

### Multi-GPU Training

For multi-GPU training:

```bash
# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Training will automatically use all available GPUs
python train_yolov10.py --batch-size 64
```

## Troubleshooting

### Common Issues

1. **Out of Memory (OOM)**
   ```
   Solution: Reduce batch size or input image size
   ```

2. **Loss Not Converging**
   ```
   Solution: Check learning rate, data quality, and model configuration
   ```

3. **Slow Training**
   ```
   Solution: Use GPU, increase batch size, optimize data loading
   ```

4. **Poor Detection Performance**
   ```
   Solution: Check dataset quality, annotation accuracy, and model architecture
   ```

### Performance Optimization

1. **Data Loading**
   - Use SSD storage for datasets
   - Implement data prefetching
   - Use multiple workers for data loading

2. **Model Optimization**
   - Use mixed precision training
   - Optimize model architecture
   - Use gradient accumulation for large effective batch sizes

3. **Hardware Utilization**
   - Monitor GPU utilization
   - Use appropriate batch sizes
   - Enable CUDA optimizations

### Debugging Training

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Check intermediate outputs:

```python
# In training loop
if batch_idx % 100 == 0:
    print(f"Batch {batch_idx}: predictions shape {predictions.shape}")
    print(f"Batch {batch_idx}: targets shape {targets.shape}")
```

## Example Training Workflow

Here's a complete example workflow:

```bash
# 1. Prepare dataset
python prepare_dataset.py \
    --input /path/to/raw/dataset \
    --output ./prepared_dataset \
    --format yolo \
    --validate

# 2. Create training configuration
python train_yolov10.py --create-config --config train_config.json

# 3. Start training
python train_yolov10.py \
    --data-path ./prepared_dataset \
    --config train_config.json \
    --epochs 100 \
    --batch-size 16

# 4. Monitor training progress
tail -f training.log

# 5. Export trained model
python -c "
from yolov10_model.model import YOLOv10Model
model = YOLOv10Model.from_checkpoint('checkpoints/best_model.json')
model.export('trained_model')
"

# 6. Test the model
python serve_yolov10.py --model-path trained_model --port 8000
```

## Next Steps

After training your YOLOv10 model:

1. **Evaluate Performance**: Test on validation/test sets
2. **Fine-tune**: Adjust hyperparameters based on results
3. **Deploy**: Export and serve the model
4. **Monitor**: Track real-world performance
5. **Iterate**: Collect more data and retrain

For more advanced training techniques, refer to the MAX documentation and YOLOv10 research papers. 