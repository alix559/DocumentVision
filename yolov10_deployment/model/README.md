# YOLOv10 Model Directory

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
