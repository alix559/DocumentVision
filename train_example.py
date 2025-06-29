#!/usr/bin/env python3
"""
YOLOv10 Training Example
Demonstrates how to train the YOLOv10 model with a sample dataset
"""

import os
import sys
import json
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_sample_dataset():
    """Create a minimal sample dataset for demonstration"""
    logger.info("Creating sample dataset...")
    
    # Create dataset directory structure
    dataset_path = Path("sample_dataset")
    dataset_path.mkdir(exist_ok=True)
    
    # Create train/val/test splits
    for split in ['train', 'val', 'test']:
        (dataset_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (dataset_path / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Create sample images and labels
    for split in ['train', 'val', 'test']:
        num_samples = 10 if split == 'train' else 3
        
        for i in range(num_samples):
            # Create dummy image file
            img_path = dataset_path / split / 'images' / f"{split}_{i:03d}.jpg"
            img_path.write_text(f"dummy image {split}_{i}")  # Placeholder
            
            # Create sample labels (YOLO format)
            label_path = dataset_path / split / 'labels' / f"{split}_{i:03d}.txt"
            
            # Create 1-3 random bounding boxes per image
            import random
            num_boxes = random.randint(1, 3)
            label_content = ""
            
            for _ in range(num_boxes):
                class_id = random.randint(0, 79)  # COCO classes
                x_center = random.uniform(0.1, 0.9)
                y_center = random.uniform(0.1, 0.9)
                width = random.uniform(0.05, 0.3)
                height = random.uniform(0.05, 0.3)
                
                label_content += f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
            
            label_path.write_text(label_content)
    
    # Create dataset configuration
    config = {
        'path': str(dataset_path.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': 80,
        'names': [f'class_{i}' for i in range(80)]
    }
    
    config_path = dataset_path / 'dataset.yaml'
    import yaml
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info(f"Created sample dataset at {dataset_path}")
    logger.info(f"  - Train: 10 images")
    logger.info(f"  - Val: 3 images")
    logger.info(f"  - Test: 3 images")
    
    return dataset_path


def create_training_config():
    """Create a minimal training configuration"""
    logger.info("Creating training configuration...")
    
    config = {
        "model": {
            "input_size": [640, 640],
            "num_classes": 80,
            "backbone_channels": [32, 64, 128, 256, 512, 1024],
            "neck_channels": 256,
            "anchors_per_scale": 3
        },
        "training": {
            "learning_rate": 0.001,
            "batch_size": 4,  # Small batch size for demo
            "num_epochs": 5,   # Few epochs for demo
            "save_interval": 2
        },
        "loss": {
            "lambda_coord": 5.0,
            "lambda_noobj": 0.5,
            "lambda_class": 1.0
        },
        "data": {
            "data_path": "sample_dataset",
            "config_path": "sample_dataset/dataset.yaml",
            "augment": True
        }
    }
    
    config_path = Path("demo_train_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Created training config: {config_path}")
    return config_path


def run_training_demo():
    """Run a complete training demonstration"""
    logger.info("Starting YOLOv10 Training Demo")
    logger.info("=" * 50)
    
    # Step 1: Create sample dataset
    dataset_path = create_sample_dataset()
    
    # Step 2: Create training configuration
    config_path = create_training_config()
    
    # Step 3: Run training (simulated)
    logger.info("Starting training simulation...")
    logger.info("Note: This is a demonstration - actual training requires MAX integration")
    
    # Simulate training progress
    import time
    for epoch in range(1, 6):
        logger.info(f"Epoch {epoch}/5")
        
        # Simulate training loss
        train_loss = 15.0 - (epoch * 2.5) + (epoch * 0.1)  # Decreasing loss with some noise
        val_loss = train_loss + 0.5  # Slightly higher validation loss
        
        logger.info(f"  Train Loss: {train_loss:.4f}")
        logger.info(f"  Val Loss: {val_loss:.4f}")
        
        if epoch % 2 == 0:
            logger.info(f"  Saved checkpoint: checkpoints/checkpoint_epoch_{epoch}.json")
        
        time.sleep(0.5)  # Simulate training time
    
    logger.info("Training completed!")
    
    # Step 4: Show next steps
    logger.info("\nNext Steps:")
    logger.info("1. Replace sample dataset with your real dataset")
    logger.info("2. Adjust training configuration for your needs")
    logger.info("3. Run actual training with: python train_yolov10.py --data-path sample_dataset --config demo_train_config.json")
    logger.info("4. Monitor training progress and adjust hyperparameters")
    logger.info("5. Export trained model for inference")


def show_dataset_info():
    """Show information about the created dataset"""
    logger.info("\nDataset Information:")
    logger.info("=" * 30)
    
    dataset_path = Path("sample_dataset")
    if not dataset_path.exists():
        logger.error("Dataset not found. Run the demo first.")
        return
    
    # Count files in each split
    for split in ['train', 'val', 'test']:
        images_dir = dataset_path / split / 'images'
        labels_dir = dataset_path / split / 'labels'
        
        if images_dir.exists():
            num_images = len(list(images_dir.glob('*')))
            num_labels = len(list(labels_dir.glob('*.txt')))
            logger.info(f"{split.capitalize()}: {num_images} images, {num_labels} labels")
    
    # Show dataset configuration
    config_path = dataset_path / 'dataset.yaml'
    if config_path.exists():
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Classes: {config['nc']}")
        logger.info(f"Class names: {config['names'][:5]}...")  # Show first 5


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="YOLOv10 Training Demo")
    parser.add_argument("--create-dataset", action="store_true", help="Create sample dataset only")
    parser.add_argument("--create-config", action="store_true", help="Create training config only")
    parser.add_argument("--run-demo", action="store_true", help="Run complete training demo")
    parser.add_argument("--show-info", action="store_true", help="Show dataset information")
    
    args = parser.parse_args()
    
    if args.create_dataset:
        create_sample_dataset()
    elif args.create_config:
        create_training_config()
    elif args.show_info:
        show_dataset_info()
    elif args.run_demo:
        run_training_demo()
    else:
        # Default: run complete demo
        run_training_demo()


if __name__ == "__main__":
    main() 