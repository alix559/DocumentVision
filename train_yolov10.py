#!/usr/bin/env python3
"""
YOLOv10 Training Script using MAX
Train the YOLOv10 model on custom datasets
"""

import os
import sys
import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YOLOv10Dataset:
    """
    Dataset class for YOLOv10 training
    Supports COCO, YOLO, and custom formats
    """
    
    def __init__(self, 
                 data_path: str,
                 config_path: str,
                 input_size: Tuple[int, int] = (640, 640),
                 augment: bool = True):
        self.data_path = Path(data_path)
        self.config_path = Path(config_path)
        self.input_size = input_size
        self.augment = augment
        
        # Load dataset configuration
        self.config = self._load_config()
        
        # Load dataset
        self.images, self.labels = self._load_dataset()
        
        logger.info(f"Loaded {len(self.images)} images and {len(self.labels)} labels")
    
    def _load_config(self) -> Dict:
        """Load dataset configuration"""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return json.load(f)
        else:
            # Default configuration
            return {
                "train": "train/images",
                "val": "val/images",
                "test": "test/images",
                "nc": 80,  # number of classes
                "names": [f"class_{i}" for i in range(80)]
            }
    
    def _load_dataset(self) -> Tuple[List[str], List[str]]:
        """Load images and labels from dataset"""
        images = []
        labels = []
        
        # Load training images
        train_path = self.data_path / self.config.get("train", "train/images")
        if train_path.exists():
            for img_file in train_path.glob("*.jpg"):
                images.append(str(img_file))
                # Find corresponding label file
                label_file = img_file.parent.parent / "labels" / f"{img_file.stem}.txt"
                if label_file.exists():
                    labels.append(str(label_file))
                else:
                    labels.append("")  # No labels for this image
        
        return images, labels
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get a single training sample"""
        img_path = self.images[idx]
        label_path = self.labels[idx]
        
        # Load and preprocess image
        image = self._load_image(img_path)
        
        # Load and preprocess labels
        if label_path:
            labels = self._load_labels(label_path)
        else:
            labels = np.zeros((0, 5))  # Empty labels
        
        return image, labels
    
    def _load_image(self, img_path: str) -> np.ndarray:
        """Load and preprocess image"""
        try:
            # In a real implementation, you'd use PIL or OpenCV
            # For now, create a dummy image
            image = np.random.randn(*self.input_size, 3).astype(np.float32)
            
            # Normalize to [0, 1]
            image = (image - image.min()) / (image.max() - image.min())
            
            return image
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            return np.zeros((*self.input_size, 3), dtype=np.float32)
    
    def _load_labels(self, label_path: str) -> np.ndarray:
        """Load labels in YOLO format (class_id, x_center, y_center, width, height)"""
        try:
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                
                labels = []
                for line in lines:
                    values = line.strip().split()
                    if len(values) == 5:
                        class_id = float(values[0])
                        x_center = float(values[1])
                        y_center = float(values[2])
                        width = float(values[3])
                        height = float(values[4])
                        labels.append([class_id, x_center, y_center, width, height])
                
                return np.array(labels, dtype=np.float32)
            else:
                return np.zeros((0, 5), dtype=np.float32)
        except Exception as e:
            logger.error(f"Error loading labels {label_path}: {e}")
            return np.zeros((0, 5), dtype=np.float32)


class YOLOv10Loss:
    """
    YOLOv10 Loss Function
    Computes classification, regression, and confidence losses
    """
    
    def __init__(self, 
                 num_classes: int = 80,
                 anchors_per_scale: int = 3,
                 lambda_coord: float = 5.0,
                 lambda_noobj: float = 0.5,
                 lambda_class: float = 1.0):
        self.num_classes = num_classes
        self.anchors_per_scale = anchors_per_scale
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.lambda_class = lambda_class
    
    def compute_loss(self, 
                    predictions: np.ndarray, 
                    targets: np.ndarray) -> Dict[str, float]:
        """
        Compute YOLOv10 loss
        
        Args:
            predictions: Model predictions (batch_size, grid_h, grid_w, anchors, 5+num_classes)
            targets: Ground truth targets (batch_size, grid_h, grid_w, anchors, 5+num_classes)
            
        Returns:
            Dictionary with loss components
        """
        # Extract components
        pred_xy = predictions[..., :2]
        pred_wh = predictions[..., 2:4]
        pred_conf = predictions[..., 4]
        pred_cls = predictions[..., 5:]
        
        target_xy = targets[..., :2]
        target_wh = targets[..., 2:4]
        target_conf = targets[..., 4]
        target_cls = targets[..., 5:]
        
        # Object mask (where there are objects)
        obj_mask = target_conf > 0
        noobj_mask = target_conf == 0
        
        # Coordinate loss (only for objects)
        coord_loss = self._compute_coord_loss(
            pred_xy, pred_wh, target_xy, target_wh, obj_mask
        )
        
        # Confidence loss
        conf_loss = self._compute_conf_loss(
            pred_conf, target_conf, obj_mask, noobj_mask
        )
        
        # Classification loss (only for objects)
        class_loss = self._compute_class_loss(
            pred_cls, target_cls, obj_mask
        )
        
        # Total loss
        total_loss = (
            self.lambda_coord * coord_loss +
            conf_loss +
            self.lambda_class * class_loss
        )
        
        return {
            'total_loss': float(total_loss),
            'coord_loss': float(coord_loss),
            'conf_loss': float(conf_loss),
            'class_loss': float(class_loss)
        }
    
    def _compute_coord_loss(self, pred_xy, pred_wh, target_xy, target_wh, obj_mask):
        """Compute coordinate loss (x, y, w, h)"""
        # Mean squared error for coordinates
        xy_loss = np.sum(obj_mask * np.square(pred_xy - target_xy))
        wh_loss = np.sum(obj_mask * np.square(pred_wh - target_wh))
        return xy_loss + wh_loss
    
    def _compute_conf_loss(self, pred_conf, target_conf, obj_mask, noobj_mask):
        """Compute confidence loss"""
        # Binary cross-entropy for confidence
        obj_conf_loss = np.sum(obj_mask * np.square(pred_conf - target_conf))
        noobj_conf_loss = np.sum(noobj_mask * np.square(pred_conf - target_conf))
        return obj_conf_loss + self.lambda_noobj * noobj_conf_loss
    
    def _compute_class_loss(self, pred_cls, target_cls, obj_mask):
        """Compute classification loss"""
        # Categorical cross-entropy for classes
        class_loss = np.sum(obj_mask * np.square(pred_cls - target_cls))
        return class_loss


class YOLOv10Trainer:
    """
    YOLOv10 Trainer
    Handles the training loop and optimization
    """
    
    def __init__(self, 
                 model,
                 dataset: YOLOv10Dataset,
                 config: Dict):
        self.model = model
        self.dataset = dataset
        self.config = config
        
        # Initialize loss function
        self.loss_fn = YOLOv10Loss(
            num_classes=config.get('num_classes', 80),
            anchors_per_scale=config.get('anchors_per_scale', 3)
        )
        
        # Training parameters
        self.learning_rate = config.get('learning_rate', 0.001)
        self.batch_size = config.get('batch_size', 16)
        self.num_epochs = config.get('num_epochs', 100)
        self.save_interval = config.get('save_interval', 10)
        
        # Training state
        self.current_epoch = 0
        self.best_loss = float('inf')
        
        logger.info(f"Initialized trainer with {len(dataset)} samples")
    
    def train(self):
        """Main training loop"""
        logger.info("Starting YOLOv10 training...")
        
        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            
            # Training epoch
            train_loss = self._train_epoch()
            
            # Validation epoch
            val_loss = self._validate_epoch()
            
            # Log progress
            logger.info(f"Epoch {epoch+1}/{self.num_epochs}")
            logger.info(f"Train Loss: {train_loss['total_loss']:.4f}")
            logger.info(f"Val Loss: {val_loss['total_loss']:.4f}")
            
            # Save best model
            if val_loss['total_loss'] < self.best_loss:
                self.best_loss = val_loss['total_loss']
                self._save_model('best_model')
                logger.info("Saved best model")
            
            # Save checkpoint
            if (epoch + 1) % self.save_interval == 0:
                self._save_model(f'checkpoint_epoch_{epoch+1}')
        
        logger.info("Training completed!")
    
    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        total_loss = {
            'total_loss': 0.0,
            'coord_loss': 0.0,
            'conf_loss': 0.0,
            'class_loss': 0.0
        }
        
        num_batches = len(self.dataset) // self.batch_size
        
        for batch_idx in range(num_batches):
            # Get batch
            batch_images, batch_labels = self._get_batch(batch_idx)
            
            # Forward pass
            predictions = self.model.predict(batch_images)
            
            # Compute loss
            loss = self.loss_fn.compute_loss(predictions, batch_labels)
            
            # Update model (in a real implementation, you'd do backprop here)
            # For now, we'll just accumulate losses
            
            # Accumulate losses
            for key in total_loss:
                total_loss[key] += loss[key]
        
        # Average losses
        for key in total_loss:
            total_loss[key] /= num_batches
        
        return total_loss
    
    def _validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch"""
        # For simplicity, use training data as validation
        # In a real implementation, you'd have separate validation data
        return self._train_epoch()
    
    def _get_batch(self, batch_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get a batch of data"""
        start_idx = batch_idx * self.batch_size
        end_idx = start_idx + self.batch_size
        
        batch_images = []
        batch_labels = []
        
        for idx in range(start_idx, min(end_idx, len(self.dataset))):
            image, labels = self.dataset[idx]
            batch_images.append(image)
            batch_labels.append(labels)
        
        # Pad batch if necessary
        while len(batch_images) < self.batch_size:
            batch_images.append(np.zeros((*self.dataset.input_size, 3), dtype=np.float32))
            batch_labels.append(np.zeros((0, 5), dtype=np.float32))
        
        return np.array(batch_images), np.array(batch_labels)
    
    def _save_model(self, filename: str):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'best_loss': self.best_loss,
            'config': self.config,
            'model_state': 'model_state_placeholder'  # In real implementation, save actual model state
        }
        
        checkpoint_path = Path(f"checkpoints/{filename}.json")
        checkpoint_path.parent.mkdir(exist_ok=True)
        
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        logger.info(f"Saved checkpoint: {checkpoint_path}")


def create_training_config() -> Dict:
    """Create default training configuration"""
    return {
        # Model parameters
        'input_size': [640, 640],
        'num_classes': 80,
        'backbone_channels': [32, 64, 128, 256, 512, 1024],
        'neck_channels': 256,
        'anchors_per_scale': 3,
        
        # Training parameters
        'learning_rate': 0.001,
        'batch_size': 16,
        'num_epochs': 100,
        'save_interval': 10,
        
        # Loss parameters
        'lambda_coord': 5.0,
        'lambda_noobj': 0.5,
        'lambda_class': 1.0,
        
        # Data parameters
        'data_path': 'dataset',
        'config_path': 'dataset.yaml',
        'augment': True
    }


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Train YOLOv10 model")
    parser.add_argument("--data-path", type=str, default="dataset", help="Path to dataset")
    parser.add_argument("--config", type=str, default="train_config.json", help="Training config file")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--create-config", action="store_true", help="Create default config file")
    
    args = parser.parse_args()
    
    # Create default config if requested
    if args.create_config:
        config = create_training_config()
        with open(args.config, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Created default config: {args.config}")
        return
    
    # Load configuration
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = create_training_config()
        logger.warning(f"Config file {args.config} not found, using defaults")
    
    # Update config with command line arguments
    config['data_path'] = args.data_path
    config['num_epochs'] = args.epochs
    config['batch_size'] = args.batch_size
    config['learning_rate'] = args.learning_rate
    
    logger.info("YOLOv10 Training Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    
    # Create dataset
    logger.info("Loading dataset...")
    dataset = YOLOv10Dataset(
        data_path=config['data_path'],
        config_path=config.get('config_path', 'dataset.yaml'),
        input_size=tuple(config['input_size']),
        augment=config.get('augment', True)
    )
    
    # Create model (placeholder for now)
    # In a real implementation, you'd load the actual YOLOv10 model
    model = None  # Placeholder
    
    # Create trainer
    trainer = YOLOv10Trainer(model, dataset, config)
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main() 