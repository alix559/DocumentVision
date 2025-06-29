#!/usr/bin/env python3
"""
Dataset Preparation Script for YOLOv10 Training
Prepare and validate datasets in various formats
"""

import os
import sys
import argparse
import json
import yaml
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetPreparator:
    """
    Prepare datasets for YOLOv10 training
    Supports COCO, YOLO, and custom formats
    """
    
    def __init__(self, input_path: str, output_path: str):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        
        # Create output directory
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Preparing dataset from {input_path} to {output_path}")
    
    def prepare_yolo_format(self, split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1)):
        """
        Prepare dataset in YOLO format
        
        Args:
            split_ratio: (train, val, test) split ratios
        """
        logger.info("Preparing YOLO format dataset...")
        
        # Create directory structure
        for split in ['train', 'val', 'test']:
            (self.output_path / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.output_path / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        # Find all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(self.input_path.rglob(f"*{ext}"))
            image_files.extend(self.input_path.rglob(f"*{ext.upper()}"))
        
        logger.info(f"Found {len(image_files)} image files")
        
        # Split dataset
        train_ratio, val_ratio, test_ratio = split_ratio
        total_files = len(image_files)
        
        train_count = int(total_files * train_ratio)
        val_count = int(total_files * val_ratio)
        test_count = total_files - train_count - val_count
        
        # Shuffle and split
        import random
        random.shuffle(image_files)
        
        train_files = image_files[:train_count]
        val_files = image_files[train_count:train_count + val_count]
        test_files = image_files[train_count + val_count:]
        
        # Process each split
        self._process_split(train_files, 'train')
        self._process_split(val_files, 'val')
        self._process_split(test_files, 'test')
        
        # Create dataset configuration
        self._create_dataset_config()
        
        logger.info("YOLO format dataset preparation completed!")
    
    def _process_split(self, image_files: List[Path], split_name: str):
        """Process a single dataset split"""
        logger.info(f"Processing {split_name} split with {len(image_files)} images")
        
        for i, img_path in enumerate(image_files):
            # Copy image
            dst_img_path = self.output_path / split_name / 'images' / f"{split_name}_{i:06d}{img_path.suffix}"
            shutil.copy2(img_path, dst_img_path)
            
            # Look for corresponding label file
            label_path = self._find_label_file(img_path)
            if label_path and label_path.exists():
                dst_label_path = self.output_path / split_name / 'labels' / f"{split_name}_{i:06d}.txt"
                shutil.copy2(label_path, dst_label_path)
            else:
                # Create empty label file
                dst_label_path = self.output_path / split_name / 'labels' / f"{split_name}_{i:06d}.txt"
                dst_label_path.write_text("")
    
    def _find_label_file(self, image_path: Path) -> Optional[Path]:
        """Find corresponding label file for an image"""
        # Common label file locations
        possible_locations = [
            image_path.parent / 'labels' / f"{image_path.stem}.txt",
            image_path.parent.parent / 'labels' / f"{image_path.stem}.txt",
            image_path.parent / f"{image_path.stem}.txt",
            image_path.parent / f"{image_path.stem}.xml",  # XML annotations
            image_path.parent / f"{image_path.stem}.json",  # JSON annotations
        ]
        
        for location in possible_locations:
            if location.exists():
                return location
        
        return None
    
    def _create_dataset_config(self):
        """Create dataset configuration file"""
        config = {
            'path': str(self.output_path),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': 80,  # number of classes
            'names': [f'class_{i}' for i in range(80)]
        }
        
        config_path = self.output_path / 'dataset.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        logger.info(f"Created dataset config: {config_path}")
    
    def convert_coco_to_yolo(self, coco_annotations: str):
        """
        Convert COCO format annotations to YOLO format
        
        Args:
            coco_annotations: Path to COCO annotations file
        """
        logger.info(f"Converting COCO annotations from {coco_annotations}")
        
        # Load COCO annotations
        with open(coco_annotations, 'r') as f:
            coco_data = json.load(f)
        
        # Create image ID to filename mapping
        image_map = {img['id']: img['file_name'] for img in coco_data['images']}
        
        # Group annotations by image
        annotations_by_image = {}
        for ann in coco_data['annotations']:
            image_id = ann['image_id']
            if image_id not in annotations_by_image:
                annotations_by_image[image_id] = []
            annotations_by_image[image_id].append(ann)
        
        # Convert each image's annotations
        for image_id, annotations in annotations_by_image.items():
            filename = image_map[image_id]
            label_filename = Path(filename).stem + '.txt'
            
            # Create label file
            label_path = self.output_path / 'labels' / label_filename
            
            with open(label_path, 'w') as f:
                for ann in annotations:
                    # Convert bbox to YOLO format
                    bbox = ann['bbox']  # [x, y, width, height]
                    category_id = ann['category_id']
                    
                    # Normalize coordinates (assuming image size is known)
                    # In a real implementation, you'd get actual image dimensions
                    img_width = 640  # placeholder
                    img_height = 640  # placeholder
                    
                    x_center = (bbox[0] + bbox[2] / 2) / img_width
                    y_center = (bbox[1] + bbox[3] / 2) / img_height
                    width = bbox[2] / img_width
                    height = bbox[3] / img_height
                    
                    # Write YOLO format: class_id x_center y_center width height
                    f.write(f"{category_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        logger.info("COCO to YOLO conversion completed!")
    
    def validate_dataset(self) -> bool:
        """Validate the prepared dataset"""
        logger.info("Validating dataset...")
        
        issues = []
        
        # Check directory structure
        required_dirs = ['train/images', 'train/labels', 'val/images', 'val/labels']
        for dir_path in required_dirs:
            if not (self.output_path / dir_path).exists():
                issues.append(f"Missing directory: {dir_path}")
        
        # Check image-label pairs
        for split in ['train', 'val']:
            images_dir = self.output_path / split / 'images'
            labels_dir = self.output_path / split / 'labels'
            
            if images_dir.exists() and labels_dir.exists():
                image_files = list(images_dir.glob('*'))
                label_files = list(labels_dir.glob('*.txt'))
                
                logger.info(f"{split}: {len(image_files)} images, {len(label_files)} labels")
                
                # Check for missing labels
                for img_file in image_files:
                    label_file = labels_dir / f"{img_file.stem}.txt"
                    if not label_file.exists():
                        issues.append(f"Missing label for {split}/{img_file.name}")
        
        # Report issues
        if issues:
            logger.error("Dataset validation issues found:")
            for issue in issues:
                logger.error(f"  - {issue}")
            return False
        else:
            logger.info("Dataset validation passed!")
            return True


def create_sample_dataset():
    """Create a sample dataset for testing"""
    logger.info("Creating sample dataset...")
    
    # Create sample directory structure
    sample_path = Path("sample_dataset")
    sample_path.mkdir(exist_ok=True)
    
    # Create sample images (dummy files)
    for i in range(10):
        img_path = sample_path / f"image_{i:03d}.jpg"
        img_path.write_text(f"dummy image {i}")  # Placeholder
    
    # Create sample labels
    labels_path = sample_path / "labels"
    labels_path.mkdir(exist_ok=True)
    
    for i in range(10):
        label_path = labels_path / f"image_{i:03d}.txt"
        # Create sample YOLO format labels
        label_content = f"0 0.5 0.5 0.2 0.3\n"  # class_id, x_center, y_center, width, height
        label_path.write_text(label_content)
    
    logger.info(f"Created sample dataset at {sample_path}")
    return sample_path


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Prepare dataset for YOLOv10 training")
    parser.add_argument("--input", type=str, required=True, help="Input dataset path")
    parser.add_argument("--output", type=str, required=True, help="Output dataset path")
    parser.add_argument("--format", type=str, default="yolo", choices=["yolo", "coco"], help="Output format")
    parser.add_argument("--coco-annotations", type=str, help="COCO annotations file (if converting from COCO)")
    parser.add_argument("--split-ratio", type=float, nargs=3, default=[0.8, 0.1, 0.1], help="Train/val/test split ratios")
    parser.add_argument("--validate", action="store_true", help="Validate dataset after preparation")
    parser.add_argument("--create-sample", action="store_true", help="Create a sample dataset for testing")
    
    args = parser.parse_args()
    
    if args.create_sample:
        sample_path = create_sample_dataset()
        logger.info(f"Sample dataset created at: {sample_path}")
        return
    
    # Create dataset preparator
    preparator = DatasetPreparator(args.input, args.output)
    
    # Prepare dataset
    if args.format == "yolo":
        if args.coco_annotations:
            preparator.convert_coco_to_yolo(args.coco_annotations)
        else:
            preparator.prepare_yolo_format(tuple(args.split_ratio))
    
    # Validate dataset
    if args.validate:
        preparator.validate_dataset()
    
    logger.info("Dataset preparation completed!")


if __name__ == "__main__":
    main() 