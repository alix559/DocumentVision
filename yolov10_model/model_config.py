"""
YOLOv10 Model Configuration
Handles configuration parsing and validation
"""

from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class YOLOv10ModelConfig:
    """Configuration for YOLOv10 model"""
    
    # Model architecture parameters
    input_size: Tuple[int, int] = (640, 640)
    num_classes: int = 80
    backbone_channels: Tuple[int, ...] = (32, 64, 128, 256, 512, 1024)
    neck_channels: int = 256
    anchors_per_scale: int = 3
    
    # Training parameters
    learning_rate: float = 0.001
    batch_size: int = 16
    num_epochs: int = 100
    
    # Inference parameters
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.4
    max_detections: int = 100
    
    # Device parameters
    device: str = "cpu"
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "YOLOv10ModelConfig":
        """Create configuration from dictionary"""
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "input_size": self.input_size,
            "num_classes": self.num_classes,
            "backbone_channels": self.backbone_channels,
            "neck_channels": self.neck_channels,
            "anchors_per_scale": self.anchors_per_scale,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "confidence_threshold": self.confidence_threshold,
            "nms_threshold": self.nms_threshold,
            "max_detections": self.max_detections,
            "device": self.device,
        }
    
    def validate(self) -> bool:
        """Validate configuration parameters"""
        if self.input_size[0] <= 0 or self.input_size[1] <= 0:
            raise ValueError("Input size must be positive")
        
        if self.num_classes <= 0:
            raise ValueError("Number of classes must be positive")
        
        if self.anchors_per_scale <= 0:
            raise ValueError("Anchors per scale must be positive")
        
        if self.confidence_threshold < 0 or self.confidence_threshold > 1:
            raise ValueError("Confidence threshold must be between 0 and 1")
        
        if self.nms_threshold < 0 or self.nms_threshold > 1:
            raise ValueError("NMS threshold must be between 0 and 1")
        
        return True


def create_default_config() -> YOLOv10ModelConfig:
    """Create default YOLOv10 configuration"""
    return YOLOv10ModelConfig()


def create_custom_config(
    input_size: Tuple[int, int] = (640, 640),
    num_classes: int = 80,
    device: str = "cpu"
) -> YOLOv10ModelConfig:
    """Create custom YOLOv10 configuration"""
    return YOLOv10ModelConfig(
        input_size=input_size,
        num_classes=num_classes,
        device=device
    ) 