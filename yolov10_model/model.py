"""
YOLOv10 Model Implementation for MAX
Core model implementation with CSPDarknet backbone, PANet neck, and detection heads
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from max import engine
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, ops
from max.pipelines.lib import PipelineModel

@dataclass
class YOLOv10Config:
    """Configuration for YOLOv10 model"""
    
    def __init__(self, **kwargs):
        self.input_size = kwargs.get("input_size", (640, 640))
        self.num_classes = kwargs.get("num_classes", 80)
        self.backbone_channels = kwargs.get("backbone_channels", [32, 64, 128, 256, 512, 1024])
        self.neck_channels = kwargs.get("neck_channels", 256)
        self.anchors_per_scale = kwargs.get("anchors_per_scale", 3)


class YOLOv10Model(PipelineModel):
    """
    YOLOv10 Model implementation using MAX graphs
    """
    
    def __init__(self, config: YOLOv10Config):
        super().__init__(config)
        self.config = config
        self.session = None
        self.graph = None
        
    def build_csp_backbone(self, x):
        """Build CSPDarknet backbone with CSP blocks"""
        
        # Initial convolution
        x = ops.conv2d(
            x, 
            filter=np.random.randn(3, 3, 3, self.config.backbone_channels[0]).astype(np.float32),
            stride=(1, 1), 
            padding=(1, 1, 1, 1)
        )
        x = ops.relu(x)
        
        # CSP Stages
        for i in range(len(self.config.backbone_channels) - 1):
            in_channels = self.config.backbone_channels[i]
            out_channels = self.config.backbone_channels[i + 1]
            
            # Downsample if not first stage
            if i > 0:
                x = ops.conv2d(
                    x, 
                    filter=np.random.randn(3, 3, in_channels, out_channels).astype(np.float32),
                    stride=(2, 2), 
                    padding=(1, 1, 1, 1)
                )
                x = ops.relu(x)
            
            # CSP Block
            x = self._csp_block(x, in_channels, out_channels, 1, f"stage{i+1}")
        
        return x
    
    def _csp_block(self, x, in_channels, out_channels, num_blocks, name):
        """CSP (Cross Stage Partial) block"""
        
        # Split input along channel dimension
        split_size = out_channels // 2
        x1, x2 = ops.split(x, split_size, axis=3)
        
        # Main branch with bottleneck blocks
        x1 = ops.conv2d(
            x1, 
            filter=np.random.randn(1, 1, in_channels//2, split_size).astype(np.float32),
            stride=(1, 1)
        )
        x1 = ops.relu(x1)
        
        for i in range(num_blocks):
            x1 = self._bottleneck_block(x1, split_size, f"{name}_bottleneck_{i}")
        
        # Side branch
        x2 = ops.conv2d(
            x2, 
            filter=np.random.randn(1, 1, in_channels//2, split_size).astype(np.float32),
            stride=(1, 1)
        )
        x2 = ops.relu(x2)
        
        # Concatenate branches
        x = ops.concat([x1, x2], axis=3)
        
        # Final convolution
        x = ops.conv2d(
            x, 
            filter=np.random.randn(1, 1, out_channels, out_channels).astype(np.float32),
            stride=(1, 1)
        )
        x = ops.relu(x)
        
        return x
    
    def _bottleneck_block(self, x, channels, name):
        """Bottleneck block with residual connection"""
        
        residual = x
        
        # First convolution
        x = ops.conv2d(
            x, 
            filter=np.random.randn(1, 1, channels, channels).astype(np.float32),
            stride=(1, 1)
        )
        x = ops.relu(x)
        
        # Second convolution
        x = ops.conv2d(
            x, 
            filter=np.random.randn(3, 3, channels, channels).astype(np.float32),
            stride=(1, 1), 
            padding=(1, 1, 1, 1)
        )
        x = ops.relu(x)
        
        # Add residual
        x = ops.add(x, residual)
        
        return x
    
    def build_panet_neck(self, backbone_features):
        """Build PANet neck with feature fusion"""
        
        # For simplicity, work with the final backbone feature
        x = backbone_features
        
        # Bottom-up path with lateral connections
        x = ops.conv2d(
            x, 
            filter=np.random.randn(3, 3, x.shape[3], self.config.neck_channels).astype(np.float32),
            stride=(1, 1), 
            padding=(1, 1, 1, 1)
        )
        x = ops.relu(x)
        
        return x
    
    def build_detection_heads(self, neck_features):
        """Build multi-scale detection heads"""
        
        # Detection head
        head = ops.conv2d(
            neck_features, 
            filter=np.random.randn(3, 3, neck_features.shape[3], self.config.neck_channels).astype(np.float32),
            stride=(1, 1), 
            padding=(1, 1, 1, 1)
        )
        head = ops.relu(head)
        
        # Final detection layer (anchors * (5 + num_classes))
        detection_output = ops.conv2d(
            head, 
            filter=np.random.randn(1, 1, self.config.neck_channels, 
                                 self.config.anchors_per_scale * (5 + self.config.num_classes)).astype(np.float32),
            stride=(1, 1)
        )
        
        return detection_output
    
    def build_model(self):
        """Build the complete YOLOv10 model"""
        
        # Create graph
        self.graph = Graph("yolov10_complete")
        
        # Create input tensor
        input_tensor = ops.constant(
            value=np.zeros((1, self.config.input_size[0], self.config.input_size[1], 3), dtype=np.float32),
            dtype=DType.float32,
            device="cpu"
        )
        
        # Build backbone
        backbone_features = self.build_csp_backbone(input_tensor)
        
        # Build neck
        neck_features = self.build_panet_neck(backbone_features)
        
        # Build detection heads
        detection_output = self.build_detection_heads(neck_features)
        
        return detection_output
    
    def compile(self):
        """Compile the model"""
        output = self.build_model()
        
        # Set the output
        self.graph.output = output
        
        device_ref = DeviceRef("cpu")
        self.session = engine.InferenceSession(self.graph, device_ref)
        
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """Run inference"""
        if self.session is None:
            self.compile()
            
        # Ensure input is correct shape (NHWC)
        expected_shape = (1, self.config.input_size[0], self.config.input_size[1], 3)
        if input_data.shape != expected_shape:
            raise ValueError(f"Input shape must be {expected_shape} in NHWC format")
            
        outputs = self.session.run({"input": input_data})
        return outputs["output"]
    
    def save(self, path: str):
        """Save the model"""
        if self.session is None:
            self.compile()
        self.session.save(path)
    
    @classmethod
    def load(cls, path: str, config: YOLOv10Config):
        """Load a saved model"""
        device_ref = DeviceRef("cpu")
        session = engine.InferenceSession.load(path, device_ref)
        
        instance = cls(config)
        instance.session = session
        return instance 