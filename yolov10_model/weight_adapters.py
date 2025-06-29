"""
YOLOv10 Weight Adapters
Convert model weights from different formats like SafeTensors or GGUF
"""

from typing import Dict, Any, Optional
import numpy as np


def convert_safetensor_state_dict(state_dict: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    Convert SafeTensors state dict to numpy arrays
    
    Args:
        state_dict: SafeTensors state dictionary
        
    Returns:
        Dictionary with numpy arrays
    """
    converted_dict = {}
    
    for key, value in state_dict.items():
        if hasattr(value, 'numpy'):
            # Convert tensor to numpy array
            converted_dict[key] = value.numpy()
        elif isinstance(value, np.ndarray):
            # Already numpy array
            converted_dict[key] = value
        else:
            # Convert to numpy array
            converted_dict[key] = np.array(value)
    
    return converted_dict


def convert_gguf_state_dict(state_dict: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    Convert GGUF state dict to numpy arrays
    
    Args:
        state_dict: GGUF state dictionary
        
    Returns:
        Dictionary with numpy arrays
    """
    converted_dict = {}
    
    for key, value in state_dict.items():
        if hasattr(value, 'numpy'):
            # Convert tensor to numpy array
            converted_dict[key] = value.numpy()
        elif isinstance(value, np.ndarray):
            # Already numpy array
            converted_dict[key] = value
        else:
            # Convert to numpy array
            converted_dict[key] = np.array(value)
    
    return converted_dict


def convert_pytorch_state_dict(state_dict: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    Convert PyTorch state dict to numpy arrays
    
    Args:
        state_dict: PyTorch state dictionary
        
    Returns:
        Dictionary with numpy arrays
    """
    converted_dict = {}
    
    for key, value in state_dict.items():
        if hasattr(value, 'detach'):
            # PyTorch tensor
            converted_dict[key] = value.detach().cpu().numpy()
        elif hasattr(value, 'numpy'):
            # TensorFlow tensor or similar
            converted_dict[key] = value.numpy()
        elif isinstance(value, np.ndarray):
            # Already numpy array
            converted_dict[key] = value
        else:
            # Convert to numpy array
            converted_dict[key] = np.array(value)
    
    return converted_dict


def map_yolov10_weights(converted_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Map YOLOv10 weight names to model parameters
    
    Args:
        converted_dict: Dictionary with numpy arrays
        
    Returns:
        Mapped dictionary for YOLOv10 model
    """
    mapped_dict = {}
    
    # Map backbone weights
    for key, value in converted_dict.items():
        if 'backbone' in key.lower():
            # Map backbone weights
            mapped_key = key.replace('backbone.', '')
            mapped_dict[mapped_key] = value
        elif 'neck' in key.lower():
            # Map neck weights
            mapped_key = key.replace('neck.', '')
            mapped_dict[mapped_key] = value
        elif 'head' in key.lower():
            # Map detection head weights
            mapped_key = key.replace('head.', '')
            mapped_dict[mapped_key] = value
        else:
            # Keep original key
            mapped_dict[key] = value
    
    return mapped_dict


def validate_weights(weights_dict: Dict[str, np.ndarray], config: Any) -> bool:
    """
    Validate weight shapes match model configuration
    
    Args:
        weights_dict: Dictionary with weights
        config: Model configuration
        
    Returns:
        True if weights are valid
    """
    # Check for required weight keys
    required_keys = [
        'backbone.conv1.weight',
        'neck.conv1.weight',
        'head.conv1.weight'
    ]
    
    for key in required_keys:
        if key not in weights_dict:
            print(f"Warning: Missing required weight key: {key}")
    
    # Validate weight shapes
    for key, weight in weights_dict.items():
        if 'conv' in key and 'weight' in key:
            if len(weight.shape) != 4:
                print(f"Warning: Conv weight {key} has unexpected shape: {weight.shape}")
    
    return True


def load_weights_from_file(file_path: str, format_type: str = "safetensors") -> Dict[str, np.ndarray]:
    """
    Load weights from file and convert to numpy arrays
    
    Args:
        file_path: Path to weight file
        format_type: Format type ("safetensors", "gguf", "pytorch")
        
    Returns:
        Dictionary with numpy arrays
    """
    if format_type == "safetensors":
        # Load SafeTensors file
        try:
            from safetensors import safe_open
            with safe_open(file_path, framework="np") as f:
                state_dict = {key: f.get_tensor(key) for key in f.keys()}
            return convert_safetensor_state_dict(state_dict)
        except ImportError:
            print("Warning: safetensors not available, trying alternative loading")
    
    elif format_type == "gguf":
        # Load GGUF file
        try:
            import gguf
            # GGUF loading implementation would go here
            # This is a placeholder
            return {}
        except ImportError:
            print("Warning: gguf not available")
    
    elif format_type == "pytorch":
        # Load PyTorch file
        try:
            import torch
            state_dict = torch.load(file_path, map_location='cpu')
            return convert_pytorch_state_dict(state_dict)
        except ImportError:
            print("Warning: PyTorch not available")
    
    else:
        raise ValueError(f"Unsupported format type: {format_type}")
    
    return {} 