from .arch import yolov10_arch

# MAX looks for this variable when loading custom architectures
ARCHITECTURES = [yolov10_arch]

__all__ = ["yolov10_arch", "ARCHITECTURES"] 