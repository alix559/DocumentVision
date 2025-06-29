from max.graph.weights import WeightsFormat
from max.nn.kv_cache import KVCacheStrategy
from max.pipelines.core import PipelineTask
from max.pipelines.lib import (
    SupportedArchitecture,
    SupportedEncoding,
    TextTokenizer,
)

from . import weight_adapters
from .model import YOLOv10Model

yolov10_arch = SupportedArchitecture(
    name="YOLOv10ForObjectDetection",
    example_repo_ids=[
        "ultralytics/yolov10",  # Example YOLOv10 model repository
        "your-org/yolov10-custom",  # Add your custom model repository
    ],
    default_encoding=SupportedEncoding.bfloat16,
    supported_encodings={
        SupportedEncoding.bfloat16: [KVCacheStrategy.PAGED],
        SupportedEncoding.float16: [KVCacheStrategy.PAGED],
        SupportedEncoding.float32: [KVCacheStrategy.PAGED],
        # YOLOv10 supports various precision formats for object detection
    },
    pipeline_model=YOLOv10Model,
    tokenizer=TextTokenizer,  # For text-based prompts if needed
    default_weights_format=WeightsFormat.safetensors,
    multi_gpu_supported=True,  # YOLOv10 can be distributed across GPUs
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_safetensor_state_dict,
        WeightsFormat.gguf: weight_adapters.convert_gguf_state_dict,
        # Add other weight formats if needed
    },
    task=PipelineTask.OBJECT_DETECTION,  # YOLOv10 is for object detection
) 