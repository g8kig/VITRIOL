"""VITRIOL Client Types"""

from dataclasses import dataclass
from typing import Optional, List


@dataclass
class GPUInfo:
    gpu_present: bool
    gpu_name: str
    vram_total: int
    vram_used: int
    vram_free: int


@dataclass
class ModelInfo:
    name: str
    path: str
    size_bytes: int
    layers: int
    quantization: str


@dataclass
class LayerInfo:
    layer_id: int
    size_bytes: int
    loaded: bool
    vram_addr: Optional[int] = None


@dataclass
class VitriolStatus:
    gpu: GPUInfo
    model_loaded: bool
    model: Optional[ModelInfo]
    layers_loaded: List[LayerInfo]
    safety_level: int
    dma_engine: str  # "ready", "not_initialized", "error"


@dataclass
class InferenceResult:
    output: str
    tokens_generated: int
    inference_time_ms: int
