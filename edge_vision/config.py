"""YAML tabanlı yapılandırma. Bilinmeyen anahtarlar sessizce atılır."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Union

import yaml


@dataclass
class CameraConfig:
    source: Union[int, str] = 0
    capture_width: int = 640
    capture_height: int = 480
    fps: int = 30
    hfov_deg: float = 62.2


@dataclass
class ProcessingConfig:
    proc_width: int = 320
    roi_top_ratio: float = 0.30
    column_stride: int = 2


@dataclass
class EdgeConfig:
    enabled: bool = True
    canny_sigma: float = 0.33


@dataclass
class DepthConfig:
    model: str = "models/depth_anything_v2_metric_indoor.onnx"
    input_size: int = 210
    providers: List[str] = field(
        default_factory=lambda: ["DmlExecutionProvider", "CPUExecutionProvider"])
    threads: int = 4
    median_ksize: int = 3

    # --- Kalibrasyon: ham model metresi -> gerçek metre ---
    calib_scale: float = 0.514
    calib_offset: float = -0.147
    min_valid_m: float = 0.05
    calib_points: List[List[float]] = field(default_factory=list)

    # --- Görsel renk skalası (sayıyı etkilemez) ---
    viz_red_max_m: float = 0.40
    viz_near_m: float = 0.05
    viz_far_m: float = 3.0


@dataclass
class AvoidanceConfig:
    max_range_m: float = 5.0
    warn_distance_m: float = 1.2
    brake_distance_m: float = 0.6
    stop_distance_m: float = 0.3
    center_zone_deg: float = 12.0
    min_gap_deg: float = 15.0
    max_tracked_objects: int = 5


@dataclass
class SmoothingConfig:
    enabled: bool = True
    rise_persist_frames: int = 4


@dataclass
class GroundConfig:
    enabled: bool = True
    margin_base_m: float = 0.06
    margin_per_m: float = 0.09
    min_inlier_ratio: float = 0.25
    min_normal_y: float = 0.80
    fit_step: int = 4
    ema: float = 0.25
    max_miss: int = 8


@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    edge: EdgeConfig = field(default_factory=EdgeConfig)
    depth: DepthConfig = field(default_factory=DepthConfig)
    avoidance: AvoidanceConfig = field(default_factory=AvoidanceConfig)
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    ground: GroundConfig = field(default_factory=GroundConfig)


def _build(cls, data: dict):
    known = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in (data or {}).items() if k in known})


def load_config(path: Union[str, Path, None] = None) -> AppConfig:
    if path is None:
        return AppConfig()
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return AppConfig(
        camera=_build(CameraConfig, raw.get("camera")),
        processing=_build(ProcessingConfig, raw.get("processing")),
        edge=_build(EdgeConfig, raw.get("edge")),
        depth=_build(DepthConfig, raw.get("depth")),
        avoidance=_build(AvoidanceConfig, raw.get("avoidance")),
        smoothing=_build(SmoothingConfig, raw.get("smoothing")),
        ground=_build(GroundConfig, raw.get("ground")),
    )
