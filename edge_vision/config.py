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
    # "dav2" (ImageNet normalizasyonu) | "yolo26" (yalnızca /255).
    # Ham değerleri ortak DEĞİL: backend değişirse kalibrasyon geçersizdir.
    backend: str = "dav2"
    model: str = "models/depth_anything_v2_metric_indoor.onnx"
    input_size: int = 518      # modelin yerleşik çözünürlüğü; düşürmek ayrım gücünü kırar
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
class YoloConfig:
    enabled: bool = True
    model: str = "models/yolo26n-seg.onnx"
    input_size: int = 640      # model sabit boyutluysa oturumdan okunan değer kazanır
    conf_thres: float = 0.35
    iou_thres: float = 0.45
    max_det: int = 20
    # CPU: derinlik ağı DirectML'i tutuyor, ikinci bir DirectML oturumu çakışıyor.
    providers: List[str] = field(default_factory=lambda: ["CPUExecutionProvider"])
    threads: int = 12
    # Açık sözlüklü model (YOLOE) için sınıf adları; sıra modeldekiyle AYNI
    # olmalı. Boşsa COCO listesi kullanılır.
    classes: List[str] = field(default_factory=list)


@dataclass
class StabilizeConfig:
    """Küresel ölçek çapası: model sahne değişince tüm haritayı ölçekliyor."""
    scale_anchor: bool = True
    ref_pct: int = 85          # "uzak arka plan" olarak alınan yüzdelik
    warmup_frames: int = 20    # referans bu kadar kare sonra kilitlenir
    anchor_min: float = 0.30   # düzeltme katsayısı bu aralığın dışına çıkamaz
    anchor_max: float = 3.00
    anchor_ema: float = 0.25   # ani sıçrama yerine yumuşak düzeltme


@dataclass
class HandConfig:
    """El tanima (MediaPipe). COCO'da "el" sinifi yok; bu katman onu ekler."""
    enabled: bool = True
    model: str = "models/hand_landmarker.task"
    num_hands: int = 2
    conf_thres: float = 0.5
    box_pad: float = 0.12      # eklem noktalari eti kapsamaz, kutu genisletilir


@dataclass
class CalibConfig:
    """Kalibrasyon hedef seçimi ve ölçüm hassasiyeti."""
    # Bu orandan küçük kutularda güvenilir medyan alacak piksel kalmıyor.
    min_target_area_frac: float = 0.02
    # Nişan kutusu (manuel kip): COCO'da olmayan cisimler — kapı, duvar, karton
    aim_frac: float = 0.28
    aim_min: float = 0.10
    aim_max: float = 0.90
    aim_step: float = 0.04

    capture_frames: int = 16    # medyanın standart hatası ~ σ/√n
    grow_log_tol: float = 0.30  # bölge büyütme adım eşiği (log = oransal)
    band_tol: float = 0.13      # yedek bant yöntemi için derinlik bandı (±%)
    stability_tol: float = 0.17  # sapma bu oranı aşarsa nokta reddedilir


@dataclass
class TrackConfig:
    """Tespit izleme: etiket titremesini ve mesafe oynamasını keser."""
    enabled: bool = True
    vote_frames: int = 15      # etiket son N karenin çoğunluğu (~0.75 s @20fps)
    min_hits: int = 3          # bu kadar kare görülmeden ekrana gelmez
    max_miss: int = 5          # kısa kayboluşta iz silinmez
    iou_match: float = 0.30    # eşleme eşiği
    switch_ratio: float = 1.5  # etiket değişimi için gereken oy üstünlüğü
    box_ema: float = 0.5       # kutu yumuşatma
    # Mesafe ölü bandı: cisim dururken okuma sabit, hareket edince hızlı takip
    deadband_m: float = 0.06
    alpha_still: float = 0.12
    alpha_move: float = 0.60


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
    yolo: YoloConfig = field(default_factory=YoloConfig)
    tracking: TrackConfig = field(default_factory=TrackConfig)
    stabilize: StabilizeConfig = field(default_factory=StabilizeConfig)
    hand: HandConfig = field(default_factory=HandConfig)
    calib: CalibConfig = field(default_factory=CalibConfig)
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
        yolo=_build(YoloConfig, raw.get("yolo")),
        tracking=_build(TrackConfig, raw.get("tracking")),
        stabilize=_build(StabilizeConfig, raw.get("stabilize")),
        hand=_build(HandConfig, raw.get("hand")),
        calib=_build(CalibConfig, raw.get("calib")),
        avoidance=_build(AvoidanceConfig, raw.get("avoidance")),
        smoothing=_build(SmoothingConfig, raw.get("smoothing")),
        ground=_build(GroundConfig, raw.get("ground")),
    )
