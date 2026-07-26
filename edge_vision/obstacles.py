"""Metrik derinlik haritasından engel çıkarımı — sözde-LiDAR sütun taraması."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .config import AvoidanceConfig, ProcessingConfig
from .intrinsics import Intrinsics


@dataclass(frozen=True)
class Obstacle:
    distance_m: float
    forward_m: float
    lateral_m: float
    bearing_deg: float
    width_m: float
    zone: str
    px_u0: int
    px_u1: int

    def to_dict(self) -> dict:
        return {
            "distance_m": round(self.distance_m, 3),
            "forward_m": round(self.forward_m, 3),
            "lateral_m": round(self.lateral_m, 3),
            "bearing_deg": round(self.bearing_deg, 1),
            "width_m": round(self.width_m, 3),
            "zone": self.zone,
            "px": [self.px_u0, self.px_u1],
        }


@dataclass
class ObstacleReport:
    frame_id: int
    timestamp: float = field(default_factory=time.time)
    obstacles: List[Obstacle] = field(default_factory=list)
    nearest: Optional[Obstacle] = None
    scan_bearing_deg: Optional[np.ndarray] = None
    scan_z: Optional[np.ndarray] = None
    scan_u: Optional[np.ndarray] = None

    def to_dict(self) -> dict:
        return {
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "obstacles": [o.to_dict() for o in self.obstacles],
            "nearest": self.nearest.to_dict() if self.nearest else None,
        }


class ObstacleScanner:
    def __init__(self, proc_cfg: ProcessingConfig, avoid_cfg: AvoidanceConfig,
                 intr: Intrinsics):
        self.stride = max(1, int(proc_cfg.column_stride))
        self.max_range = float(avoid_cfg.max_range_m)
        self.center_zone = float(avoid_cfg.center_zone_deg)
        self.intr = intr

    def scan(self, depth_m: np.ndarray, roi_top: int,
             ignore_mask: Optional[np.ndarray] = None):
        """Sütun başına en yakın yüzey → (u, x, z, bearing_deg); yoksa z = inf.

        ignore_mask True olan pikseller (zemin) taramaya girmez. Zemin
        ayıklanmazsa yer her sütunda en yakın yüzey olur ve tüm sütunlar tek
        sahte engelde birleşir.
        """
        h, w = depth_m.shape
        region = depth_m[roi_top:, ::self.stride]
        u = np.arange(0, w, self.stride, dtype=np.intp)

        region = np.where(region > 1e-3, region, np.inf)
        if ignore_mask is not None:
            region = np.where(ignore_mask[roi_top:, ::self.stride], np.inf, region)

        # En yakın 5 pikselin ortalaması + en az 3 destek: tek gürültülü piksel
        # sahte yakın engel üretmesin (gürültü uzakta belirginleşir).
        region_sorted = np.sort(region, axis=0)
        near_k = region_sorted[:min(5, region_sorted.shape[0])]
        finite_k = np.isfinite(near_k)
        support = finite_k.sum(0)
        sums = np.where(finite_k, near_k, 0.0).sum(0)
        z = np.where(support >= 3, sums / np.maximum(support, 1), np.inf).astype(np.float32)

        z = np.where(z <= self.max_range, z, np.inf).astype(np.float32)
        ray = (u - self.intr.cx) / self.intr.fx
        # Alt-küme ile çarp: tüm dizide z*ray, merkez sütunda inf*0=NaN üretirdi.
        finite = np.isfinite(z)
        x = np.zeros_like(z)
        x[finite] = z[finite] * ray[finite]
        bearing = np.degrees(np.arctan(ray)).astype(np.float32)
        return u, x, z, bearing

    def cluster(self, u, x, z, bearing) -> List[Obstacle]:
        valid = np.isfinite(z)
        obstacles: List[Obstacle] = []
        start = None
        for i in range(len(z) + 1):
            is_valid = i < len(z) and valid[i]
            if is_valid and start is None:
                start = i
                continue
            if is_valid and start is not None:
                gap = max(0.3, 0.15 * z[i - 1])
                if abs(z[i] - z[i - 1]) > gap:
                    obstacles.extend(self._make(u, x, z, start, i))
                    start = i
                continue
            if start is not None:
                obstacles.extend(self._make(u, x, z, start, i))
                start = None
        return obstacles

    def _make(self, u, x, z, i0, i1) -> List[Obstacle]:
        if i1 - i0 < 3:
            return []
        zs, xs = z[i0:i1], x[i0:i1]
        forward = float(np.percentile(zs, 20))
        lateral = float(np.mean(xs))
        dist = math.hypot(lateral, forward)
        brg = math.degrees(math.atan2(lateral, forward))
        zone = ("center" if abs(brg) <= self.center_zone
                else "left" if brg < 0 else "right")
        return [Obstacle(
            distance_m=dist, forward_m=forward, lateral_m=lateral,
            bearing_deg=brg, width_m=float(abs(xs[-1] - xs[0])), zone=zone,
            px_u0=int(u[i0]), px_u1=int(u[i1 - 1]),
        )]
