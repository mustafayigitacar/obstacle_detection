"""Kaçınma çıktı katmanı: "en geniş açık boşluğu izle" (follow-the-gap)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .config import AvoidanceConfig
from .obstacles import ObstacleReport


@dataclass(frozen=True)
class AvoidanceCommand:
    action: str            # CLEAR | CAUTION | FOLLOW_GAP | STOP
    steer_deg: float
    speed_scale: float
    nearest_m: float
    reason: str

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "steer_deg": round(self.steer_deg, 1),
            "speed_scale": round(self.speed_scale, 2),
            "nearest_m": round(self.nearest_m, 3) if np.isfinite(self.nearest_m) else None,
            "reason": self.reason,
        }


class AvoidancePlanner:
    def __init__(self, cfg: AvoidanceConfig):
        self.cfg = cfg

    def plan(self, report: ObstacleReport) -> AvoidanceCommand:
        cfg = self.cfg
        if report.scan_z is None or len(report.scan_z) == 0:
            return AvoidanceCommand("STOP", 0.0, 0.0, float("inf"), "tarama verisi yok")

        z, brg = report.scan_z, report.scan_bearing_deg
        center = np.abs(brg) <= cfg.center_zone_deg
        nearest = float(z[center].min()) if center.any() else float("inf")

        if nearest <= cfg.stop_distance_m:
            return AvoidanceCommand("STOP", 0.0, 0.0, nearest,
                                    f"engel {nearest:.2f} m — acil durus")

        gap = self._widest_gap(brg, z)

        if nearest <= cfg.brake_distance_m:
            if gap is None:
                return AvoidanceCommand("STOP", 0.0, 0.0, nearest,
                                        "yakin engel var, yönlenilebilir bosluk yok")
            return AvoidanceCommand("FOLLOW_GAP", gap, 0.3, nearest,
                                    f"fren bölgesi; bosluga yönel ({gap:+.0f}°)")

        if nearest <= cfg.warn_distance_m:
            span = cfg.warn_distance_m - cfg.brake_distance_m
            scale = 0.3 + 0.7 * (nearest - cfg.brake_distance_m) / max(span, 1e-6)
            steer = gap if gap is not None else 0.0
            return AvoidanceCommand("CAUTION", steer, float(np.clip(scale, 0.3, 1.0)),
                                    nearest, "uyari bölgesi; yavasla")

        return AvoidanceCommand("CLEAR", 0.0, 1.0, nearest, "yol açik")

    def _widest_gap(self, bearing: np.ndarray, z: np.ndarray) -> Optional[float]:
        free = z > self.cfg.warn_distance_m
        best: Optional[Tuple[float, float]] = None
        start = None
        for i in range(len(free) + 1):
            if i < len(free) and free[i]:
                if start is None:
                    start = i
                continue
            if start is not None:
                width = float(bearing[i - 1] - bearing[start])
                if width >= self.cfg.min_gap_deg and (best is None or width > best[0]):
                    best = (width, float((bearing[start] + bearing[i - 1]) / 2.0))
                start = None
        return None if best is None else best[1]
