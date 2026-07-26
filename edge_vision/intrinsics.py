"""Kamera iç parametreleri — yanal konum (X) ve yön açısı için."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


def intrinsics_from_hfov(width: int, height: int, hfov_deg: float) -> Intrinsics:
    fx = width / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))
    return Intrinsics(fx=fx, fy=fx, cx=width / 2.0, cy=height / 2.0)
