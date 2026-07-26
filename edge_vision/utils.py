"""Üstel hareketli ortalama ve FPS sayacı."""

from __future__ import annotations

import time
from typing import Optional


class EMA:
    def __init__(self, alpha: float):
        self.alpha = float(alpha)
        self.value: Optional[float] = None

    def update(self, x: float) -> float:
        if self.value is None:
            self.value = float(x)
        else:
            self.value += self.alpha * (float(x) - self.value)
        return self.value


class FpsMeter:
    def __init__(self, alpha: float = 0.1):
        self._ema = EMA(alpha)
        self._last: Optional[float] = None

    def tick(self) -> float:
        now = time.perf_counter()
        if self._last is not None:
            dt = now - self._last
            if dt > 0:
                self._ema.update(1.0 / dt)
        self._last = now
        return self._ema.value or 0.0
