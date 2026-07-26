"""Zamansal yumuşatma: hızlı-yaklaşma / yavaş-uzaklaşma histerezisi."""

from __future__ import annotations

from typing import Optional

import numpy as np


class ScanSmoother:
    def __init__(self, persist_frames: int = 4):
        self.persist_frames = max(1, int(persist_frames))
        self._current: Optional[np.ndarray] = None
        self._rising_count: Optional[np.ndarray] = None

    def smooth(self, z: np.ndarray):
        if self._current is None or len(self._current) != len(z):
            self._current = z.copy()
            self._rising_count = np.zeros(len(z), dtype=np.int32)
            return self._current.copy(), np.zeros(len(z), dtype=bool)

        was_finite = np.isfinite(self._current)
        closer_or_new = (~was_finite) | (z < self._current)
        unchanged = was_finite & (z == self._current)
        farther = was_finite & (z > self._current)

        self._rising_count = np.where(farther, self._rising_count + 1, 0)
        confirmed_farther = farther & (self._rising_count >= self.persist_frames)

        accept = closer_or_new | unchanged | confirmed_farther
        held = ~accept
        self._current = np.where(accept, z, self._current).astype(np.float32)
        return self._current.copy(), held
