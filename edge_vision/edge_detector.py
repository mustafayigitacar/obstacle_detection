"""Otomatik eşikli Canny kenar tespiti. Yalnızca görselleştirme ürünü."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


class AutoCannyDetector:
    def __init__(self, sigma: float = 0.33):
        self.sigma = float(sigma)
        self._blur: Optional[np.ndarray] = None

    def detect(self, gray: np.ndarray) -> np.ndarray:
        if self._blur is None or self._blur.shape != gray.shape:
            self._blur = np.empty_like(gray)
        cv2.GaussianBlur(gray, (5, 5), 0, dst=self._blur)
        m = float(np.median(self._blur[::4, ::4]))
        lo = int(max(0.0, (1.0 - self.sigma) * m))
        hi = int(min(255.0, (1.0 + self.sigma) * m))
        return cv2.Canny(self._blur, lo, hi, L2gradient=False)
