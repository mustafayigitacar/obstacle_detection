"""Küçük yardımcılar: model dosyası kontrolü, EMA, FPS sayacı."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional


def require_model(path: str) -> str:
    """Model dosyası yoksa ne yapılacağını SÖYLEYEN hata verir.

    Model dosyaları depoya girmez, dolayısıyla taze bir klonda yokturlar;
    ONNX Runtime'in ham `NO_SUCHFILE` hatası ise ne yapılacağını anlatmaz.
    """
    if Path(path).exists():
        return path
    raise FileNotFoundError(
        f"Model dosyasi yok: {path}\n"
        f"Model dosyalari depoya girmez (.gitignore). Uretmek icin:\n"
        f"    pip install ultralytics\n"
        f"    python setup_models.py\n"
        f"Ayrintilar: README 'Kurulum' bolumu."
    )


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
