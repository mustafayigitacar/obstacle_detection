"""Kamera yakalama. Canlı kamerada yalnızca EN SON kare saklanır (lag olmasın)."""

from __future__ import annotations

import os
import sys
import threading
import time
from typing import Optional, Tuple, Union

import cv2
import numpy as np

from .log import get_logger

log = get_logger(__name__)


def _open_capture(source: Union[int, str]) -> cv2.VideoCapture:
    # Windows'ta MSMF arka ucu kamerayı açarken 10+ sn takılabiliyor; DSHOW ~0.5 sn.
    if isinstance(source, int) and sys.platform == "win32":
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        if cap.isOpened():
            return cap
        cap.release()
    return cv2.VideoCapture(source)


class ThreadedCamera:
    def __init__(self, source: Union[int, str], width: int, height: int, fps: int):
        # Yerel video dosyası senkron okunur (kare atlanmaz, sonu = akışın sonu).
        # URL/RTSP akışı ise canlı kamera gibi davranır.
        self._sync = isinstance(source, str) and os.path.isfile(source)
        self.cap = _open_capture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Kamera/video açılamadı: {source!r}")
        if not self._sync:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._frame_id = 0
        self._stopped = False
        self._thread: Optional[threading.Thread] = None
        self._fails = 0
        self._last_ok = time.perf_counter()

    @property
    def is_file(self) -> bool:
        """Kaynak sonlu bir video dosyası mı (None kare = akışın sonu)."""
        return self._sync

    @property
    def age_s(self) -> float:
        """Son başarılı karenin yaşı (sn): canlı kamerada kopma tespiti için."""
        return time.perf_counter() - self._last_ok

    def start(self) -> "ThreadedCamera":
        if not self._sync:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def _loop(self) -> None:
        while not self._stopped:
            ok, frame = self.cap.read()
            if not ok:
                # Kamera koptuysa read() anında döner: beklemeden döngü
                # bir çekirdeği %100 yakıyordu.
                self._fails += 1
                if self._fails in (1, 30) or self._fails % 300 == 0:
                    log.warning("kamera: kare okunamadi (%d ardisik)", self._fails)
                time.sleep(0.01 if self._fails < 30 else 0.1)
                continue
            if self._fails:
                log.info("kamera: akis geri geldi (%d basarisiz okumadan sonra)",
                         self._fails)
                self._fails = 0
            self._last_ok = time.perf_counter()
            with self._lock:
                self._frame = frame
                self._frame_id += 1

    def read(self) -> Tuple[int, Optional[np.ndarray]]:
        if self._sync:
            ok, frame = self.cap.read()
            if not ok:
                return self._frame_id, None
            self._frame_id += 1
            self._last_ok = time.perf_counter()
            return self._frame_id, frame
        with self._lock:
            if self._frame is None:
                return 0, None
            return self._frame_id, self._frame.copy()

    def stop(self) -> None:
        self._stopped = True
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.cap.release()

    def __enter__(self) -> "ThreadedCamera":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()
