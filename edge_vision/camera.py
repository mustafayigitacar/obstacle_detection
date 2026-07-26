"""Kamera yakalama. Canlı kamerada yalnızca EN SON kare saklanır (lag olmasın)."""

from __future__ import annotations

import os
import sys
import threading
from typing import Optional, Tuple, Union

import cv2
import numpy as np


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

    def start(self) -> "ThreadedCamera":
        if not self._sync:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def _loop(self) -> None:
        while not self._stopped:
            ok, frame = self.cap.read()
            if not ok:
                continue
            with self._lock:
                self._frame = frame
                self._frame_id += 1

    def read(self) -> Tuple[int, Optional[np.ndarray]]:
        if self._sync:
            ok, frame = self.cap.read()
            if not ok:
                return self._frame_id, None
            self._frame_id += 1
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
