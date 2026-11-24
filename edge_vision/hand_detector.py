"""El tanima — MediaPipe HandLandmarker. Eli "el" olarak etiketler.

COCO'da "el" sinifi yoktur; model eli "insan" sayar. Acik sozluklu YOLOE ile de
denendi ve tutmadi (503 karede "hand" guveni 0.00). MediaPipe ele ozel egitilmis
21 eklem noktasi verir.

Uretilen "el" tespiti, ayni bolgeyi kaplayan YOLO "insan" kutusunun YERINE gecer
(bkz. VisionPipeline._merge).

MediaPipe istege baglidir: paket ya da model yoksa katman sessizce kapanir.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from .config import HandConfig
from .log import get_logger
from .object_detector import Detection

HAND_CLASS_ID = -1          # COCO sinifi degil; izleyicide ayri kimlik olarak yeter

log = get_logger(__name__)


class HandDetector:
    """MediaPipe el tanima, ayri arka plan is parcaciginda (diger aglar gibi)."""

    def __init__(self, cfg: HandConfig):
        import mediapipe as mp
        from mediapipe.tasks import python as mpp
        from mediapipe.tasks.python import vision

        self._mp = mp
        opts = vision.HandLandmarkerOptions(
            base_options=mpp.BaseOptions(model_asset_path=cfg.model),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=int(cfg.num_hands),
            min_hand_detection_confidence=float(cfg.conf_thres),
            min_tracking_confidence=0.5,
        )
        self._lm = vision.HandLandmarker.create_from_options(opts)
        self.pad = float(cfg.box_pad)

        self._latest_frame: Optional[np.ndarray] = None
        self._dets: List[Detection] = []
        self._lock = threading.Lock()
        self._frame_lock = threading.Lock()
        self._stopped = False
        self._thread: Optional[threading.Thread] = None
        self._ts = 0
        self._infer_ms = 0.0
        self._update_rate = 0.0
        self._last_pub = 0.0
        self._version = 0
        self._errors = 0

    @staticmethod
    def create(cfg: HandConfig) -> Optional["HandDetector"]:
        """Kurulamazsa None doner — el katmani opsiyoneldir, sistemi durdurmaz."""
        if not cfg.enabled:
            return None
        if not Path(cfg.model).exists():
            log.warning("el: model yok (%s), el tanima kapali", cfg.model)
            return None
        try:
            return HandDetector(cfg)
        except ImportError:
            log.warning("el: mediapipe kurulu degil (pip install mediapipe), "
                        "el tanima kapali")
        except Exception as exc:                            # noqa: BLE001
            log.exception("el: baslatilamadi: %s", exc)
        return None

    # --- arka plan is parcacigi ---

    def start(self) -> "HandDetector":
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def submit(self, bgr: np.ndarray) -> None:
        with self._frame_lock:
            self._latest_frame = bgr.copy()

    def _loop(self) -> None:
        while not self._stopped:
            with self._frame_lock:
                frame = self._latest_frame
                self._latest_frame = None
            if frame is None:
                time.sleep(0.002)
                continue
            try:
                dets = self._infer(frame)
            except Exception:                               # noqa: BLE001
                # Tek bir hatali kare is parcacigini oldurmesin (harita donardi).
                self._errors += 1
                if self._errors <= 3 or self._errors % 100 == 0:
                    log.exception("el cikarimi basarisiz (#%d)", self._errors)
                time.sleep(0.05)
                continue
            with self._lock:
                self._dets = dets
                self._version += 1
            now = time.perf_counter()
            if self._last_pub:
                dt = now - self._last_pub
                if dt > 0:
                    self._update_rate = 0.8 * self._update_rate + 0.2 * (1.0 / dt)
            self._last_pub = now

    def stop(self) -> None:
        self._stopped = True
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        try:
            self._lm.close()
        except Exception:                                   # noqa: BLE001
            pass

    def _infer(self, frame: np.ndarray) -> List[Detection]:
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        self._ts += 34                       # VIDEO kipi artan zaman damgasi ister
        t0 = time.perf_counter()
        res = self._lm.detect_for_video(mp_img, self._ts)
        self._infer_ms = (time.perf_counter() - t0) * 1000.0

        dets: List[Detection] = []
        for i, lms in enumerate(res.hand_landmarks or []):
            xs = [p.x for p in lms]
            ys = [p.y for p in lms]
            bw = (max(xs) - min(xs)) * w
            bh = (max(ys) - min(ys)) * h
            # Eklem noktalari parmak UCLARINI verir, eti degil: kutu biraz genisletilir.
            x0 = int(np.clip(min(xs) * w - self.pad * bw, 0, w - 1))
            y0 = int(np.clip(min(ys) * h - self.pad * bh, 0, h - 1))
            x1 = int(np.clip(max(xs) * w + self.pad * bw, 0, w - 1))
            y1 = int(np.clip(max(ys) * h + self.pad * bh, 0, h - 1))
            if x1 <= x0 or y1 <= y0:
                continue
            conf = 0.9
            if res.handedness and i < len(res.handedness) and res.handedness[i]:
                conf = float(res.handedness[i][0].score)
            dets.append(Detection(HAND_CLASS_ID, "el", conf, x0, y0, x1, y1))
        return dets

    # --- ana dongunun okudugu arayuz ---

    @property
    def version(self) -> int:
        return self._version

    @property
    def update_rate(self) -> float:
        return self._update_rate

    @property
    def infer_ms(self) -> float:
        return self._infer_ms

    def get_detections(self) -> List[Detection]:
        with self._lock:
            return list(self._dets)
