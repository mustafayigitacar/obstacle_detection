"""Metrik derinlik ağı — sistemin TEK mesafe kaynağı.

İki motor desteklenir, `depth.backend` ile seçilir:
    dav2   — Depth Anything V2 (kare girdi, ImageNet normalizasyonu)
    yolo26 — YOLO26 derinlik (4:3 girdi, yalnızca /255)

Model AYRI BİR ARKA PLAN İŞ PARÇACIĞINDA koşar; ana döngü onun son çıktısını
bloke olmadan okur. 30 FPS'in anahtarı budur.
"""

from __future__ import annotations

import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np

from . import depth_color
from .config import DepthConfig
from .log import get_logger
from .utils import require_model

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

log = get_logger(__name__)


class MetricDepthNet:
    def __init__(self, cfg: DepthConfig):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = cfg.threads
        avail = set(ort.get_available_providers())
        providers = [p for p in cfg.providers if p in avail] or ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(require_model(cfg.model), so,
                                            providers=providers)
        self.active_provider = self.session.get_providers()[0]
        log.info("derinlik modeli: %s  saglayici=%s", cfg.model, self.active_provider)
        if providers[0] != cfg.providers[0]:
            log.warning("derinlik: istenen saglayici %s yok, %s kullaniliyor",
                        cfg.providers[0], providers[0])
        self.input_name = self.session.get_inputs()[0].name
        self.backend = str(getattr(cfg, "backend", "dav2")).lower()
        # YOLO26 sabit girdiyle aktarılır (boyut oturumdan); DA V2'de 14'ün katı.
        shape = self.session.get_inputs()[0].shape
        if self.backend == "yolo26" and isinstance(shape[-1], int):
            self.in_hw = (int(shape[-2]), int(shape[-1]))
            self.size = int(shape[-1])
        else:
            self.size = int(round(cfg.input_size / 14.0) * 14)
            self.in_hw = (self.size, self.size)

        self.calib_scale = float(cfg.calib_scale)
        self.calib_offset = float(cfg.calib_offset)
        self.min_valid_m = float(cfg.min_valid_m)
        self.median_ksize = int(cfg.median_ksize) if int(cfg.median_ksize) in (3, 5) else 0
        self.viz_near_m = float(cfg.viz_near_m)
        self.viz_far_m = float(cfg.viz_far_m)
        self.viz_red_max_m = float(cfg.viz_red_max_m)
        depth_color.set_red_max(self.viz_red_max_m)

        self._table: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._out_hw: Optional[Tuple[int, int]] = None
        self._latest_frame: Optional[np.ndarray] = None
        self._depth_m: Optional[np.ndarray] = None
        self._depth_raw: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._frame_lock = threading.Lock()
        self._stopped = False
        self._thread: Optional[threading.Thread] = None
        self._infer_ms = 0.0
        self._update_rate = 0.0
        self._last_pub = 0.0
        self._version = 0

        # Kilitler kurulduktan SONRA çağrılmalı: _lock kullanır.
        self.set_calibration_table(cfg.calib_points)

    # --- arka plan iş parçacığı ---

    def start(self, out_hw: Tuple[int, int]) -> "MetricDepthNet":
        self._out_hw = out_hw
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def submit(self, bgr_proc: np.ndarray) -> None:
        with self._frame_lock:
            self._latest_frame = bgr_proc.copy()

    def _loop(self) -> None:
        while not self._stopped:
            with self._frame_lock:
                frame = None if self._latest_frame is None else self._latest_frame
                self._latest_frame = None
            if frame is None:
                time.sleep(0.002)
                continue
            try:
                depth_raw, depth = self._infer(frame)
            except Exception:                               # noqa: BLE001
                # Tek bir hatali kare is parcacigini oldurmesin (harita donardi).
                self._errors = getattr(self, "_errors", 0) + 1
                if self._errors <= 3 or self._errors % 100 == 0:
                    log.exception("derinlik cikarimi basarisiz (#%d)", self._errors)
                time.sleep(0.05)
                continue
            with self._lock:
                self._depth_raw = depth_raw
                self._depth_m = depth
                self._version += 1
            now = time.perf_counter()
            if self._last_pub:
                dt = now - self._last_pub
                if dt > 0:
                    self._update_rate = 0.8 * self._update_rate + 0.2 * (1.0 / dt)
            self._last_pub = now

    def _infer(self, bgr_proc: np.ndarray):
        ph, pw = self._out_hw
        ih, iw = self.in_hw
        inp = cv2.resize(bgr_proc, (iw, ih), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        if self.backend == "yolo26":
            # Yalnızca /255 bekler; ImageNet normalizasyonu değerleri kaydırır.
            x = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])
        else:
            x = ((rgb - _IMAGENET_MEAN) / _IMAGENET_STD).transpose(2, 0, 1)[None]
        x = x.astype(np.float32)
        t0 = time.perf_counter()
        out = self.session.run(None, {self.input_name: x})[0]
        out = np.asarray(out).squeeze().astype(np.float32)
        self._infer_ms = (time.perf_counter() - t0) * 1000.0

        depth_raw = cv2.resize(out, (pw, ph), interpolation=cv2.INTER_LINEAR)
        cal = self._calibrate(depth_raw)
        if self.median_ksize:
            valid = cal > 1e-3
            cal = np.where(valid, cv2.medianBlur(cal, self.median_ksize), cal)
        return depth_raw, cal

    def stop(self) -> None:
        self._stopped = True
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    # --- kalibrasyon ---

    def set_calibration_table(self, points) -> None:
        """[[ham, gerçek], ...] tablosu (>=2 nokta) afin dönüşümün yerine geçer."""
        pts = [(float(a), float(b)) for a, b in (points or [])]
        pts.sort(key=lambda p: p[0])
        xs, ys = [], []
        for a, b in pts:
            if xs and abs(a - xs[-1]) < 1e-6:
                ys[-1] = b
            else:
                xs.append(a); ys.append(b)
        with self._lock:
            self._table = ((np.asarray(xs, dtype=np.float32),
                            np.asarray(ys, dtype=np.float32))
                           if len(xs) >= 2 else None)

    @property
    def table_size(self) -> int:
        return 0 if self._table is None else int(self._table[0].size)

    def get_calibration_table(self) -> list:
        with self._lock:
            if self._table is None:
                return []
            xs, ys = self._table
            return list(zip(xs.tolist(), ys.tolist()))

    def _calibrate(self, depth_raw: np.ndarray) -> np.ndarray:
        valid = depth_raw > 1e-3
        tbl = self._table
        if tbl is not None:
            xs, ys = tbl
            cal = np.interp(depth_raw, xs, ys).astype(np.float32)
            # Ekstrapolasyon tablonun GENEL eğimiyle yapılır, son parçanınkiyle
            # değil: yakın iki noktanın gürültülü eğimi tüm menzili çökertirdi.
            span_x = float(xs[-1] - xs[0])
            slope = float((ys[-1] - ys[0]) / span_x) if span_x > 1e-6 else 1.0
            lo = depth_raw < xs[0]
            cal[lo] = ys[0] + (depth_raw[lo] - xs[0]) * slope
            hi = depth_raw > xs[-1]
            cal[hi] = ys[-1] + (depth_raw[hi] - xs[-1]) * slope
        else:
            cal = self.calib_scale * depth_raw + self.calib_offset
        np.clip(cal, self.min_valid_m, None, out=cal)
        cal[~valid] = 0.0
        return cal.astype(np.float32)

    # --- ana döngünün okuduğu (bloke olmayan) arayüz ---

    @property
    def version(self) -> int:
        return self._version

    @property
    def update_rate(self) -> float:
        return self._update_rate

    @property
    def infer_ms(self) -> float:
        return self._infer_ms

    def get_depth(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._depth_m is None else self._depth_m.copy()

    def get_raw_depth(self) -> Optional[np.ndarray]:
        """Kalibrasyon ölçümleri bunu kullanır: ham değer mutlak referanstır."""
        with self._lock:
            return None if self._depth_raw is None else self._depth_raw.copy()

    def color_range(self, depth_m: Optional[np.ndarray] = None) -> Tuple[float, float]:
        return self.viz_near_m, self.viz_far_m

    def visualize(self, depth_m: Optional[np.ndarray],
                  rng: Optional[Tuple[float, float]] = None) -> Optional[np.ndarray]:
        if depth_m is None:
            return None
        near, far = rng if rng is not None else self.color_range(depth_m)
        invalid = depth_m <= 1e-3
        vis = depth_color.colorize(depth_m, near, far)
        vis[invalid] = (45, 45, 45)
        cv2.putText(vis, f"KIRMIZI={depth_color.fmt(near)}-"
                    f"{depth_color.fmt(self.viz_red_max_m)}  mavi={depth_color.fmt(far)}"
                    f"   {self._infer_ms:.0f}ms {self._update_rate:.0f}fps",
                    (4, 14), cv2.FONT_HERSHEY_PLAIN, 0.9, (255, 255, 255), 1)

        valid = depth_m[depth_m > 1e-3]
        if valid.size:
            if self._table is not None:
                ys = self._table[1]
                kind = f"tablo({self.table_size}) {ys[0]:.2f}-{ys[-1]:.2f}m"
            else:
                kind = "afin"
            cv2.putText(vis, f"harita en yakin: {float(valid.min()):.2f} m  kalib: {kind}",
                        (4, 28), cv2.FONT_HERSHEY_PLAIN, 0.9, (255, 255, 255), 1)
        return vis
