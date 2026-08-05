"""Uçtan uca görü hattı.

    pipe = VisionPipeline(cfg).start()
    report, cmd, dbg = pipe.process(frame_id, bgr_frame)
    pipe.stop()
"""

from __future__ import annotations

from dataclasses import replace
from typing import Optional, Tuple

import cv2
import numpy as np

from .avoidance import AvoidanceCommand, AvoidancePlanner
from .config import AppConfig
from .edge_detector import AutoCannyDetector
from .ground import GroundRemover
from .intrinsics import intrinsics_from_hfov
from .metric_depth import MetricDepthNet
from .hand_detector import HandDetector
from .object_detector import DetectionTracker, YoloDetector, _iou, fuse_distance
from .obstacles import ObstacleReport, ObstacleScanner
from .utils import FpsMeter


class ScaleAnchor:
    """Küresel ölçek kaymasını uzak arka plana göre düzeltir.

    Model, sahneye yakın bir cisim girdiğinde TÜM haritayı yeniden ölçekliyor.
    Ölçüldü: görüntünün hiç değişmeyen yarısındaki mesafe %50 kaydı; çapa bunu
    %15'e indiriyor. Sahnenin uzak kısmı gerçekte sabit kaldığından kaymanın
    göstergesidir — referans/ölçüm oranı haritayı ilk ölçeğine geri getirir.

    Cismin GERÇEK yaklaşması arka planı değiştirmez, o yüzden bastırılmaz.
    """

    def __init__(self, cfg):
        self.enabled = bool(cfg.scale_anchor)
        self.pct = int(cfg.ref_pct)
        self.warmup = int(cfg.warmup_frames)
        self.lo = float(cfg.anchor_min)
        self.hi = float(cfg.anchor_max)
        self.ema = float(cfg.anchor_ema)
        self.ref: Optional[float] = None
        self.k = 1.0
        self._seen = 0

    def measure(self, raw: Optional[np.ndarray]) -> Optional[float]:
        if raw is None:
            return None
        v = raw[raw > 1e-3]
        if v.size <= 500:
            return None
        return float(np.percentile(v, self.pct))

    def update(self, value: Optional[float]) -> float:
        if not self.enabled or not value or value <= 1e-6:
            return self.k
        if self.ref is None:
            # Referans oturana kadar bekle: ilk kareler ağ ısınırken gürültülü.
            self._seen += 1
            if self._seen < self.warmup:
                return 1.0
            self.ref = value
            return 1.0
        target = float(np.clip(self.ref / value, self.lo, self.hi))
        self.k += self.ema * (target - self.k)
        return self.k


class VisionPipeline:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        cam = cfg.camera
        self.pw = int(cfg.processing.proc_width)
        self.ph = int(round(self.pw * cam.capture_height / cam.capture_width / 2) * 2)

        self.intr = intrinsics_from_hfov(self.pw, self.ph, cam.hfov_deg)
        self.detector = AutoCannyDetector(cfg.edge.canny_sigma) if cfg.edge.enabled else None
        self.depth = MetricDepthNet(cfg.depth)
        self.scanner = ObstacleScanner(cfg.processing, cfg.avoidance, self.intr)
        self.planner = AvoidancePlanner(cfg.avoidance)
        self.ground = GroundRemover(self.intr, cfg.ground)
        self.yolo = YoloDetector(cfg.yolo) if cfg.yolo.enabled else None
        self.hand = HandDetector.create(cfg.hand)
        self.tracker = DetectionTracker(cfg.tracking)
        self.anchor = ScaleAnchor(cfg.stabilize)
        self._yolo_version = -1
        self._hand_version = -1

        self.smoother = None
        if cfg.smoothing.enabled:
            from .scan_smoother import ScanSmoother
            self.smoother = ScanSmoother(cfg.smoothing.rise_persist_frames)

        self._fps = FpsMeter()
        self._proc_buf: Optional[np.ndarray] = None
        self._last_depth: Optional[np.ndarray] = None
        self._depth_version = -1
        self._ground_mask: Optional[np.ndarray] = None

    def start(self) -> "VisionPipeline":
        self.depth.start((self.ph, self.pw))
        if self.yolo is not None:
            self.yolo.start()
        if self.hand is not None:
            self.hand.start()
        return self

    def stop(self) -> None:
        self.depth.stop()
        if self.yolo is not None:
            self.yolo.stop()
        if self.hand is not None:
            self.hand.stop()

    def process(self, frame_id: int, frame_bgr: np.ndarray, want_edges: bool = True
                ) -> Tuple[ObstacleReport, AvoidanceCommand, dict]:
        if self._proc_buf is None:
            self._proc_buf = np.empty((self.ph, self.pw, 3), dtype=np.uint8)
        proc = cv2.resize(frame_bgr, (self.pw, self.ph),
                          dst=self._proc_buf, interpolation=cv2.INTER_AREA)

        self.depth.submit(proc)
        if self.yolo is not None:
            self.yolo.submit(proc)
        if self.hand is not None:
            self.hand.submit(proc)

        edges = None
        if want_edges and self.detector is not None:
            edges = self.detector.detect(cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY))

        # Sürüm sayacı: ana döngü kamera hızında döner, derinlik ağı daha yavaş
        # yayınlar. Aynı harita için kopyalama ve düzlem uydurma tekrarlanmaz.
        depth_new = False
        ver = self.depth.version
        if ver != self._depth_version:
            d = self.depth.get_depth()
            if d is not None:
                self._last_depth = d
                self._depth_version = ver
                depth_new = True
        depth_m = self._last_depth

        roi_top = int(self.ph * self.cfg.processing.roi_top_ratio)

        if depth_m is None:
            # Derinlik gelmeden de etiketler görünür; mesafe harita gelince dolar.
            dets, _ = self._detect(None, None)
            u = np.arange(0, self.pw, self.cfg.processing.column_stride, dtype=np.intp)
            z = np.full(len(u), np.inf, dtype=np.float32)
            report = ObstacleReport(frame_id=frame_id, scan_u=u, scan_z=z,
                                    scan_bearing_deg=np.zeros(len(u), np.float32),
                                    detections=dets)
            command = self.planner.plan(report)
            return report, command, self._debug(proc, edges, depth_m, roi_top)

        if depth_new or self._ground_mask is None:
            self.ground.update(depth_m, roi_top)
            self._ground_mask = self.ground.ground_mask(depth_m, roi_top)
        ground_mask = self._ground_mask

        # Ölçek çapası: harita global olarak kaydıysa geri getir. HAM değerden
        # ölçülür (kalibrasyondan bağımsız referans). Maske ölçeklemeye bağışık
        # olduğu için maskeden sonra uygulanır, yeniden hesap gerekmez.
        if depth_new:
            self.anchor.update(self.anchor.measure(self.depth.get_raw_depth()))
        if abs(self.anchor.k - 1.0) > 1e-3:
            depth_m = (depth_m * self.anchor.k).astype(np.float32)

        detections, nearest_named = self._detect(depth_m, ground_mask)

        u, x, z, bearing = self.scanner.scan(depth_m, roi_top, ground_mask)

        if self.smoother is not None:
            z, held = self.smoother.smooth(z)
            if held.any():
                x[held] = z[held] * np.tan(np.radians(bearing[held]))

        obstacles = self.scanner.cluster(u, x, z, bearing)

        # Ön-plan/arka-plan ayrımı: yalnızca en yakın N küme engel sayılır.
        cap_n = self.cfg.avoidance.max_tracked_objects
        obstacles.sort(key=lambda o: o.distance_m)
        foreground = obstacles[:cap_n]
        if len(obstacles) > cap_n:
            fg_mask = np.zeros(len(z), dtype=bool)
            for ob in foreground:
                fg_mask |= (u >= ob.px_u0) & (u <= ob.px_u1)
            z = np.where(fg_mask, z, np.inf).astype(np.float32)
            x = np.where(fg_mask, x, 0.0).astype(np.float32)
        obstacles = foreground

        nearest = min(obstacles, key=lambda o: o.distance_m) if obstacles else None
        report = ObstacleReport(
            frame_id=frame_id, obstacles=obstacles, nearest=nearest,
            scan_bearing_deg=bearing, scan_z=z, scan_u=u,
            detections=detections, nearest_named=nearest_named,
        )
        command = self.planner.plan(report)
        return report, command, self._debug(proc, edges, depth_m, roi_top)

    def _merge(self):
        """YOLO + el tespitlerini birleştirir; el, kendi bölgesindeki "insan"ın
        YERİNE geçer — kameraya uzatılan elde o kutu zaten elin kendisidir."""
        yolo = self.yolo.get_detections() if self.yolo is not None else []
        hands = self.hand.get_detections() if self.hand is not None else []
        if hands:
            yolo = [d for d in yolo
                    if not (d.class_id == 0 and any(_iou(d, h) > 0.5 for h in hands))]
        return yolo + hands

    def _detect(self, depth_m, ground_mask):
        """YOLO tespitlerini izler, etiketi oylar, derinlikle füzyona sokar.

        İzleyici yalnızca yeni çıkarım geldiğinde beslenir; mesafe ise HER kare
        o anki haritadan yeniden hesaplanır ve iz başına EMA ile dengelenir.
        """
        if self.yolo is None and self.hand is None:
            return [], None

        new = False
        if self.yolo is not None and self.yolo.version != self._yolo_version:
            self._yolo_version = self.yolo.version
            new = True
        if self.hand is not None and self.hand.version != self._hand_version:
            self._hand_version = self.hand.version
            new = True
        if new:
            self.tracker.update(self._merge())

        dets = self.tracker.current() if self.tracker.enabled else self._merge()
        if depth_m is None:
            return dets, None

        out = []
        for d in dets:
            dist = fuse_distance(d, depth_m, ground_mask)
            if self.tracker.enabled:
                dist = self.tracker.set_distance(d.tid, dist)
            out.append(replace(d, distance_m=dist))
        named = [d for d in out if np.isfinite(d.distance_m)]
        nearest_named = min(named, key=lambda d: d.distance_m) if named else None
        return out, nearest_named

    def _debug(self, proc, edges, depth_m, roi_top) -> dict:
        return {
            "proc": proc,
            "edges": edges,
            "roi_top": roi_top,
            "depth": self.depth,
            "depth_m": depth_m,
            "ground_locked": self.ground.locked,
            "ground_mask": self._ground_mask,
            "scale_k": self.anchor.k,
            "yolo": self.yolo,
            "hand": self.hand,
            "fps": self._fps.tick(),
        }
