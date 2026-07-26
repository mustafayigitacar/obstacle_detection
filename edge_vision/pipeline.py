"""Uçtan uca görü hattı.

    pipe = VisionPipeline(cfg).start()
    report, cmd, dbg = pipe.process(frame_id, bgr_frame)
    pipe.stop()
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from .avoidance import AvoidanceCommand, AvoidancePlanner
from .config import AppConfig
from .edge_detector import AutoCannyDetector
from .ground import GroundRemover
from .intrinsics import intrinsics_from_hfov
from .metric_depth import MetricDepthNet
from .obstacles import ObstacleReport, ObstacleScanner
from .utils import FpsMeter


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
        return self

    def stop(self) -> None:
        self.depth.stop()

    def process(self, frame_id: int, frame_bgr: np.ndarray, want_edges: bool = True
                ) -> Tuple[ObstacleReport, AvoidanceCommand, dict]:
        if self._proc_buf is None:
            self._proc_buf = np.empty((self.ph, self.pw, 3), dtype=np.uint8)
        proc = cv2.resize(frame_bgr, (self.pw, self.ph),
                          dst=self._proc_buf, interpolation=cv2.INTER_AREA)

        self.depth.submit(proc)

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
            u = np.arange(0, self.pw, self.cfg.processing.column_stride, dtype=np.intp)
            z = np.full(len(u), np.inf, dtype=np.float32)
            report = ObstacleReport(frame_id=frame_id, scan_u=u, scan_z=z,
                                    scan_bearing_deg=np.zeros(len(u), np.float32))
            command = self.planner.plan(report)
            return report, command, self._debug(proc, edges, depth_m, roi_top)

        if depth_new or self._ground_mask is None:
            self.ground.update(depth_m, roi_top)
            self._ground_mask = self.ground.ground_mask(depth_m, roi_top)
        ground_mask = self._ground_mask

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
        )
        command = self.planner.plan(report)
        return report, command, self._debug(proc, edges, depth_m, roi_top)

    def _debug(self, proc, edges, depth_m, roi_top) -> dict:
        return {
            "proc": proc,
            "edges": edges,
            "roi_top": roi_top,
            "depth": self.depth,
            "depth_m": depth_m,
            "ground_locked": self.ground.locked,
            "fps": self._fps.tick(),
        }
