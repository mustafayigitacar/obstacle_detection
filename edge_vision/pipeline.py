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
from .log import get_logger
from .object_detector import DetectionTracker, YoloDetector, _iou, fuse_distance
from .obstacles import ObstacleReport, ObstacleScanner
from .stability import SceneMonitor, SceneState
from .utils import FpsMeter

log = get_logger(__name__)


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
        self.rebase_on_cut = bool(getattr(cfg, "rebase_on_cut", True))
        self.ref: Optional[float] = None
        self.k = 1.0
        self._seen = 0

    def rebase(self) -> None:
        """Sahne kesmesi: eski referans artık "uzak arka plan" değil.

        Mevcut düzeltme korunur (harita sıçramasın); yeni sahnenin ölçülen
        değeri, o düzeltmeyle aynı sonucu verecek şekilde referans alınır.
        """
        if not self.enabled or not self.rebase_on_cut:
            return
        self.ref = None
        self._seen = 0
        log.info("olcek capasi: sahne kesmesi, referans yeniden kuruluyor (k=%.2f)", self.k)

    def measure(self, raw: Optional[np.ndarray]) -> Optional[float]:
        if raw is None:
            return None
        v = raw[raw > 1e-3]
        if v.size <= 500:
            return None
        return float(np.percentile(v, self.pct))

    def update(self, value: Optional[float], hold: bool = False) -> float:
        """`hold`: sahne kararsız (ışık/bulanıklık) — ölçüm güvenilmez, k sabit."""
        if not self.enabled or not value or value <= 1e-6 or hold:
            return self.k
        if self.ref is None:
            # Referans oturana kadar bekle: ilk kareler ağ ısınırken gürültülü.
            self._seen += 1
            if self._seen < self.warmup:
                return self.k
            # Yeniden kurulumda mevcut k korunur: ref = ölçüm * k.
            self.ref = value * self.k
            log.info("olcek capasi: referans kilitlendi (p%d=%.3f, k=%.2f)",
                     self.pct, self.ref, self.k)
            return self.k
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
        self.scene = SceneMonitor(cfg.scene)
        self.scene_state = SceneState()
        self._ground_locked = None
        self._depth_med: Optional[float] = None
        self._depth_jump = 0.0
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
        log.info("hat basliyor: islem %dx%d, derinlik=%s (%s), yolo=%s, el=%s",
                 self.pw, self.ph, self.cfg.depth.backend, self.depth.active_provider,
                 "acik" if self.yolo else "kapali", "acik" if self.hand else "kapali")
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

        st = self.scene_state = self.scene.update(proc)
        if st.cut:
            self.anchor.rebase()
        if self.tracker.enabled:
            self.tracker.strict = not st.stable

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
            if self.ground.locked != self._ground_locked:
                if self._ground_locked is not None:
                    log.info("zemin: %s", "kilitlendi" if self.ground.locked else "kilit koptu")
                self._ground_locked = self.ground.locked
        ground_mask = self._ground_mask

        # Ölçek çapası: harita global olarak kaydıysa geri getir. HAM değerden
        # ölçülür (kalibrasyondan bağımsız referans). Maske ölçeklemeye bağışık
        # olduğu için maskeden sonra uygulanır, yeniden hesap gerekmez.
        if depth_new:
            raw = self.depth.get_raw_depth()
            self._track_depth_jump(raw)
            self.anchor.update(self.anchor.measure(raw), hold=not st.stable)
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

    def _track_depth_jump(self, raw: Optional[np.ndarray]) -> None:
        """Ham haritanın küresel medyanındaki kareden kareye log-sıçrama.

        Görüntü sakinken bu büyükse model haritayı yeniden ölçeklemiştir —
        çapanın düzelttiği durum. Telemetriye girer, büyükse loglanır.
        """
        if raw is None:
            return
        v = raw[raw > 1e-3]
        if v.size < 500:
            return
        med = float(np.median(v))
        if self._depth_med is not None and self._depth_med > 1e-6:
            self._depth_jump = float(np.log(med / self._depth_med))
            if abs(self._depth_jump) > 0.25:
                log.debug("derinlik: kuresel olcek sicramasi %+.0f%% (sahne %s)",
                          100 * (np.exp(self._depth_jump) - 1),
                          "kararli" if self.scene_state.stable else
                          ",".join(self.scene_state.events) or "oturuyor")
        self._depth_med = med

    def telemetry_record(self, report: ObstacleReport, command: AvoidanceCommand,
                         fps: float) -> dict:
        """Kare başına tek satırlık analiz/ML kaydı (bkz. log.Telemetry)."""
        def num(x):
            return None if x is None or not np.isfinite(x) else round(float(x), 4)
        st = self.scene_state
        n, nn = report.nearest, report.nearest_named
        return {
            "frame": report.frame_id,
            "fps": round(fps, 1),
            "action": command.action,
            "stable": st.stable,
            "events": st.events,
            **{k: num(v) for k, v in st.features.items()},
            "depth_med_raw": num(self._depth_med),
            "depth_jump_log": num(self._depth_jump),
            "anchor_k": num(self.anchor.k),
            "anchor_ref": num(self.anchor.ref),
            "ground": bool(self.ground.locked),
            "nearest_m": num(n.distance_m) if n else None,
            "nearest_deg": num(n.bearing_deg) if n else None,
            "named": nn.label if nn else None,
            "named_m": num(nn.distance_m) if nn else None,
            "n_det": len(report.detections),
        }

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
            "scene": self.scene_state,
            "yolo": self.yolo,
            "hand": self.hand,
            "fps": self._fps.tick(),
        }
