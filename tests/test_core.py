"""Model dosyası / kamera gerektirmeyen çekirdek mantık testleri.

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import logging
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from edge_vision.avoidance import AvoidanceCommand, AvoidancePlanner
from edge_vision.calib_overlay import steepest_segment, worst_kink
from edge_vision.config import (AvoidanceConfig, DepthConfig, ProcessingConfig,
                                SceneConfig, TrackConfig, load_config)
from edge_vision.intrinsics import intrinsics_from_hfov
from edge_vision.metric_depth import MetricDepthNet
from edge_vision.object_detector import Detection, DetectionTracker, fuse_distance
from edge_vision.obstacles import ObstacleReport, ObstacleScanner
from edge_vision.pipeline import VisionPipeline
from edge_vision.scan_smoother import ScanSmoother
from edge_vision.stability import SceneMonitor

ROOT = Path(__file__).resolve().parents[1]


def _report(z, bearing):
    z = np.asarray(z, np.float32)
    return ObstacleReport(frame_id=0, scan_z=z,
                          scan_bearing_deg=np.asarray(bearing, np.float32),
                          scan_u=np.arange(len(z)))


class PlannerTest(unittest.TestCase):
    def setUp(self):
        self.planner = AvoidancePlanner(AvoidanceConfig())
        self.brg = np.linspace(-30, 30, 61)

    def test_clear(self):
        cmd = self.planner.plan(_report(np.full(61, np.inf), self.brg))
        self.assertEqual(cmd.action, "CLEAR")

    def test_stop_when_center_close(self):
        z = np.full(61, np.inf); z[28:33] = 0.2
        self.assertEqual(self.planner.plan(_report(z, self.brg)).action, "STOP")

    def test_follow_gap_in_brake_zone(self):
        z = np.full(61, np.inf); z[20:45] = 0.5          # merkez + sağ dolu
        cmd = self.planner.plan(_report(z, self.brg))
        self.assertEqual(cmd.action, "FOLLOW_GAP")
        self.assertLess(cmd.steer_deg, 0)               # boşluk solda

    def test_no_scan_is_stop(self):
        cmd = self.planner.plan(ObstacleReport(frame_id=0))
        self.assertEqual(cmd.action, "STOP")


class ScannerTest(unittest.TestCase):
    def test_box_becomes_single_obstacle(self):
        w, h = 160, 120
        intr = intrinsics_from_hfov(w, h, 62.2)
        sc = ObstacleScanner(ProcessingConfig(column_stride=2), AvoidanceConfig(), intr)
        depth = np.full((h, w), 4.9, np.float32)
        depth[40:110, 60:100] = 1.0
        u, x, z, brg = sc.scan(depth, 30)
        obs = sc.cluster(u, x, z, brg)
        near = min(obs, key=lambda o: o.distance_m)
        self.assertAlmostEqual(near.forward_m, 1.0, places=3)
        self.assertEqual(near.zone, "center")
        self.assertFalse(np.isnan(x).any())


class SmootherTest(unittest.TestCase):
    def test_retreat_needs_persistence(self):
        s = ScanSmoother(3)
        s.smooth(np.array([1.0], np.float32))
        for _ in range(2):
            z, held = s.smooth(np.array([2.0], np.float32))
            self.assertEqual(z[0], 1.0); self.assertTrue(held[0])
        z, _ = s.smooth(np.array([2.0], np.float32))
        self.assertEqual(z[0], 2.0)
        z, _ = s.smooth(np.array([0.5], np.float32))    # yaklaşma anında kabul
        self.assertEqual(z[0], 0.5)


class TrackerTest(unittest.TestCase):
    def test_label_vote_and_jump_gate(self):
        tr = DetectionTracker(TrackConfig(min_hits=2, jump_persist=3))
        for c in (0, 0, 41, 0):                         # tek karelik yanlış sınıf
            tr.update([Detection(c, str(c), 0.9, 10, 10, 50, 50)])
        cur = tr.current()
        self.assertEqual(len(cur), 1)
        self.assertEqual(cur[0].class_id, 0)
        tid = cur[0].tid
        self.assertEqual(tr.set_distance(tid, 1.0), 1.0)
        self.assertEqual(tr.set_distance(tid, 3.0), 1.0)   # sıçrama hemen kabul edilmez
        tr.set_distance(tid, 3.0)
        self.assertEqual(tr.set_distance(tid, 3.0), 3.0)   # tutarlıysa kabul

    def test_fuse_distance_uses_mask(self):
        depth = np.full((40, 40), 3.0, np.float32)
        depth[10:20, 10:20] = 1.0
        mask = np.zeros_like(depth, bool); mask[10:20, 10:20] = True
        d = Detection(0, "x", 0.9, 5, 5, 30, 30, mask=mask)
        self.assertAlmostEqual(fuse_distance(d, depth), 1.0)


class CalibrationTest(unittest.TestCase):
    def _net(self, points, scale=0.5, offset=0.0):
        net = MetricDepthNet.__new__(MetricDepthNet)   # ONNX oturumu kurmadan
        import threading
        net._lock = threading.Lock()
        net.calib_scale, net.calib_offset, net.min_valid_m = scale, offset, 0.05
        net.set_calibration_table(points)
        return net

    def test_affine_when_no_table(self):
        net = self._net([])
        out = net._calibrate(np.array([[2.0, 0.0]], np.float32))
        self.assertAlmostEqual(float(out[0, 0]), 1.0)
        self.assertEqual(float(out[0, 1]), 0.0)          # geçersiz piksel 0 kalır

    def test_table_interp_and_extrapolation(self):
        net = self._net([[1.0, 0.5], [2.0, 1.5]])
        out = net._calibrate(np.array([1.5, 3.0], np.float32))
        np.testing.assert_allclose(out, [1.0, 2.5], rtol=1e-5)

    def test_kink_checks(self):
        self.assertIsNone(worst_kink([(1, 1), (2, 2)]))
        steep = steepest_segment([(1.0, 0.2), (1.05, 1.0), (3.0, 1.5)])
        self.assertGreater(steep[0], 3.0)


class SceneTest(unittest.TestCase):
    def test_light_jump_is_unstable(self):
        mon = SceneMonitor(SceneConfig(settle_frames=2))
        rng = np.random.default_rng(0)
        base = rng.integers(60, 120, (120, 160, 3), dtype=np.uint8)
        for _ in range(15):
            mon.update(base)
        st = mon.update(np.clip(base.astype(int) + 80, 0, 255).astype(np.uint8))
        self.assertFalse(st.stable)
        self.assertIn("isik", st.events)


class SafetyTest(unittest.TestCase):
    def _pipe(self, age, max_age=1.0):
        pipe = VisionPipeline.__new__(VisionPipeline)   # modelsiz hat iskeleti
        pipe.cfg = SimpleNamespace(depth=DepthConfig(max_age_s=max_age))
        pipe.depth = SimpleNamespace(age_s=age)
        pipe._depth_stale = False
        return pipe

    def test_stale_depth_forces_stop(self):
        cmd = AvoidanceCommand("CLEAR", 0.0, 1.0, 3.0, "yol acik")
        logging.disable(logging.CRITICAL)
        try:
            self.assertEqual(self._pipe(5.0)._guard_stale(cmd).action, "STOP")
            self.assertEqual(self._pipe(0.1)._guard_stale(cmd).action, "CLEAR")
            self.assertEqual(self._pipe(5.0, max_age=0)._guard_stale(cmd).action, "CLEAR")
        finally:
            logging.disable(logging.NOTSET)


class ConfigTest(unittest.TestCase):
    def test_repo_config_loads(self):
        cfg = load_config(ROOT / "config.yaml")
        self.assertGreater(cfg.processing.proc_width, 0)

    def test_unknown_key_warns(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "c.yaml"
            p.write_text("depth:\n  trheads: 2\nfoo: {}\n", encoding="utf-8")
            with self.assertLogs("edge_vision.config", "WARNING") as cm:
                cfg = load_config(p)
            self.assertEqual(cfg.depth.threads, DepthConfig().threads)
            joined = "\n".join(cm.output)
            self.assertIn("trheads", joined)
            self.assertIn("foo", joined)


if __name__ == "__main__":
    unittest.main()
