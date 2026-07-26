"""Zemin düzlemi ayıklama — metrik derinlikten, kalibrasyon gerektirmeden."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .config import GroundConfig
from .intrinsics import Intrinsics


class GroundRemover:
    def __init__(self, intr: Intrinsics, cfg: GroundConfig):
        self.intr = intr
        self.enabled = bool(cfg.enabled)
        self.margin_base = float(cfg.margin_base_m)
        self.margin_per_m = float(cfg.margin_per_m)
        self.min_inlier_ratio = float(cfg.min_inlier_ratio)
        self.min_normal_y = float(cfg.min_normal_y)
        self.fit_step = max(1, int(cfg.fit_step))
        self.ema = float(cfg.ema)
        self.max_miss = int(cfg.max_miss)

        self._plane: Optional[np.ndarray] = None
        self.locked = False
        self.inlier_ratio = 0.0
        self._miss = 0

    # --- düzlem uydurma ---

    def _fit(self, X, Y, Z) -> Optional[np.ndarray]:
        """a·X + b·Y + c·Z = 1, kırpılmış aykırı-nokta ayıklamalı en küçük kareler.

        Eşik YÜZDELİKTİR, sabit metre DEĞİL: sabit eşik, derinlik ölçeği
        değiştiğinde uydurmayı tamamen iptal edip zemin ayıklamayı sessizce
        devre dışı bırakıyordu.
        """
        pts = np.stack([X, Y, Z], axis=1)
        if pts.shape[0] < 200:
            return None
        keep = np.ones(pts.shape[0], dtype=bool)
        coef = None
        for _ in range(4):
            sub = pts[keep]
            if sub.shape[0] < 100:
                break
            c, *_ = np.linalg.lstsq(sub, np.ones(sub.shape[0]), rcond=None)
            nrm = float(np.linalg.norm(c))
            if nrm < 1e-9:
                return None
            coef = c
            resid = np.abs(pts @ coef - 1.0) / nrm
            keep = resid <= max(0.03, float(np.percentile(resid, 60)))
        if coef is None:
            return None

        nrm = float(np.linalg.norm(coef))
        if nrm < 1e-9:
            return None
        normal = coef / nrm
        d = 1.0 / nrm
        if normal[1] < 0:
            normal, d = -normal, -d
        resid = np.abs(pts @ coef - 1.0) / nrm
        self.inlier_ratio = float((resid < 0.08).mean())
        return np.array([normal[0], normal[1], normal[2], d], dtype=np.float64)

    def update(self, depth_m: np.ndarray, roi_top: int) -> None:
        """Uydurma başarısız olsa da son düzlem max_miss kare korunur (yapışkan)."""
        if not self.enabled or depth_m is None:
            self.locked = False
            return

        plane = self._try_fit(depth_m, roi_top)
        if plane is not None:
            if self._plane is None:
                self._plane = plane
            else:
                a = self.ema
                self._plane = (1.0 - a) * self._plane + a * plane
                n = np.linalg.norm(self._plane[:3])
                if n > 1e-9:
                    self._plane[:3] /= n
            self._miss = 0
            self.locked = True
        else:
            self._miss += 1
            if self._miss > self.max_miss:
                self._plane = None
            self.locked = self._plane is not None

    def _try_fit(self, depth_m: np.ndarray, roi_top: int) -> Optional[np.ndarray]:
        h, w = depth_m.shape[:2]
        s = self.fit_step
        v_start = roi_top + (h - roi_top) // 2
        Zg = depth_m[v_start::s, ::s]
        vs = np.arange(v_start, h, s)[:Zg.shape[0]]
        us = np.arange(0, w, s)[:Zg.shape[1]]
        U, V = np.meshgrid(us, vs)

        ok = Zg > 1e-3
        if int(ok.sum()) < 200:
            return None
        Z = Zg[ok].astype(np.float64)
        X = Z * (U[ok] - self.intr.cx) / self.intr.fx
        Y = Z * (V[ok] - self.intr.cy) / self.intr.fy

        plane = self._fit(X, Y, Z)
        if plane is None:
            return None
        if abs(plane[1]) < self.min_normal_y or self.inlier_ratio < self.min_inlier_ratio:
            return None
        return plane

    # --- maske ---

    def ground_mask(self, depth_m: np.ndarray, roi_top: int) -> Optional[np.ndarray]:
        """Zemin (ve altı) piksellerini True işaretler; taramadan çıkarılır."""
        if not self.locked or self._plane is None or depth_m is None:
            return None
        h, w = depth_m.shape[:2]
        Z = depth_m[roi_top:, :].astype(np.float32)
        vs = np.arange(roi_top, h, dtype=np.float32)[:, None]
        us = np.arange(0, w, dtype=np.float32)[None, :]
        X = Z * (us - self.intr.cx) / self.intr.fx
        Y = Z * (vs - self.intr.cy) / self.intr.fy

        nx, ny, nz, d = self._plane
        height = -(nx * X + ny * Y + nz * Z - d)
        # Marj mesafeyle büyür: derinlik gürültüsü uzakta arttığı için sabit
        # marj uzaktaki zemini engel diye sızdırıyordu.
        margin = self.margin_base + self.margin_per_m * np.maximum(Z, 0.0)
        mask = np.zeros((h, w), dtype=bool)
        mask[roi_top:, :] = (height <= margin) & (Z > 1e-3)
        return mask

    @property
    def camera_height_m(self) -> Optional[float]:
        return None if self._plane is None else float(abs(self._plane[3]))
