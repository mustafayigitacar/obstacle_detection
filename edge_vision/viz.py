"""Hata ayıklama görselleştirmesi.

Sol panel: kamera + yön iğneleri + kuşbakışı radar.
Sağ panel: kenar haritası veya metrik derinlik haritası + renk cetveli.

Mesafe kamera görüntüsüne bindirilmez; sözde-LiDAR verisinin doğru gösterimi
kuşbakışı radardır (kamera altta ortada, ileri yukarı, yanal yana).
"""

from __future__ import annotations

import numpy as np
import cv2

from . import depth_color
from .obstacles import ObstacleReport


def _label(out, text, org, color, scale=0.9) -> None:
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, scale, 1)
    x, y = int(org[0]), int(org[1])
    x = max(2, min(out.shape[1] - tw - 2, x))
    y = max(th + 2, min(out.shape[0] - 3, y))
    cv2.rectangle(out, (x - 2, y - th - 2), (x + tw + 2, y + 3), (25, 25, 25), -1)
    cv2.putText(out, text, (x, y), cv2.FONT_HERSHEY_PLAIN, scale, color, 1)


def _blur_background(proc_bgr: np.ndarray, report: ObstacleReport) -> np.ndarray:
    h, w = proc_bgr.shape[:2]
    if not report.obstacles:
        return proc_bgr.copy()
    fg = np.zeros((h, w), dtype=bool)
    for ob in report.obstacles:
        u0, u1 = max(0, ob.px_u0), min(w, ob.px_u1 + 1)
        if u1 > u0:
            fg[:, u0:u1] = True
    blurred = cv2.GaussianBlur(proc_bgr, (21, 21), 0)
    return np.where(fg[:, :, None], proc_bgr, blurred).astype(np.uint8)


def _draw_pins(out: np.ndarray, report: ObstacleReport, roi_top: int,
               near: float, far: float) -> None:
    h, w = out.shape[:2]
    nearest = report.nearest
    for ob in report.obstacles:
        color = depth_color.color_at(ob.distance_m, near, far)
        uc = (max(0, ob.px_u0) + min(w - 1, ob.px_u1)) // 2
        is_near = nearest is not None and ob.px_u0 == nearest.px_u0 \
            and ob.px_u1 == nearest.px_u1
        cv2.line(out, (uc, roi_top + 1), (uc, roi_top + 16), color, 2 if is_near else 1)
        cv2.circle(out, (uc, roi_top + 1), 3, color, -1)
        _label(out, f"{ob.distance_m:.2f}m", (uc - 14, roi_top + 30), color, 0.8)


def _draw_radar(out: np.ndarray, report: ObstacleReport,
                near: float, far: float) -> None:
    """Kuşbakışı radar. Çizim alt-tuvale yapılır; taşanlar otomatik kırpılır."""
    h, w = out.shape[:2]
    rw, rh = int(w * 0.42), int(h * 0.50)
    bx, by = w - rw - 3, h - rh - 3
    cv = out[by:by + rh, bx:bx + rw]

    cv[:] = (cv.astype(np.float32) * 0.35 + np.array([18, 18, 18]) * 0.65).astype(np.uint8)
    cv2.rectangle(cv, (0, 0), (rw - 1, rh - 1), (90, 90, 90), 1)

    ox, oy = rw // 2, rh - 5
    rng = max(1.0, far)
    ppm = (rh - 14) / rng

    cv2.line(cv, (ox, oy), (ox, 2), (60, 60, 60), 1)
    for r in range(1, int(rng) + 1):
        rr = int(r * ppm)
        cv2.ellipse(cv, (ox, oy), (rr, rr), 0, 180, 360, (70, 70, 70), 1)
        cv2.putText(cv, f"{r}m", (ox + 3, oy - rr + 10), cv2.FONT_HERSHEY_PLAIN,
                    0.7, (150, 150, 150), 1)

    if report.scan_z is not None and report.scan_bearing_deg is not None:
        z = report.scan_z
        fin = np.isfinite(z) & (z <= rng)
        if fin.any():
            zf = z[fin]
            px = (ox + zf * np.tan(np.radians(report.scan_bearing_deg[fin])) * ppm).astype(np.int32)
            py = (oy - zf * ppm).astype(np.int32)
            ins = (px >= 1) & (px < rw - 1) & (py >= 1) & (py < rh - 1)
            if ins.any():
                cols = (depth_color.colors_for(zf[ins], near, far) * 0.55).astype(np.uint8)
                cv[py[ins], px[ins]] = cols

    nearest = report.nearest
    for ob in report.obstacles:
        px = int(np.clip(ox + ob.lateral_m * ppm, 2, rw - 2))
        py = int(np.clip(oy - ob.forward_m * ppm, 2, rh - 2))
        color = depth_color.color_at(ob.distance_m, near, far)
        is_near = nearest is not None and ob.px_u0 == nearest.px_u0 \
            and ob.px_u1 == nearest.px_u1
        rad = 5 if is_near else 4
        cv2.circle(cv, (px, py), rad, color, -1)
        cv2.circle(cv, (px, py), rad, (255, 255, 255), 1)
        if is_near:
            cv2.circle(cv, (px, py), 8, (255, 255, 255), 1)

    cv2.putText(cv, "RADAR ust gorunum", (4, 12), cv2.FONT_HERSHEY_PLAIN,
                0.8, (210, 210, 210), 1)
    if nearest is not None:
        _label(cv, f"EN YAKIN {nearest.distance_m:.2f}m", (4, rh - 4),
               depth_color.color_at(nearest.distance_m, near, far), 0.8)


def draw_debug(proc_bgr, report, command, roi_top, fps,
               near: float = 0.05, far: float = 3.0,
               show_obstacles: bool = True) -> np.ndarray:
    out = _blur_background(proc_bgr, report) if show_obstacles else proc_bgr.copy()
    h, w = out.shape[:2]

    cv2.line(out, (0, roi_top), (w, roi_top), (80, 80, 80), 1)

    if show_obstacles:
        _draw_pins(out, report, roi_top, near, far)
        _draw_radar(out, report, near, far)

    cv2.putText(out, f"{command.action} steer={command.steer_deg:+.0f} "
                     f"v={command.speed_scale:.2f}",
                (4, 14), cv2.FONT_HERSHEY_PLAIN, 0.9, (255, 255, 255), 1)
    cv2.putText(out, f"fps={fps:.1f}", (4, h - 6),
                cv2.FONT_HERSHEY_PLAIN, 0.9, (255, 255, 255), 1)
    return out


def edge_view(edges, roi_top) -> np.ndarray:
    if edges is None:
        return np.zeros((10, 10, 3), np.uint8)
    out = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cv2.line(out, (0, roi_top), (out.shape[1], roi_top), (80, 80, 80), 1)
    cv2.putText(out, "kenar haritasi", (4, 14),
                cv2.FONT_HERSHEY_PLAIN, 0.9, (255, 255, 255), 1)
    return out


def compose_view(proc_bgr, edges, report, command, roi_top, fps, scale=2,
                 depth=None, depth_m=None, right_panel="depth",
                 max_range_m=5.0, show_obstacles=True) -> np.ndarray:
    # Renk ölçeği ÖNCE hesaplanır: iki panel AYNI aralığı kullanmalı, yoksa
    # aynı renk iki panelde farklı mesafe anlamına gelir.
    if depth is not None:
        near, far = depth.color_range(depth_m)
    else:
        near, far = 0.05, max_range_m

    left = draw_debug(proc_bgr, report, command, roi_top, fps,
                      near=near, far=far, show_obstacles=show_obstacles)
    right = None
    if right_panel == "depth" and depth is not None:
        right = depth.visualize(depth_m, (near, far))
        if right is not None:
            depth_color.draw_legend(right, near, far)
    if right is None:
        right = edge_view(edges, roi_top)
    if right.shape[:2] != left.shape[:2]:
        right = cv2.resize(right, (left.shape[1], left.shape[0]))
    panel = np.hstack([left, right])
    if scale > 1:
        panel = cv2.resize(panel, (panel.shape[1] * scale, panel.shape[0] * scale),
                           interpolation=cv2.INTER_NEAREST)
    return panel
