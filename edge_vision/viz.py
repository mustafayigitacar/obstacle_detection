"""Hata ayıklama görselleştirmesi.

Sol panel: kamera + nesne kutuları/maskeleri.
Sağ panel: kenar haritası veya metrik derinlik haritası + renk cetveli.

Mesafe okuması nesne etiketlerinden ve üstteki "EN YAKIN NESNE" bandından gelir.
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
    if not report.obstacles and not report.detections:
        return proc_bgr.copy()
    fg = np.zeros((h, w), dtype=bool)
    for ob in report.obstacles:
        u0, u1 = max(0, ob.px_u0), min(w, ob.px_u1 + 1)
        if u1 > u0:
            fg[:, u0:u1] = True
    # Tanınan nesneler de ön plandır; maske varsa sütun değil cismin kendisi.
    for d in report.detections:
        if d.mask is not None and d.mask.shape == (h, w):
            fg |= d.mask
            continue
        u0, u1 = max(0, d.x0), min(w, d.x1 + 1)
        if u1 > u0:
            fg[:, u0:u1] = True
    blurred = cv2.GaussianBlur(proc_bgr, (21, 21), 0)
    return np.where(fg[:, :, None], proc_bgr, blurred).astype(np.uint8)


def _draw_detections(out: np.ndarray, report: ObstacleReport) -> None:
    """Nesne kutuları: hepsi soluk, en yakın olan vurgulu (ad + mesafe).

    Engel katmanı "önümde bir şey var" der; bu katman onun ADINI verir.
    """
    nn = report.nearest_named
    for d in report.detections:
        is_near = nn is not None and d is nn
        color = (0, 235, 255) if is_near else (170, 170, 170)
        # Maske varsa cismin KENDİ sınırı çizilir; kutu yalnızca çerçevedir.
        if d.mask is not None and d.mask.shape == out.shape[:2]:
            cnts, _ = cv2.findContours(d.mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, cnts, -1, color, 2 if is_near else 1)
        cv2.rectangle(out, (d.x0, d.y0), (d.x1, d.y1), color, 1)
        dist = f" {d.distance_m:.2f}m" if np.isfinite(d.distance_m) else ""
        # Kutu üste yapışıksa etiket içeri alınır; yoksa HUD satırlarının
        # (komut + en-yakın bandı, alt kenar ~34 px) arkasında kaybolur.
        ly = d.y0 - 4 if d.y0 - 4 >= 46 else max(46, d.y0 + 16)
        _label(out, f"{d.label}{dist}", (d.x0, ly), color, 1.0 if is_near else 0.8)
    if nn is not None:
        dist = f"{nn.distance_m:.2f} m" if np.isfinite(nn.distance_m) else "?"
        _label(out, f"EN YAKIN NESNE: {nn.label}  {dist}", (4, 30), (0, 235, 255), 1.0)


def draw_debug(proc_bgr, report, command, fps,
               show_obstacles: bool = True, yolo=None) -> np.ndarray:
    out = _blur_background(proc_bgr, report) if show_obstacles else proc_bgr.copy()
    h, w = out.shape[:2]

    if show_obstacles:
        _draw_detections(out, report)

    cv2.putText(out, f"{command.action} steer={command.steer_deg:+.0f} "
                     f"v={command.speed_scale:.2f}",
                (4, 14), cv2.FONT_HERSHEY_PLAIN, 0.9, (255, 255, 255), 1)
    stat = f"fps={fps:.1f}"
    if yolo is not None:
        stat += f"  yolo={yolo.update_rate:.0f}fps({yolo.infer_ms:.0f}ms)"
    cv2.putText(out, stat, (4, h - 6), cv2.FONT_HERSHEY_PLAIN, 0.9, (255, 255, 255), 1)
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
                 max_range_m=5.0, show_obstacles=True, yolo=None) -> np.ndarray:
    # Renk ölçeği ÖNCE hesaplanır: iki panel AYNI aralığı kullanmalı, yoksa
    # aynı renk iki panelde farklı mesafe anlamına gelir.
    if depth is not None:
        near, far = depth.color_range(depth_m)
    else:
        near, far = 0.05, max_range_m

    left = draw_debug(proc_bgr, report, command, fps,
                      show_obstacles=show_obstacles, yolo=yolo)
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
