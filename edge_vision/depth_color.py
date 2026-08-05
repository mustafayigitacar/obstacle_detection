"""Derinlik → renk eşlemesi. TEK KAYNAK: her iki panel aynı anlamı taşır."""

from __future__ import annotations

import cv2
import numpy as np

DN_LO, DN_HI = 22, 228
DN_RED = 210
RED_MAX_M = 0.40


def set_red_max(v: float) -> None:
    global RED_MAX_M
    RED_MAX_M = float(v)


def norm(z, near: float, far: float, red_max: float = None) -> np.ndarray:
    """[near, red_max] → hepsi kırmızı; [red_max, far] → turuncu..mavi."""
    rm = RED_MAX_M if red_max is None else float(red_max)
    z = np.asarray(z, dtype=np.float32)
    rm = max(rm, near + 1e-6)

    t_red = np.clip((z - near) / (rm - near), 0.0, 1.0)
    dn_red = DN_HI - t_red * (DN_HI - DN_RED)

    t_rest = np.clip((z - rm) / max(1e-6, far - rm), 0.0, 1.0)
    dn_rest = DN_RED - np.sqrt(t_rest) * (DN_RED - DN_LO)

    return np.clip(np.where(z <= rm, dn_red, dn_rest), 0, 255).astype(np.uint8)


def colorize(depth_m: np.ndarray, near: float, far: float) -> np.ndarray:
    return cv2.applyColorMap(norm(depth_m, near, far), cv2.COLORMAP_TURBO)


def fmt(z: float) -> str:
    return f"{z * 100:.0f}cm" if z < 1.0 else f"{z:.1f}m"


def draw_legend(img: np.ndarray, near: float, far: float) -> None:
    """Dikey renk cetveli + sayısal etiket (yerinde çizer)."""
    h, w = img.shape[:2]
    bh, bw = int(h * 0.52), 7
    y0 = int(h * 0.32)
    x1 = w - 5
    x0 = x1 - bw

    zs = near + np.linspace(1.0, 0.0, bh, dtype=np.float32) * (far - near)
    bar = colorize(zs.reshape(-1, 1), near, far)
    img[y0:y0 + bh, x0:x1] = np.repeat(bar, bw, axis=1)
    cv2.rectangle(img, (x0 - 1, y0 - 1), (x1, y0 + bh), (230, 230, 230), 1)

    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        z = far + (near - far) * frac
        yy = int(y0 + frac * (bh - 1))
        cv2.line(img, (x0 - 3, yy), (x0 - 1, yy), (230, 230, 230), 1)
        lbl = fmt(z)
        (tw, _), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_PLAIN, 0.7, 1)
        cv2.putText(img, lbl, (x0 - 5 - tw, yy + 3),
                    cv2.FONT_HERSHEY_PLAIN, 0.7, (255, 255, 255), 1)
