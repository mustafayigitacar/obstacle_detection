"""Pencere-içi derinlik kalibrasyonu. Terminal gerektirmez.

    c/ESC = aç/kapa   ENTER = ölçümü kaydet   f = uygula
    x = noktaları sil   r = başlangıca dön   rakam/nokta = mesafe yaz

Noktalar HAM değerden alınır (kalibre değerden değil): ham mutlaktır, bu yüzden
üst üste kalibrasyonlar birbirini bozmaz.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

AIM = (0.36, 0.64, 0.36, 0.64)
CAPTURE_FRAMES = 12
CAPTURE_TIMEOUT_S = 5.0


def _set_key(txt: str, key: str, val: str) -> str:
    pat = rf"(?m)^(\s*){key}:.*$"
    if re.search(pat, txt):
        return re.sub(pat, rf"\g<1>{key}: {val}", txt)
    return re.sub(r"(?m)^(\s*threads:.*)$", rf"\1\n  {key}: {val}", txt, count=1)


def write_calib_points(config_path: str, points) -> None:
    body = ", ".join(f"[{a:.4f}, {b:.4f}]" for a, b in points)
    p = Path(config_path)
    p.write_text(_set_key(p.read_text(encoding="utf-8"), "calib_points", f"[{body}]"),
                 encoding="utf-8")


def aim_box(ph: int, pw: int) -> Tuple[int, int, int, int]:
    return (int(ph * AIM[0]), int(ph * AIM[1]), int(pw * AIM[2]), int(pw * AIM[3]))


def measure_object(depth_m: Optional[np.ndarray], tol: float = 0.15
                   ) -> Tuple[Optional[float], Optional[np.ndarray]]:
    """Nişan kutusundaki EN ÖN cismi ayırır → (mesafe, maske).

    Düz medyan yerine bağlı bileşen: cisim kutuyu doldurmuyorsa arka plan
    ölçüme karışır ve kalibrasyonu bozar.
    """
    if depth_m is None:
        return None, None
    ph, pw = depth_m.shape[:2]
    y0, y1, x0, x1 = aim_box(ph, pw)
    box = depth_m[y0:y1, x0:x1]
    v = box[box > 1e-3]
    if v.size < 20:
        return None, None

    near = float(np.percentile(v, 10))
    margin = tol * near + 0.02
    band = ((depth_m >= near - margin) & (depth_m <= near + margin)
            & (depth_m > 1e-3)).astype(np.uint8)

    n_lab, labels = cv2.connectedComponents(band, connectivity=8)
    if n_lab <= 1:
        return None, None
    sub = labels[y0:y1, x0:x1]
    ids, counts = np.unique(sub[sub > 0], return_counts=True)
    if ids.size == 0:
        return None, None
    mask = labels == int(ids[int(np.argmax(counts))])

    boxm = np.zeros(mask.shape, dtype=bool)
    boxm[y0:y1, x0:x1] = True
    vals = depth_m[mask & boxm]
    if vals.size < 20:
        return None, None
    return float(np.median(vals)), mask


class CalibOverlay:
    def __init__(self, config_path: str, net):
        self.config_path = config_path
        self.net = net
        self._orig_points = list(net.get_calibration_table())

        self.active = False
        self.buffer = ""
        self.points: List[Tuple[float, float]] = list(self._orig_points)
        self.message = "cismi nisan kutusuna koy"
        self.live_dist: Optional[float] = None
        self.live_raw: Optional[float] = None
        self.mask: Optional[np.ndarray] = None
        self._pending: Optional[dict] = None

    # --- döngü ---

    def toggle(self) -> None:
        self.active = not self.active
        self.buffer = ""
        self._pending = None
        self.message = "cismi nisan kutusuna koy, mesafeyi yaz" if self.active else ""

    def update(self, depth_m: Optional[np.ndarray],
               raw_m: Optional[np.ndarray] = None) -> None:
        self.live_dist, self.mask = measure_object(depth_m)
        self.live_raw = None
        if self.mask is not None and raw_m is not None and raw_m.shape == self.mask.shape:
            ph, pw = raw_m.shape[:2]
            y0, y1, x0, x1 = aim_box(ph, pw)
            boxm = np.zeros(self.mask.shape, dtype=bool)
            boxm[y0:y1, x0:x1] = True
            v = raw_m[self.mask & boxm]
            if v.size:
                self.live_raw = float(np.median(v))

        if self._pending is None:
            return
        if self.live_raw is not None:
            self._pending["samples"].append(self.live_raw)
        if len(self._pending["samples"]) >= CAPTURE_FRAMES:
            self._finalize()
        elif time.time() - self._pending["t0"] > CAPTURE_TIMEOUT_S:
            self._pending = None
            self.message = "olcum alinamadi — cisim nisan kutusunda mi?"

    def handle_key(self, key: int) -> None:
        if key == 27:
            self.toggle()
        elif (ord("0") <= key <= ord("9")) or key == ord("."):
            self.buffer += chr(key)
        elif key == 8:
            self.buffer = self.buffer[:-1]
        elif key == ord("x"):
            self.points.clear()
            self.message = "noktalar temizlendi"
        elif key == ord("r"):
            self._restore()
        elif key in (13, 10):
            self._start_capture()
        elif key == ord("f"):
            self._apply()

    # --- yakalama ---

    def _start_capture(self) -> None:
        try:
            real = float(self.buffer.replace(",", "."))
        except ValueError:
            self.message = "once gecerli mesafe yaz (or 0.30)"
            return
        if real <= 0:
            self.message = "mesafe pozitif olmali"
            return
        self._pending = {"real": real, "samples": [], "t0": time.time()}
        self.message = "olculuyor... cismi SABIT tut"

    def _finalize(self) -> None:
        pend, self._pending = self._pending, None
        s = np.array(pend["samples"], dtype=np.float64)
        raw, spread = float(np.median(s)), float(np.std(s))
        real = pend["real"]

        if spread > max(0.05, 0.20 * raw):
            self.message = f"kararsiz olcum (+-{spread:.2f}) — sabit tut, tekrar"
            return

        self.points = [p for p in self.points if abs(p[1] - real) > 1e-3]
        self.points.append((raw, real))
        self.points.sort(key=lambda p: p[0])
        self.buffer = ""
        self.message = f"OK {real:.2f}m <- ham {raw:.2f} (nokta={len(self.points)})"

    # --- uygulama ---

    def _apply(self) -> None:
        if len(self.points) < 2:
            self.message = "en az 2 nokta gerekli (yakin + uzak)"
            return
        pts = sorted(self.points, key=lambda p: p[0])
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]

        for i in range(1, len(xs)):
            if xs[i] - xs[i - 1] < 1e-3:
                self.message = "ayni ham deger iki mesafede — model ayirt edemiyor"
                return
            if ys[i] <= ys[i - 1]:
                self.message = f"tutarsiz nokta ({ys[i-1]:.2f}/{ys[i]:.2f}m) — x=sil"
                return

        # ARALIK GENİŞLİĞİ — kritik. Birbirine çok yakın iki noktanın eğimi
        # gürültüye boğulur ama ekstrapolasyonla TÜM menzili yönetir; gerçek
        # bir vakada sahneyi 0.5-1.1 m'ye sıkıştırıp maviyi yok etmişti.
        span = ys[-1] - ys[0]
        ratio = ys[-1] / max(ys[0], 1e-6)
        if span < 0.30 and ratio < 2.0:
            self.message = (f"aralik cok dar ({span*100:.0f}cm) — "
                            f"birbirinden UZAK mesafeler gir")
            return

        try:
            write_calib_points(self.config_path, pts)
        except Exception as exc:                        # noqa: BLE001
            self.message = f"YAZMA HATASI: {exc}"
            return
        self.net.set_calibration_table(pts)
        lo, hi = ys[0], ys[-1]
        warn = "  UYARI: 0.5m alti nokta YOK" if lo > 0.5 else ""
        self.message = f"UYGULANDI {len(pts)} nokta  {lo:.2f}-{hi:.2f}m{warn}"

    def _restore(self) -> None:
        try:
            write_calib_points(self.config_path, self._orig_points)
        except Exception as exc:                        # noqa: BLE001
            self.message = f"YAZMA HATASI: {exc}"
            return
        self.net.set_calibration_table(self._orig_points)
        self.points = list(self._orig_points)
        self.message = "baslangic kalibrasyonuna donuldu"

    # --- çizim ---

    def draw(self, view: np.ndarray, scale: int, pw: int, ph: int) -> None:
        if not self.active:
            return
        lw, lh = pw * scale, ph * scale
        left = view[:lh, :lw]

        dim = cv2.GaussianBlur(left, (21, 21), 0)
        dim = (dim.astype(np.float32) * 0.45).astype(np.uint8)
        if self.mask is not None:
            m8 = cv2.resize(self.mask.astype(np.uint8), (lw, lh),
                            interpolation=cv2.INTER_NEAREST)
            keep = m8.astype(bool)
            left[~keep] = dim[~keep]
            cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(left, cnts, -1, (0, 255, 255), 2)
        else:
            left[:] = dim

        y0, y1, x0, x1 = aim_box(ph, pw)
        cv2.rectangle(left, (x0 * scale, y0 * scale), (x1 * scale, y1 * scale),
                      (0, 255, 255), 1)

        shown = "--" if self.live_dist is None else f"{self.live_dist:.2f}m"
        rawv = "--" if self.live_raw is None else f"{self.live_raw:.2f}"
        pend = "" if self._pending is None else \
            f"  [olculuyor {len(self._pending['samples'])}/{CAPTURE_FRAMES}]"
        tbl = getattr(self.net, "table_size", 0)
        lines = [
            f"KALIBRASYON   c/ESC=cik   r=geri al   tablo:{tbl} nokta",
            f"konturdaki cisim: {shown}   (ham {rawv}){pend}",
            f"gercek mesafe(m): {self.buffer}_",
            "ENTER=kaydet   x=noktalari sil   f=uygula",
            f"nokta:{len(self.points)}   {self.message}",
        ]
        y = 30
        for i, ln in enumerate(lines):
            (tw, th), _ = cv2.getTextSize(ln, cv2.FONT_HERSHEY_PLAIN, 1.1, 1)
            cv2.rectangle(view, (5, y - th - 4), (13 + tw, y + 5), (18, 18, 18), -1)
            col = (0, 255, 255) if i != 4 else (140, 255, 140)
            cv2.putText(view, ln, (9, y), cv2.FONT_HERSHEY_PLAIN, 1.1, col, 1)
            y += th + 9
