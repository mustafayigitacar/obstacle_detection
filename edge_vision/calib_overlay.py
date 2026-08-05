"""Pencere-içi derinlik kalibrasyonu. Terminal gerektirmez.

    c/ESC = aç/kapa   ENTER = ölçümü kaydet   f = uygula
    n = sonraki nesne   m = nişan kutusu   +/- = kutu boyutu
    x = noktaları sil   r = başlangıca dön   rakam/nokta = mesafe yaz

Hedef bir nesneye KİLİTLENİR ve `n`'e basılana kadar değişmez; ölçüm ortasında
cisim değiştiremez.

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

AIM_FRAC = 0.28         # nişan kutusunun varsayılan kenar oranı
CAPTURE_TIMEOUT_S = 6.0
BAND_TOL = 0.13         # yedek bant yöntemi için derinlik bandı (±%)
GROW_LOG_TOL = 0.30     # bölge büyütme adım eşiği (log derinlik ≈ oransal %35)
MAX_KINK = 3.0          # komşu parçalar arası izin verilen en büyük eğim oranı
MAX_STEEP = 3.0         # bir parça tablonun genel eğiminin en fazla bu katı olabilir
MIN_REAL_M = 0.02       # bunun dışı yazım hatası ("210" = santim/metre karışması)
MAX_REAL_M = 20.0


def table_slopes(points) -> List[Tuple[float, float, float]]:
    """Parça eğimleri: (alt_m, üst_m, m / ham birim)."""
    pts = sorted(points, key=lambda p: p[0])
    out = []
    for i in range(1, len(pts)):
        dx = pts[i][0] - pts[i - 1][0]
        if dx > 1e-9:
            out.append((pts[i - 1][1], pts[i][1], (pts[i][1] - pts[i - 1][1]) / dx))
    return out


def worst_kink(points) -> Optional[Tuple[float, float, float]]:
    """En kötü DÜŞEN eğim oranı → (oran, suçlu parçanın alt_m, üst_m).

    Kontrol ASİMETRİKTİR: yalnızca eğimin mesafeyle azaldığı durum cezalandırılır.
    Model uzağı sıkıştırdığı için metre/ham birim mesafeyle ARTAR — uzak parçanın
    dik olması normaldir. Tersi ölçüm hatasının imzasıdır ve okumayı zıplatır.
    """
    sl = table_slopes(points)
    if len(sl) < 2:
        return None
    worst = None
    for a, b in zip(sl, sl[1:]):
        ratio = a[2] / max(b[2], 1e-9)      # yakın parça / uzak parça
        if worst is None or ratio > worst[0]:
            worst = (ratio, a[0], a[1])
    return worst


def steepest_segment(points) -> Optional[Tuple[float, float, float]]:
    """Tablonun GENEL eğimine göre en dik parça → (oran, alt_m, üst_m).

    Dirsek kontrolü yalnızca komşulara bakar ve eğimin mesafeyle artmasını normal
    sayar; ama iki nokta ham eksende çok yakınsa o parça tek başına uçar
    (ölçüldü: 0.044 ham birim aralıkla eğim 15.8, gürültü mesafede ±25 cm).
    """
    sl = table_slopes(points)
    if not sl:
        return None
    pts = sorted(points, key=lambda p: p[0])
    span = pts[-1][0] - pts[0][0]
    if span <= 1e-9:
        return None
    overall = (pts[-1][1] - pts[0][1]) / span
    if overall <= 1e-9:
        return None
    lo, hi, s = max(sl, key=lambda t: t[2])
    return (s / overall, lo, hi)


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


def aim_box(ph: int, pw: int, frac: float = AIM_FRAC) -> Tuple[int, int, int, int]:
    """Ortalanmış nişan kutusu. `frac` kenar oranı (0.28 = kenarın %28'i)."""
    f = min(0.98, max(0.05, float(frac))) / 2.0
    return (int(ph * (0.5 - f)), int(ph * (0.5 + f)),
            int(pw * (0.5 - f)), int(pw * (0.5 + f)))


def measure_object(depth_m: Optional[np.ndarray], target=None,
                   ground_mask: Optional[np.ndarray] = None, tol: float = BAND_TOL,
                   aim_frac: float = AIM_FRAC, log_tol: float = GROW_LOG_TOL
                   ) -> Tuple[Optional[float], Optional[np.ndarray]]:
    """Ölçülecek cismi ayırır → (mesafe, maske).

    Öncelik sırası: segmentasyon maskesi → bölge büyütme → bant (yedek).
    `target` maskesizse sınır onun kutusudur; hedef yoksa nişan kutusuna düşülür
    (COCO dışı cisimler için). Ayırma ölçütü mutlak mesafe değil, derinlik
    sürekliliğidir: sabit bant yüzü gövdeden koparıyordu.

    Kontur = ölçülen piksel kümesi; taşarsa ölçüm de taşmıştır.
    """
    if depth_m is None:
        return None, None
    ph, pw = depth_m.shape[:2]
    if target is not None:
        y0, y1 = max(0, target.y0), min(ph, target.y1 + 1)
        x0, x1 = max(0, target.x0), min(pw, target.x1 + 1)
        near_pct = 25            # kutu içi: ön yüzey, kenar gürültüsüne kapılmadan
    else:
        y0, y1, x0, x1 = aim_box(ph, pw, aim_frac)
        near_pct = 10            # nişan kutusu: en ön yüzeye yaslan
    if y1 - y0 < 4 or x1 - x0 < 4:
        return None, None

    valid = depth_m > 1e-3
    if ground_mask is not None and ground_mask.shape == depth_m.shape:
        valid = valid & ~ground_mask

    # Maske varsa tahmin yok. Zemin filtresi burada UYGULANMAZ: maske cismin
    # piksellerini zaten veriyor, düzlem filtresi (ölçüldü) %63'ünü siliyordu.
    seg = getattr(target, "mask", None) if target is not None else None
    if seg is not None and seg.shape == depth_m.shape:
        mask = seg & (depth_m > 1e-3)
        vals = depth_m[mask]
        if vals.size >= 20:
            return float(np.median(vals)), mask

    v = depth_m[y0:y1, x0:x1][valid[y0:y1, x0:x1]]
    if v.size < 20:
        return None, None

    near = float(np.percentile(v, near_pct))

    mask = _grow_region(depth_m, valid, (y0, y1, x0, x1), near, tol, log_tol)
    if mask is None or mask.sum() < 20:
        # Yedek: büyüme tutmazsa (parçalı/gürültülü yüzey) klasik bant.
        mask = _band_region(depth_m, valid, (y0, y1, x0, x1), near, tol)
    if mask is None:
        return None, None

    vals = depth_m[mask]
    if vals.size < 20:
        return None, None
    return float(np.median(vals)), mask


def _grow_region(depth_m, valid, box, near: float, tol: float,
                 log_tol: float = 0.30):
    """Derinlik sürekliliğine göre bölge büyütme; kutu dışına çıkamaz.

    `floodFill` sabit-aralık kapalıyken farkı komşu pikselle karşılaştırır.
    Büyüme LOG derinlikte yapılır: eşik mutlak değil oransal olur, böylece aynı
    eşik her mesafede çalışır (yüz→gövde sıçraması 0.29, arka plan 1.43 birim).
    """
    ph, pw = depth_m.shape[:2]
    y0, y1, x0, x1 = box
    sub = depth_m[y0:y1, x0:x1]
    ok = valid[y0:y1, x0:x1]

    # Tohum ön yüzeye en yakın piksel; kutu merkezi arka plan olabilir.
    hh, ww = sub.shape
    yy, xx = np.mgrid[0:hh, 0:ww]
    cost = np.where(ok, np.abs(sub - near), np.inf)
    cost = cost + 0.002 * np.hypot(yy - hh / 2.0, xx - ww / 2.0)
    if not np.isfinite(cost).any():
        return None
    sy, sx = np.unravel_index(int(np.argmin(cost)), cost.shape)

    ff = np.ones((ph + 2, pw + 2), np.uint8)     # her yer kapalı
    ff[1 + y0:1 + y1, 1 + x0:1 + x1] = 0         # yalnızca hedef kutusu açık
    ff[1:-1, 1:-1][~valid] = 1                   # geçersiz + zemin kapalı

    img = np.log(np.maximum(depth_m, 1e-3)).astype(np.float32)
    step = float(log_tol)
    cv2.floodFill(np.ascontiguousarray(img), ff,
                  (x0 + int(sx), y0 + int(sy)), 0, (step,), (step,),
                  4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8))
    return ff[1:-1, 1:-1] == 255


def _band_region(depth_m, valid, box, near: float, tol: float):
    """Yedek yöntem: ön yüzey etrafında sabit bant + en büyük bağlı bileşen."""
    y0, y1, x0, x1 = box
    margin = tol * near + 0.02
    band = (valid & (depth_m >= near - margin) & (depth_m <= near + margin))
    boxm = np.zeros(band.shape, dtype=bool)
    boxm[y0:y1, x0:x1] = True
    band = (band & boxm).astype(np.uint8)

    n_lab, labels = cv2.connectedComponents(band, connectivity=8)
    if n_lab <= 1:
        return None
    ids, counts = np.unique(labels[labels > 0], return_counts=True)
    if ids.size == 0:
        return None
    return labels == int(ids[int(np.argmax(counts))])


class CalibOverlay:
    def __init__(self, config_path: str, net, cfg=None):
        from .config import CalibConfig

        self.config_path = config_path
        self.net = net
        self.cfg = cfg or CalibConfig()
        self._orig_points = list(net.get_calibration_table())

        self.active = False
        self.buffer = ""
        self.points: List[Tuple[float, float]] = list(self._orig_points)
        self.message = "hedefi sec, gercek mesafeyi yaz"
        self.live_dist: Optional[float] = None
        self.live_raw: Optional[float] = None
        self.mask: Optional[np.ndarray] = None
        self._pending: Optional[dict] = None
        # Hedef seçimi: bir nesneye KİLİTLENİR, `n` sıradakine geçirir.
        self.manual = False
        self.aim_frac = float(self.cfg.aim_frac)
        self._sel = 0
        self._locked_tid: Optional[int] = None
        self._cands: List = []
        self._small: List = []
        self._target = None
        self._clipped = False    # ölçülen bölge nişan kutusuna dayandı mı

    # --- döngü ---

    def toggle(self) -> None:
        self.active = not self.active
        self.buffer = ""
        self._pending = None
        self._locked_tid = None          # her girişte hedef yeniden seçilsin
        # Nesne kipi varsayılandır: `m` yapışkan olduğu için farkında olmadan
        # nişan kutusuyla ölçmeye devam etmek mümkündü, kutu da cismi kırpıyordu.
        self.manual = False
        self.message = "hedefi sec (n), gercek mesafeyi yaz" if self.active else ""

    def _pick_target(self, detections, shape) -> None:
        """Hedefi iz kimliğine (`tid`) KİLİTLER: `n`'e basılmadıkça ya da hedef
        sahneden çıkmadıkça değişmez — ölçüm ortasında cisim değiştiremez.

        `min_target_area_frac` altındaki kutular aday sayılmaz: kapı kolu gibi
        birkaç yüz piksellik cisimde güvenilir medyan alacak piksel kalmıyor.
        Bunlar ekranda "kucuk" diye gösterilir, nişan kutusuyla (`m`) ölçülebilir.
        """
        self._cands, self._small = [], []
        if shape is None:
            self._target = None
            return
        ph, pw = shape
        min_area = float(self.cfg.min_target_area_frac) * ph * pw
        for d in (detections or []):
            area = (d.x1 - d.x0 + 1) * (d.y1 - d.y0 + 1)
            (self._cands if area >= min_area else self._small).append(d)
        self._cands.sort(key=lambda d: d.x0 + d.x1)

        if self.manual or not self._cands:
            self._target = None
            return

        for i, d in enumerate(self._cands):
            if d.tid >= 0 and d.tid == self._locked_tid:
                self._sel, self._target = i, d
                return

        # Kilit yok/kayboldu: en BÜYÜK aday — en çok piksel, en güvenilir medyan.
        self._sel = max(range(len(self._cands)),
                        key=lambda i: (self._cands[i].x1 - self._cands[i].x0)
                        * (self._cands[i].y1 - self._cands[i].y0))
        self._target = self._cands[self._sel]
        self._locked_tid = self._target.tid

    def update(self, depth_m: Optional[np.ndarray],
               raw_m: Optional[np.ndarray] = None,
               detections=None, ground_mask: Optional[np.ndarray] = None) -> None:
        self._pick_target(detections, None if depth_m is None else depth_m.shape[:2])
        self.live_dist, self.mask = measure_object(depth_m, self._target, ground_mask,
                                                   tol=float(self.cfg.band_tol),
                                                   aim_frac=self.aim_frac,
                                                   log_tol=float(self.cfg.grow_log_tol))
        # Büyüme kutuyu AŞAMAZ; sınır eşik değil kutudur (ölçüldü: eşiği iki
        # katına çıkarmak kapsamayı %18.6 → %19.9 yapıyor). Kenara dayandıysa
        # ekranda `+` önerilir.
        self._clipped = False
        if self._target is None and self.mask is not None and depth_m is not None:
            ph, pw = depth_m.shape[:2]
            by0, by1, bx0, bx1 = aim_box(ph, pw, self.aim_frac)
            m = self.mask
            self._clipped = bool(m[by0, bx0:bx1].any() or m[by1 - 1, bx0:bx1].any()
                                 or m[by0:by1, bx0].any() or m[by0:by1, bx1 - 1].any())

        self.live_raw = None
        if self.mask is not None and raw_m is not None and raw_m.shape == self.mask.shape:
            # Ham değer kalibre değerle AYNI piksel kümesinden okunur.
            v = raw_m[self.mask]
            if v.size:
                self.live_raw = float(np.median(v))

        if self._pending is None:
            return
        if self.live_raw is not None:
            self._pending["samples"].append(self.live_raw)
        if len(self._pending["samples"]) >= int(self.cfg.capture_frames):
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
        elif key == ord("n"):
            if self._cands and not self.manual:
                self._sel = (self._sel + 1) % len(self._cands)
                self._locked_tid = self._cands[self._sel].tid
                self.message = (f"hedef {self._sel + 1}/{len(self._cands)}: "
                                f"{self._cands[self._sel].label}")
            elif self._small:
                self.message = (f"{len(self._small)} nesne olcume KUCUK — "
                                f"m ile nisan kutusu, +/- ile boyutlandir")
            else:
                self.message = "taninan nesne yok — m ile nisan kutusu"
        elif key == ord("m"):
            self.manual = not self.manual
            self._locked_tid = None
            self.message = ("nisan kutusu: +/- ile boyutlandir" if self.manual
                            else "nesne secimi")
        elif key in (ord("+"), ord("=")):
            # Boyutlandırmak nişan kutusunu kullanmak demektir.
            self.manual = True
            self.aim_frac = min(self.cfg.aim_max, self.aim_frac + self.cfg.aim_step)
            self.message = f"nisan kutusu %{self.aim_frac * 100:.0f}"
        elif key in (ord("-"), ord("_")):
            self.manual = True
            self.aim_frac = max(self.cfg.aim_min, self.aim_frac - self.cfg.aim_step)
            self.message = f"nisan kutusu %{self.aim_frac * 100:.0f}"
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
        if real > MAX_REAL_M:
            self.message = (f"{real:.0f} m fazla buyuk — SANTIM mi yazdiniz? "
                            f"{real/100:.2f} m icin '{real/100:.2f}' yazin")
            return
        if real < MIN_REAL_M:
            self.message = f"{real:.3f} m fazla kucuk (en az {MIN_REAL_M:.2f} m)"
            return
        self._pending = {"real": real, "samples": [], "t0": time.time()}
        self.message = "olculuyor... cismi SABIT tut"

    def _finalize(self) -> None:
        pend, self._pending = self._pending, None
        s = np.array(pend["samples"], dtype=np.float64)
        raw, spread = float(np.median(s)), float(np.std(s))
        real = pend["real"]

        if spread > max(0.04, float(self.cfg.stability_tol) * raw):
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

        # Config'ten elle girilmiş imkânsız noktalar da burada durdurulur.
        bad = [y for y in ys if y > MAX_REAL_M or y < MIN_REAL_M]
        if bad:
            self.message = (f"imkansiz nokta {bad[0]:.2f} m (sinir "
                            f"{MIN_REAL_M}-{MAX_REAL_M} m) — x ile temizleyin")
            return

        for i in range(1, len(xs)):
            if xs[i] - xs[i - 1] < 1e-3:
                self.message = "ayni ham deger iki mesafede — model ayirt edemiyor"
                return
            if ys[i] <= ys[i - 1]:
                self.message = f"tutarsiz nokta ({ys[i-1]:.2f}/{ys[i]:.2f}m) — x=sil"
                return

        # ARALIK GENİŞLİĞİ: birbirine çok yakın iki noktanın gürültülü eğimi
        # ekstrapolasyonla TÜM menzili yönetir ve sahneyi birkaç on santime sıkar.
        span = ys[-1] - ys[0]
        ratio = ys[-1] / max(ys[0], 1e-6)
        if span < 0.30 and ratio < 2.0:
            self.message = (f"aralik cok dar ({span*100:.0f}cm) — "
                            f"birbirinden UZAK mesafeler gir")
            return

        # EĞİM DİRSEĞİ: hatalı ölçüm ham gürültüyü büyütüp okumayı zıplatır.
        kink = worst_kink(pts)
        if kink is not None and kink[0] > MAX_KINK:
            self.message = (f"{kink[1]:.2f}-{kink[2]:.2f}m parcasi {kink[0]:.1f}x "
                            f"dik — model bu araligi ayirmiyor, o noktayi degistir")
            return

        # TEK BAŞINA DİK PARÇA: iki nokta ham eksende çok yakınsa o aralık
        # gürültüyü katlayarak büyütür (ölçüldü: 15.8 eğim → ±25 cm oynama).
        steep = steepest_segment(pts)
        if steep is not None and steep[0] > MAX_STEEP:
            self.message = (f"{steep[1]:.2f}-{steep[2]:.2f}m parcasi genelin "
                            f"{steep[0]:.1f}x'i — bu iki noktayi birbirinden "
                            f"UZAK olcun, okuma yerinde durmaz")
            return

        try:
            write_calib_points(self.config_path, pts)
        except Exception as exc:                        # noqa: BLE001
            self.message = f"YAZMA HATASI: {exc}"
            return
        self.net.set_calibration_table(pts)
        lo, hi = ys[0], ys[-1]
        warn = "  UYARI: 0.5m alti nokta YOK" if lo > 0.5 else ""
        # Menzilin ötesi ekstrapolasyondur; uzak nokta yoksa oda sıkışır.
        if hi < 1.0:
            warn += f"  UYARI: en uzak nokta {hi:.2f}m — otesi TAHMIN"
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

        # Küçük kutular yine çizilir: sistem o cismi görüyor ama hedef saymıyor.
        for d in self._small:
            cv2.rectangle(left, (d.x0 * scale, d.y0 * scale),
                          (d.x1 * scale, d.y1 * scale), (90, 90, 190), 1)
            cv2.putText(left, f"{d.label} kucuk",
                        (d.x0 * scale + 3, min(lh - 4, d.y1 * scale + 12)),
                        cv2.FONT_HERSHEY_PLAIN, 0.9, (120, 120, 210), 1)

        if self._target is None:
            y0, y1, x0, x1 = aim_box(ph, pw, self.aim_frac)
            cv2.rectangle(left, (x0 * scale, y0 * scale), (x1 * scale, y1 * scale),
                          (0, 255, 255), 1)
        else:
            # Seçili nesne sarı, diğer adaylar soluk ve numaralı.
            for i, d in enumerate(self._cands):
                sel = d is self._target
                col = (0, 255, 255) if sel else (130, 130, 130)
                cv2.rectangle(left, (d.x0 * scale, d.y0 * scale),
                              (d.x1 * scale, d.y1 * scale), col, 2 if sel else 1)
                # Etiket alt kenarda: üst şerit kalibrasyon metniyle dolu,
                # lh-18 ise en alttaki fps satırının üstüne binmeyi önler.
                tag = f"{i + 1}.{d.label}" + (" KILITLI" if sel else "")
                cv2.putText(left, tag,
                            (d.x0 * scale + 4, min(lh - 18, d.y1 * scale - 5)),
                            cv2.FONT_HERSHEY_PLAIN, 1.0, col, 1)

        skipped = f"  ({len(self._small)} kucuk atlandi)" if self._small else ""
        if self._target is not None:
            frac = ((self._target.x1 - self._target.x0 + 1)
                    * (self._target.y1 - self._target.y0 + 1)) / float(ph * pw)
            tgt = (f"{self._target.label} [{self._sel + 1}/{len(self._cands)}] "
                   f"%{frac * 100:.0f}{skipped}")
        elif self.manual:
            warn = "  <- KUTU KESIYOR, + ile buyut" if self._clipped else ""
            tgt = f"nisan kutusu %{self.aim_frac * 100:.0f}{skipped}{warn}"
        else:
            tgt = f"taninan nesne yok — nisan kutusu{skipped}"

        # Uyarı `f`'den ÖNCE görünür: hangi noktanın tabloyu bozduğu uygulamayı
        # denemeden anlaşılsın.
        kink = worst_kink(self.points)
        steep = steepest_segment(self.points)
        khint = ""
        if kink is not None and kink[0] > MAX_KINK:
            khint = (f"   dirsek {kink[1]:.2f}-{kink[2]:.2f}m {kink[0]:.1f}x ZIPLAR!")
        elif steep is not None and steep[0] > MAX_STEEP:
            khint = (f"   {steep[1]:.2f}-{steep[2]:.2f}m parcasi {steep[0]:.1f}x "
                     f"dik ZIPLAR!")
        elif kink is not None:
            khint = f"   en dik dirsek {kink[0]:.1f}x"

        shown = "--" if self.live_dist is None else f"{self.live_dist:.2f}m"
        rawv = "--" if self.live_raw is None else f"{self.live_raw:.2f}"
        pend = "" if self._pending is None else \
            f"  [olculuyor {len(self._pending['samples'])}/{self.cfg.capture_frames}]"
        tbl = getattr(self.net, "table_size", 0)
        lines = [
            f"KALIBRASYON   c/ESC=cik   r=geri al   tablo:{tbl} nokta",
            f"hedef: {tgt}",
            "n=sonraki nesne   m=nisan kutusu   +/- =kutu boyutu",
            f"konturdaki cisim: {shown}   (ham {rawv}){pend}",
            f"gercek mesafe(m): {self.buffer}_",
            "ENTER=kaydet   x=noktalari sil   f=uygula",
            f"nokta:{len(self.points)}{khint}",
            f"{self.message}",
        ]
        y = 30
        for i, ln in enumerate(lines):
            (tw, th), _ = cv2.getTextSize(ln, cv2.FONT_HERSHEY_PLAIN, 1.1, 1)
            cv2.rectangle(view, (5, y - th - 4), (13 + tw, y + 5), (18, 18, 18), -1)
            col = (0, 255, 255) if i != len(lines) - 1 else (140, 255, 140)
            cv2.putText(view, ln, (9, y), cv2.FONT_HERSHEY_PLAIN, 1.1, col, 1)
            y += th + 9
