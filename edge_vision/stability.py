"""Ani değişim algılama: ışık sıçraması, bulanıklık, sahne kesmesi.

Derinlik ağı ve YOLO her kareyi BAĞIMSIZ yorumlar; görüntü bir anda
değişince (otomatik pozlama ışığı sıçratır, kamera sarsılıp bulanıklaşır,
robot başka odaya döner) çıktılar da bir anda kayar. Bu modül o anları
görüntünün kendisinden ucuzca (80x60 gri, <0.5 ms) tespit eder ki hat:

  * kararsız karede ölçek çapasını GÜNCELLEMESİN (yanlış referansa kaymasın),
  * iz mesafelerinde büyük sıçramayı hemen kabul etmesin,
  * GERÇEK sahne değişiminde (kesme) çapayı yeni sahneye göre yeniden kursun.

Eşikler sabit değil: her öznitelik için çevrimiçi (EWMA) ortalama/varyans
öğrenilir, z-skoru bu ortama göre hesaplanır. Yani "normal" titreşim
seviyesini sistem kendisi öğrenir (çevrimiçi, denetimsiz anomali tespiti);
loş ve gürültülü bir odada eşik kendiliğinden gevşer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2
import numpy as np

from .log import get_logger

log = get_logger(__name__)


class OnlineStat:
    """EWMA ortalama/varyans; z-skoru ortamın kendi gürültüsüne göre ölçer."""

    def __init__(self, alpha: float = 0.02, min_std: float = 1e-3):
        self.alpha = float(alpha)
        self.min_std = float(min_std)
        self.mean: Optional[float] = None
        self.var = 0.0
        self.n = 0

    def z(self, x: float) -> float:
        if self.mean is None or self.n < 10:
            return 0.0
        return (x - self.mean) / max(np.sqrt(self.var), self.min_std)

    def update(self, x: float) -> None:
        self.n += 1
        if self.mean is None:
            self.mean = float(x)
            return
        d = x - self.mean
        self.mean += self.alpha * d
        self.var = (1 - self.alpha) * (self.var + self.alpha * d * d)


@dataclass
class SceneState:
    stable: bool = True
    cut: bool = False                       # bu karede sahne kesmesi onaylandı
    events: List[str] = field(default_factory=list)
    features: Dict[str, float] = field(default_factory=dict)
    unstable_for: int = 0                   # ardışık kararsız kare sayısı


class SceneMonitor:
    def __init__(self, cfg):
        self.enabled = bool(cfg.enabled)
        self.light_jump = float(cfg.light_jump)
        self.blur_ratio = float(cfg.blur_ratio)
        self.cut_corr = float(cfg.cut_corr)
        self.cut_persist = max(1, int(cfg.cut_persist))
        self.z_max = float(cfg.z_max)
        self.settle = max(0, int(cfg.settle_frames))
        self._prev_small: Optional[np.ndarray] = None
        self._prev_lum: Optional[float] = None
        self._ref: Optional[np.ndarray] = None
        self._cand: Optional[np.ndarray] = None
        self._cand_n = 0
        self._settle_left = 0
        self._unstable_for = 0
        self._stats = {k: OnlineStat(float(cfg.learn_rate))
                       for k in ("diff", "dlum", "sharp")}
        self._last_events: tuple = ()
        self.n_cuts = 0

    @staticmethod
    def _thumb(small: np.ndarray) -> np.ndarray:
        """20x15 bulanık küçük resim, sıfır ortalama / birim sapma.

        Parlaklık ve kontrasttan bağımsızdır: pozlama değişimi "yeni sahne"
        sayılmaz, yalnızca YAPI değişirse benzerlik düşer.
        """
        t = cv2.resize(small, (20, 15), interpolation=cv2.INTER_AREA).astype(np.float32)
        t -= t.mean()
        return t / max(float(t.std()), 1.0)

    @staticmethod
    def _corr(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.mean(a * b))          # normalize çapraz korelasyon (NCC)

    def update(self, bgr: np.ndarray) -> SceneState:
        if not self.enabled:
            return SceneState()
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (80, 60), interpolation=cv2.INTER_AREA)
        lum = float(small.mean())
        sharp = float(cv2.Laplacian(small, cv2.CV_32F).var())
        thumb = self._thumb(small)

        feats = {"lum": lum, "sharp": sharp}
        events: List[str] = []
        cut = False

        if self._prev_small is None:
            self._prev_small, self._prev_lum, self._ref = small, lum, thumb
            return SceneState(features=feats)

        # Parlaklık normalize edilmiş fark: pozlama değişimi "hareket" sayılmasın.
        a = small.astype(np.float32) / max(lum, 1.0)
        b = self._prev_small.astype(np.float32) / max(self._prev_lum, 1.0)
        diff = float(np.mean(np.abs(a - b)))
        dlum = abs(lum - self._prev_lum) / max(self._prev_lum, 1.0)
        corr = self._corr(thumb, self._ref)
        feats.update(diff=diff, dlum=dlum, corr=corr)

        z = {k: self._stats[k].z(v) for k, v in (("diff", diff), ("dlum", dlum),
                                                    ("sharp", sharp))}
        feats.update({f"z_{k}": v for k, v in z.items()})

        if dlum > self.light_jump or z["dlum"] > self.z_max:
            events.append("isik")
        sm = self._stats["sharp"]
        if sm.n >= 10 and sharp < self.blur_ratio * sm.mean:
            events.append("bulanik")
        if z["diff"] > self.z_max:
            events.append("sarsinti")

        # Sahne kesmesi: referanstan uzak AMA kendi içinde tutarlı yeni sahne,
        # birkaç kare boyunca. Geçen el / kısa örtme bunu sağlamaz.
        if corr < self.cut_corr:
            if self._cand is not None and self._corr(thumb, self._cand) > 0.8:
                self._cand_n += 1
            else:
                self._cand_n = 1
            self._cand = thumb
            if self._cand_n >= self.cut_persist:
                cut = True
                self.n_cuts += 1
                self._ref = thumb
                self._cand, self._cand_n = None, 0
                events.append("kesme")
            else:
                events.append("degisim")
        else:
            self._cand, self._cand_n = None, 0
            # Referans yavaşça güncellenir: kademeli değişim kesme sayılmaz.
            self._ref = 0.95 * self._ref + 0.05 * thumb

        # Anomali olmayan karelerden öğren: sıçrama "normal"i bozmasın.
        if not events:
            for k, v in (("diff", diff), ("dlum", dlum), ("sharp", sharp)):
                self._stats[k].update(v)

        if events:
            self._settle_left = self.settle
        elif self._settle_left > 0:
            self._settle_left -= 1
        stable = not events and self._settle_left == 0
        self._unstable_for = 0 if stable else self._unstable_for + 1

        key = tuple(events)
        if key != self._last_events:
            if events:
                lvl = log.warning if cut else log.info
                lvl("sahne: %s  (dlum=%.2f diff=%.3f corr=%.2f sharp=%.0f)",
                    ",".join(events), dlum, diff, corr, sharp)
            else:
                log.debug("sahne: kararli")
            self._last_events = key

        self._prev_small, self._prev_lum = small, lum
        return SceneState(stable=stable, cut=cut, events=events, features=feats,
                          unstable_for=self._unstable_for)
