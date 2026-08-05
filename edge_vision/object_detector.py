"""YOLO nesne tanıma — sistemin gördüğü şeyin ADINI (ve maskesini) verir.

Derinlik ağı "ne kadar uzakta" sorusunu cevaplar ama "ne" olduğunu bilmez.
Bu modül COCO sınıflarıyla etiket üretir; kutu/maske metrik derinlik haritasıyla
füzyona girer ve her nesne için ad + metre çifti oluşur.

İki model biçimi desteklenir, hangisinin yüklendiği oturumdan anlaşılır:

    yolo11n.onnx      (1, 84, 8400)                        sınıf skorları + NMS
    yolo26n-seg.onnx  (1, 300, 38) + (1, 32, 160, 160)     uçtan uca, NMS yok, maske

Segmentasyon maskesi mesafe kestirimindeki tahmini ortadan kaldırır: kutu her
zaman arka plan da içerir (ölçüldü: bir insanda kutunun yalnızca %54'ü cisim).

Ağ, derinlik ağı gibi AYRI BİR ARKA PLAN İŞ PARÇACIĞINDA koşar: ana döngü
kamera hızında akmaya devam eder, tespitler hazır olduğunda yayınlanır.
"""

from __future__ import annotations

import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .config import TrackConfig, YoloConfig
from .utils import require_model

# COCO 80 sınıfı, model çıktı sırasıyla (indeks = class_id).
COCO_TR = [
    "insan", "bisiklet", "araba", "motosiklet", "uçak", "otobüs", "tren",
    "kamyon", "tekne", "trafik ışığı", "yangın musluğu", "dur tabelası",
    "parkmetre", "bank", "kuş", "kedi", "köpek", "at", "koyun", "inek", "fil",
    "ayı", "zebra", "zürafa", "sırt çantası", "şemsiye", "el çantası", "kravat",
    "bavul", "frizbi", "kayak", "snowboard", "top", "uçurtma", "beyzbol sopası",
    "beyzbol eldiveni", "kaykay", "sörf tahtası", "tenis raketi", "şişe",
    "şarap kadehi", "bardak", "çatal", "bıçak", "kaşık", "kase", "muz", "elma",
    "sandviç", "portakal", "brokoli", "havuç", "sosisli", "pizza", "donut",
    "pasta", "sandalye", "koltuk", "saksı bitkisi", "yatak", "yemek masası",
    "tuvalet", "televizyon", "dizüstü", "fare", "kumanda", "klavye", "telefon",
    "mikrodalga", "fırın", "ekmek kızartıcı", "lavabo", "buzdolabı", "kitap",
    "saat", "vazo", "makas", "oyuncak ayı", "saç kurutma", "diş fırçası",
]


@dataclass(frozen=True)
class Detection:
    class_id: int
    label: str
    conf: float
    x0: int
    y0: int
    x1: int
    y1: int
    distance_m: float = float("inf")   # derinlik füzyonundan; bilinmiyorsa inf
    tid: int = -1                      # iz kimliği; kareler arası aynı cisim
    # compare=False: dizi karşılaştırması dataclass eşitliğini patlatır.
    mask: Optional[np.ndarray] = field(default=None, compare=False, repr=False)

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "class_id": self.class_id,
            "conf": round(self.conf, 3),
            "box": [self.x0, self.y0, self.x1, self.y1],
            "distance_m": (round(self.distance_m, 3)
                           if np.isfinite(self.distance_m) else None),
            "track_id": self.tid,
            # Maskenin kendisi JSON'a konmaz (on binlerce piksel), alanı yeter.
            "mask_px": None if self.mask is None else int(self.mask.sum()),
        }


class YoloDetector:
    def __init__(self, cfg: YoloConfig):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = cfg.threads
        avail = set(ort.get_available_providers())
        providers = [p for p in cfg.providers if p in avail] or ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(require_model(cfg.model), so,
                                            providers=providers)
        self.active_provider = self.session.get_providers()[0]
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        # Model girdi boyutu SABİTSE (yolo11n dışa aktarımı 640x640) config'teki
        # değer değil o kullanılır; uyumsuz boyut ONNX'te doğrudan hata verirdi.
        static = inp.shape[-1] if isinstance(inp.shape[-1], int) else None
        self.size = int(static or cfg.input_size)
        self.conf_thres = float(cfg.conf_thres)
        self.iou_thres = float(cfg.iou_thres)
        self.max_det = int(cfg.max_det)
        self.names = list(cfg.classes) if cfg.classes else COCO_TR

        # Biçim oturumdan anlaşılır, config'te bayrak tutulmaz: yanlış ayarlanmış
        # bir bayrak sessizce saçma kutular üretirdi.
        outs = self.session.get_outputs()
        self.has_masks = len(outs) >= 2 and len(outs[1].shape) == 4
        o0 = outs[0].shape
        self.end2end = len(o0) == 3 and isinstance(o0[2], int) and o0[2] <= 64

        self._latest_frame: Optional[np.ndarray] = None
        self._dets: List[Detection] = []
        self._lock = threading.Lock()
        self._frame_lock = threading.Lock()
        self._stopped = False
        self._thread: Optional[threading.Thread] = None
        self._infer_ms = 0.0
        self._update_rate = 0.0
        self._last_pub = 0.0
        self._version = 0

    # --- arka plan iş parçacığı ---

    def start(self) -> "YoloDetector":
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def submit(self, bgr: np.ndarray) -> None:
        with self._frame_lock:
            self._latest_frame = bgr.copy()

    def _loop(self) -> None:
        while not self._stopped:
            with self._frame_lock:
                frame = self._latest_frame
                self._latest_frame = None
            if frame is None:
                time.sleep(0.002)
                continue
            dets = self._infer(frame)
            with self._lock:
                self._dets = dets
                self._version += 1
            now = time.perf_counter()
            if self._last_pub:
                dt = now - self._last_pub
                if dt > 0:
                    self._update_rate = 0.8 * self._update_rate + 0.2 * (1.0 / dt)
            self._last_pub = now

    def stop(self) -> None:
        self._stopped = True
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    # --- çıkarım ---

    def _preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, float, int, int]:
        """Letterbox: en-boy oranı korunur, kalan alan griyle doldurulur.

        Oran bozulursa kutular da bozulur; YOLO eğitimi letterbox varsayar.
        """
        h0, w0 = frame.shape[:2]
        s = self.size / max(h0, w0)
        nh, nw = int(round(h0 * s)), int(round(w0 * s))
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.size, self.size, 3), 114, dtype=np.uint8)
        top, left = (self.size - nh) // 2, (self.size - nw) // 2
        canvas[top:top + nh, left:left + nw] = resized
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = rgb.transpose(2, 0, 1)[None]
        return np.ascontiguousarray(x), s, left, top

    def _infer(self, frame: np.ndarray) -> List[Detection]:
        x, s, pad_l, pad_t = self._preprocess(frame)
        t0 = time.perf_counter()
        outs = self.session.run(None, {self.input_name: x})
        self._infer_ms = (time.perf_counter() - t0) * 1000.0
        geom = (s, pad_l, pad_t, frame.shape[0], frame.shape[1])
        if self.end2end:
            return self._decode_end2end(outs, geom)
        return self._decode_classic(outs[0], geom)

    def _decode_end2end(self, outs, geom) -> List[Detection]:
        """YOLO26: `(1, N, 4+1+1+nm)`, NMS YOK.

        Satır düzeni: `[x0, y0, x1, y1, güven, sınıf, maske katsayıları...]`
        (ultralytics'in kendi çıktısıyla karşılaştırılarak doğrulandı).
        """
        s, pad_l, pad_t, h0, w0 = geom
        pred = outs[0][0]
        keep = pred[:, 4] >= self.conf_thres
        if not keep.any():
            return []
        rows = pred[keep][:self.max_det]
        protos = outs[1][0] if self.has_masks else None

        dets: List[Detection] = []
        for row in rows:
            x0 = int(np.clip((row[0] - pad_l) / s, 0, w0 - 1))
            y0 = int(np.clip((row[1] - pad_t) / s, 0, h0 - 1))
            x1 = int(np.clip((row[2] - pad_l) / s, 0, w0 - 1))
            y1 = int(np.clip((row[3] - pad_t) / s, 0, h0 - 1))
            if x1 <= x0 or y1 <= y0:
                continue
            c = int(row[5])
            label = self.names[c] if 0 <= c < len(self.names) else str(c)
            mask = None
            if protos is not None:
                mask = self._build_mask(row[6:], protos, geom, (x0, y0, x1, y1))
            dets.append(Detection(c, label, float(row[4]), x0, y0, x1, y1,
                                  mask=mask))
        return dets

    def _build_mask(self, coeffs, protos, geom, box) -> Optional[np.ndarray]:
        """Maske katsayıları × prototipler → kare boyutunda bool maske.

        Prototip ızgarası letterbox tuvalinin ölçekli halidir. Tüm maskeyi
        büyütüp kırpmak yerine ÖNCE prototip uzayında kırpılır: aynı sonuç,
        dörtte bir maliyet.
        """
        s, pad_l, pad_t, h0, w0 = geom
        nm, gh, gw = protos.shape
        if coeffs.size != nm:
            return None
        m = coeffs @ protos.reshape(nm, -1)
        m = 1.0 / (1.0 + np.exp(-m.reshape(gh, gw)))

        r = gh / float(self.size)                      # prototip / letterbox oranı
        ry0, ry1 = int(pad_t * r), int(round((pad_t + h0 * s) * r))
        rx0, rx1 = int(pad_l * r), int(round((pad_l + w0 * s) * r))
        sub = m[max(0, ry0):max(ry0 + 1, ry1), max(0, rx0):max(rx0 + 1, rx1)]
        if sub.size == 0:
            return None
        full = cv2.resize(sub.astype(np.float32), (w0, h0),
                          interpolation=cv2.INTER_LINEAR) > 0.5

        # Kutu dışını kırp: prototip maskesi komşu örneklere taşabiliyor.
        out = np.zeros_like(full)
        x0, y0, x1, y1 = box
        out[y0:y1 + 1, x0:x1 + 1] = full[y0:y1 + 1, x0:x1 + 1]
        return out

    def _decode_classic(self, out, geom) -> List[Detection]:
        """YOLO11: `(1, 84, 8400)` — sınıf skorları, NMS gerekir."""
        s, pad_l, pad_t, h0, w0 = geom
        pred = out[0].T                                 # (8400, 84): 4 kutu + 80 sınıf
        scores = pred[:, 4:]
        cls = scores.argmax(1)
        conf = scores[np.arange(scores.shape[0]), cls]
        keep = conf >= self.conf_thres
        if not keep.any():
            return []
        boxes = pred[keep, :4]                          # cx, cy, w, h (letterbox)
        conf = conf[keep]
        cls = cls[keep]

        # cxcywh -> xywh (sol-üst köşe); NMSBoxes bu biçimi bekler.
        xy = boxes[:, :2] - boxes[:, 2:] / 2
        nms_boxes = np.concatenate([xy, boxes[:, 2:]], axis=1).tolist()
        idx = cv2.dnn.NMSBoxes(nms_boxes, conf.tolist(), self.conf_thres, self.iou_thres)
        if len(idx) == 0:
            return []
        idx = np.array(idx).flatten()[:self.max_det]

        dets: List[Detection] = []
        for i in idx:
            cx, cy, bw, bh = boxes[i]
            # Letterbox koordinatından kare koordinatına: dolguyu çıkar, ölçeği böl.
            x0 = int(np.clip((cx - bw / 2 - pad_l) / s, 0, w0 - 1))
            x1 = int(np.clip((cx + bw / 2 - pad_l) / s, 0, w0 - 1))
            y0 = int(np.clip((cy - bh / 2 - pad_t) / s, 0, h0 - 1))
            y1 = int(np.clip((cy + bh / 2 - pad_t) / s, 0, h0 - 1))
            if x1 <= x0 or y1 <= y0:
                continue
            c = int(cls[i])
            label = self.names[c] if 0 <= c < len(self.names) else str(c)
            dets.append(Detection(c, label, float(conf[i]), x0, y0, x1, y1))
        return dets

    # --- ana döngünün okuduğu (bloke olmayan) arayüz ---

    @property
    def version(self) -> int:
        """Yeni çıkarım sayacı. Ana döngü ağdan hızlı döndüğü için aynı liste
        tekrar okunur; izleyici onu iki kez oylamamalı."""
        return self._version

    @property
    def update_rate(self) -> float:
        return self._update_rate

    @property
    def infer_ms(self) -> float:
        return self._infer_ms

    def get_detections(self) -> List[Detection]:
        with self._lock:
            return list(self._dets)


def _iou(a, b) -> float:
    ix0, iy0 = max(a.x0, b.x0), max(a.y0, b.y0)
    ix1, iy1 = min(a.x1, b.x1), min(a.y1, b.y1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = max(1, (a.x1 - a.x0) * (a.y1 - a.y0))
    area_b = max(1, (b.x1 - b.x0) * (b.y1 - b.y0))
    return inter / float(area_a + area_b - inter)


class _Track:
    """Kareler arası aynı cisim: kutu + sınıf oyları + yumuşatılmış mesafe."""

    __slots__ = ("tid", "box", "votes", "shown", "conf", "hits", "misses",
                 "dist", "mask")

    def __init__(self, tid: int, det: Detection, vote_frames: int):
        self.tid = tid
        self.box = [det.x0, det.y0, det.x1, det.y1]
        self.votes = deque([det.class_id], maxlen=vote_frames)
        self.shown = det.class_id
        self.conf = det.conf
        self.hits = 1
        self.misses = 0
        self.dist = float("inf")
        # Maske yumuşatılmaz: ortalanabilir bir büyüklük değil.
        self.mask = det.mask

    def vote(self, class_id: int, switch_ratio: float) -> None:
        """Oy ekler; gösterilen etiketi ancak AÇIK ÜSTÜNLÜKTE değiştirir.

        Düz çoğunluk yetmiyor: pencere kayarken çoğunluk ara ara el değiştiriyor
        (ölçüldü: 80 karede 3 kez). Üstünlük şartı bunu 1'e indirir; gerçek bir
        sınıf değişimi pencereyi doldurunca yine geçer (~0.5 s).
        """
        self.votes.append(class_id)
        cnt = Counter(self.votes)
        top, top_n = cnt.most_common(1)[0]
        if top == self.shown:
            return
        if top_n >= switch_ratio * cnt.get(self.shown, 0):
            self.shown = top

    # _iou() ile aynı arayüz: iz, tespitle doğrudan karşılaştırılabilsin.
    @property
    def x0(self) -> int:
        return self.box[0]

    @property
    def y0(self) -> int:
        return self.box[1]

    @property
    def x1(self) -> int:
        return self.box[2]

    @property
    def y1(self) -> int:
        return self.box[3]

    def as_detection(self, names=None) -> Detection:
        c = self.shown
        # Adlar tespitlerden öğrenilir: model COCO da olabilir, açık sözlüklü de.
        label = (names or {}).get(c) or (COCO_TR[c] if 0 <= c < len(COCO_TR)
                                         else str(c))
        return Detection(c, label, self.conf, self.box[0], self.box[1],
                         self.box[2], self.box[3], self.dist, self.tid,
                         mask=self.mask)


class DetectionTracker:
    """YOLO tespitlerini kareler arasında eşler ve etiketi OYLAMAYLA belirler.

    YOLO her kareyi bağımsız işler; iki sınıfın skoru yakınsa (bardak/insan)
    etiket kare kare zıplar. Son N karenin çoğunluk sınıfı bunu keser.

      * `min_hits` — iz gösterilmeden önce bu kadar kare görülmeli (hayalet kutu)
      * `max_miss` — kısa kayboluşta iz silinmez, kutu korunur (yanıp sönme)
      * ölü bantlı EMA — cisim dururken mesafe sabit, hareket edince hızlı takip
    """

    def __init__(self, cfg: TrackConfig):
        self.enabled = bool(cfg.enabled)
        self.vote_frames = max(1, int(cfg.vote_frames))
        self.min_hits = max(1, int(cfg.min_hits))
        self.max_miss = max(0, int(cfg.max_miss))
        self.iou_match = float(cfg.iou_match)
        self.switch_ratio = float(cfg.switch_ratio)
        self.box_ema = float(cfg.box_ema)
        self.deadband = float(cfg.deadband_m)
        self.a_still = float(cfg.alpha_still)
        self.a_move = float(cfg.alpha_move)
        self._tracks: List[_Track] = []
        self._next_id = 1
        self._names: dict = {}

    def update(self, dets: List[Detection]) -> None:
        """Yeni bir YOLO çıkarımı geldiğinde çağrılır (kare başına DEĞİL)."""
        if not self.enabled:
            return
        for d in dets:
            self._names.setdefault(d.class_id, d.label)
        used = set()
        for tr in self._tracks:
            best, best_iou = -1, self.iou_match
            for i, d in enumerate(dets):
                if i in used:
                    continue
                v = _iou(tr, d)
                if v > best_iou:
                    best, best_iou = i, v
            if best < 0:
                tr.misses += 1
                continue
            d = dets[best]
            used.add(best)
            a = self.box_ema
            tr.box = [int(round(tr.box[k] + a * (v - tr.box[k])))
                      for k, v in enumerate((d.x0, d.y0, d.x1, d.y1))]
            tr.vote(d.class_id, self.switch_ratio)
            tr.conf = d.conf
            tr.mask = d.mask
            tr.hits += 1
            tr.misses = 0

        for i, d in enumerate(dets):
            if i not in used:
                self._tracks.append(_Track(self._next_id, d, self.vote_frames))
                self._next_id += 1

        self._tracks = [t for t in self._tracks if t.misses <= self.max_miss]

    def set_distance(self, tid: int, dist: float) -> float:
        """Ölü bantlı EMA: küçük oynamalar bastırılır, gerçek hareket geçer."""
        for tr in self._tracks:
            if tr.tid != tid:
                continue
            if not np.isfinite(dist):
                tr.dist = dist
                return dist
            if not np.isfinite(tr.dist):
                tr.dist = dist
            else:
                a = self.a_still if abs(dist - tr.dist) < self.deadband else self.a_move
                tr.dist += a * (dist - tr.dist)
            return tr.dist
        return dist

    def current(self) -> List[Detection]:
        """Gösterilecek izler; kısa kayboluşta kutu tutulur (yanıp sönmeyi önler)."""
        return [t.as_detection(self._names) for t in self._tracks
                if t.hits >= self.min_hits and t.misses <= self.max_miss]


def fuse_distance(det: Detection, depth_m: np.ndarray,
                  ground_mask: Optional[np.ndarray] = None) -> float:
    """Nesnenin metrik mesafesini kestirir.

    Segmentasyon maskesi varsa doğrudan o kullanılır. Maske yoksa (kutu-only
    modeller) kutunun MERKEZ ÇEKİRDEĞİ (orta %50) örneklenir: köşeler neredeyse
    her zaman arka planı içerir, nesnenin gövdesi ortadadır. Zemin maskesi
    yalnızca bu ikinci yolda uygulanır.
    """
    h, w = depth_m.shape[:2]
    if det.mask is not None and det.mask.shape == depth_m.shape:
        # Zemin filtresi UYGULANMAZ: maske cismin piksellerini zaten veriyor,
        # düzlem filtresi (ölçüldü) maskenin %63'ünü siliyordu.
        sel = det.mask & (depth_m > 1e-3)
        vals = depth_m[sel]
        if vals.size >= 10:
            return float(np.median(vals))
        # Maske boşsa (geçersiz derinlik) kutu yöntemine düş.

    x0, y0 = max(0, det.x0), max(0, det.y0)
    x1, y1 = min(w, det.x1 + 1), min(h, det.y1 + 1)
    if x1 <= x0 or y1 <= y0:
        return float("inf")

    mx, my = (x1 - x0) // 4, (y1 - y0) // 4
    cx0, cy0, cx1, cy1 = x0 + mx, y0 + my, x1 - mx, y1 - my
    if cx1 <= cx0 or cy1 <= cy0:
        cx0, cy0, cx1, cy1 = x0, y0, x1, y1
    core = depth_m[cy0:cy1, cx0:cx1]
    valid = core > 1e-3
    if ground_mask is not None:
        valid &= ~ground_mask[cy0:cy1, cx0:cx1]
    vals = core[valid]
    if vals.size < 10:
        # Çekirdek boşsa (küçük kutu / tamamı zemin) tüm kutuya düş.
        sub = depth_m[y0:y1, x0:x1]
        vals = sub[sub > 1e-3]
        if vals.size < 5:
            return float("inf")
    return float(np.median(vals))
