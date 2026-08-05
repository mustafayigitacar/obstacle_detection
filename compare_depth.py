"""İki derinlik motorunu karşılaştırır ve hangisinin metrik olarak doğru
olduğunu METRE KULLANMADAN ayırt etmeyi sağlar.

    python compare_depth.py

**Yöntem:** Sistem zaten zemini bir düzlem olarak uyduruyor ve o düzlemden
kameranın yerden yüksekliğini çıkarıyor (`GroundRemover.camera_height_m`).
Bir derinlik modelinin ölçeği k katı yanlışsa, bulduğu kamera yüksekliği de
k katı yanlış çıkar. Kameranızın yerden yüksekliğini biliyorsunuz — o yüzden
bu tek sayı, iki motoru ayırt etmek için yeterlidir.

**Nasıl kullanılır:** Kamerayı ZEMİNİN GÖRÜNDÜĞÜ bir yöne çevirin (öne doğru,
yer görüş alanının alt kısmında olsun) ve çalıştırın. Çıkan kamera yüksekliği
gerçeğe daha yakın olan motor, ölçeği doğru olandır.

Karşılaştırma ayrıca şunları raporlar: çıkarım süresi, harita tazeleme hızı,
zemin kilidi oranı ve düzlem iç-nokta oranı (yüzeyin ne kadar düz çıktığı).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from edge_vision import ThreadedCamera, VisionPipeline, load_config

BACKENDS = (
    # (backend, model, kalibrasyon uygulanır mı)
    ("dav2", "models/depth_anything_v2_metric_indoor.onnx", True),
    ("yolo26", "models/yolo26n-depth.onnx", False),
)
SECONDS = 12.0


def run(backend: str, model: str, keep_calib: bool, seconds: float) -> dict:
    cfg = load_config("config.yaml")
    cfg.depth.backend = backend
    cfg.depth.model = model
    if not keep_calib:
        # yolo26 doğrudan metre üretir: kalibrasyon uygulanırsa ÜSTÜNE biner.
        cfg.depth.calib_points = []
        cfg.depth.calib_scale = 1.0
        cfg.depth.calib_offset = 0.0

    pipe = VisionPipeline(cfg).start()
    heights, inliers, locks, ms, rate, near = [], [], [], [], [], []
    with ThreadedCamera(cfg.camera.source, cfg.camera.capture_width,
                        cfg.camera.capture_height, cfg.camera.fps) as cam:
        t0, last = time.time(), -1
        while time.time() - t0 < seconds:
            fid, frame = cam.read()
            if frame is None or fid == last:
                time.sleep(0.005)
                continue
            last = fid
            _, _, dbg = pipe.process(fid, frame, want_edges=False)
            if dbg["depth_m"] is None:
                continue
            locks.append(bool(dbg["ground_locked"]))
            h = pipe.ground.camera_height_m
            if dbg["ground_locked"] and h:
                heights.append(h)
                inliers.append(pipe.ground.inlier_ratio)
            v = dbg["depth_m"][dbg["depth_m"] > 1e-3]
            if v.size:
                near.append(float(np.percentile(v, 2)))
            ms.append(pipe.depth.infer_ms)
            rate.append(pipe.depth.update_rate)
    prov = pipe.depth.active_provider
    pipe.stop()
    # Isinma atlanir: ilk cikarimlar oturum kurulumunu da tasir, tazeleme hizi
    # ise EMA oldugu icin sifirdan baslar — medyanini almak sistematik olarak
    # dusuk gosterirdi.
    warm = min(30, len(ms) // 3)
    ms = ms[warm:] or ms
    rate = rate[warm:] or rate
    return {
        "provider": prov, "frames": len(ms),
        "ms": float(np.median(ms)) if ms else float("nan"),
        "fps": float(rate[-1]) if rate else float("nan"),
        "lock": float(np.mean(locks)) if locks else 0.0,
        "height": float(np.median(heights)) if heights else float("nan"),
        "height_std": float(np.std(heights)) if heights else float("nan"),
        "inlier": float(np.mean(inliers)) if inliers else float("nan"),
        "near": float(np.mean(near)) if near else float("nan"),
    }


def main() -> int:
    print("Kamerayi ZEMININ GORUNDUGU bir yone cevirin (yer, goruntunun alt")
    print("kisminda olsun). Her motor icin ~12 saniye olculecek.\n")
    out = {}
    for backend, model, keep in BACKENDS:
        if not Path(model).exists():
            print(f"[atlandi] {backend}: {model} yok")
            continue
        print(f"olculuyor: {backend} ...", flush=True)
        out[backend] = run(backend, model, keep, SECONDS)

    if not out:
        print("Hicbir model bulunamadi.")
        return 1

    print(f"\n{'':>10} {'saglayici':>10} {'cikarim':>9} {'tazeleme':>9} "
          f"{'zemin kilidi':>13} {'ic-nokta':>9} {'KAMERA YUKSEKLIGI':>19} {'en yakin':>9}")
    for b, r in out.items():
        prov = r["provider"].replace("ExecutionProvider", "")
        hh = ("--" if not np.isfinite(r["height"])
              else f"{r['height']:.2f} +-{r['height_std']*100:.0f}cm")
        print(f"{b:>10} {prov:>10} {r['ms']:>7.1f}ms {r['fps']:>6.1f}fps "
              f"{r['lock']*100:>12.0f}% {r['inlier']:>8.2f} {hh:>19} "
              f"{r['near']:>7.2f}m")

    print("\nKARAR: kameranizin yerden yuksekligini biliyorsunuz.")
    print("Yukaridaki 'KAMERA YUKSEKLIGI' degeri gercege yakin olan motorun")
    print("olcegi dogrudur; digeri o oranda sapmali okur.")
    print("\nDIKKAT: 'zemin kilidi' dusukse ya da 'ic-nokta' 0.5 altindaysa")
    print("uydurulan duzlem ZEMIN olmayabilir (masa yuzeyi / duvar). O satirin")
    print("yukseklik degeri anlamsizdir — kamerayi yere dogrultup tekrarlayin.")
    valid = {b: r["height"] for b, r in out.items() if np.isfinite(r["height"])}
    if len(valid) == 2:
        (b1, h1), (b2, h2) = valid.items()
        print(f"\noran: {b1} / {b2} = {h1 / max(h2, 1e-6):.2f}x  "
              f"(hangisi dogruysa digeri bu katsayiyla sapiyor)")
    elif len(valid) < 2:
        print("\nUYARI: en az bir motorda zemin duzlemi kilitlenmedi — kamerayi")
        print("zeminin GORUNDUGU bir yone cevirip tekrar calistirin.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
