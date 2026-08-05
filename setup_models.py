"""Model dosyalarini hazirlar — kod ya da config duzenlemeden, tek komutla.

    python setup_models.py                 # tanima + el modeli + derinlik kontrolu
    python setup_models.py --depth yolo26  # derinligi de uret, config'i ayarla
    python setup_models.py --yoloe         # acik sozluklu model (kapi, duvar...)
    python setup_models.py --no-hand       # el modelini indirme

Depoda model dosyasi yoktur (`.gitignore`): ~130 MB'lik ikili dosya git'e girmez.
Bu betik eksik olanlari uretir/indirir ve `models/` altina yerlestirir. Zaten
varsa dokunmaz, tekrar tekrar calistirilabilir.

Tek istisna Depth Anything V2: ultralytics'in modeli degildir, disa aktarim
komutu yoktur. Yoksa betik iki secenegi de yazar ve sessizce cokmez.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
CONFIG = ROOT / "config.yaml"

DAV2 = "depth_anything_v2_metric_indoor.onnx"

# El modeli MediaPipe'in resmi dagitimindan gelir (~7.5 MB); ONNX degildir,
# ultralytics ile uretilemez.
HAND_FILE = "hand_landmarker.task"
HAND_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/1/hand_landmarker.task")

# ONNX adi -> (ultralytics agirligi, disa aktarim boyutu)
# yolo26n-depth 4:3 aktarilir: kare letterbox'in gri dolgusu modelin sahne
# yorumunu degistirip degerleri kaydiriyor (olculdu: p50 1.82 -> 2.46).
EXPORTS = {
    "yolo26n-seg.onnx": ("yolo26n-seg.pt", 640),
    "yolo26n-depth.onnx": ("yolo26n-depth.pt", [480, 640]),
}

YOLOE_CLASSES = [
    "person", "door", "window", "wall", "stairs", "television", "chair",
    "couch", "dining table", "bottle", "cup", "laptop", "backpack",
    "potted plant", "refrigerator", "cabinet",
]
YOLOE_TR = ("[insan, kapi, pencere, duvar, merdiven, televizyon, sandalye,\n"
            "            koltuk, yemek masasi, sise, bardak, dizustu, "
            "sirt cantasi,\n            saksi bitkisi, buzdolabi, dolap]")


def _ultralytics():
    try:
        import ultralytics  # noqa: F401
    except ImportError:
        sys.exit("HATA: ultralytics kurulu degil.\n"
                 "  pip install ultralytics\n"
                 "(yalnizca model uretmek icin gerekir, calisma zamaninda degil)")
    return __import__("ultralytics", fromlist=["YOLO", "YOLOE"])


def _place(produced: Path, target: Path) -> None:
    """Uretilen ONNX'i models/ altina tasir; ultralytics onu koke birakir."""
    shutil.move(str(produced), str(target))
    weight = ROOT / produced.name.replace(".onnx", ".pt")
    if weight.exists():
        weight.unlink()                      # ara agirlik dosyasi birakilmaz


def export_model(onnx_name: str) -> bool:
    target = MODELS / onnx_name
    if target.exists():
        print(f"  var    {onnx_name}")
        return True
    weight, imgsz = EXPORTS[onnx_name]
    print(f"  uretim {onnx_name} ({weight}) ...")
    ul = _ultralytics()
    out = Path(ul.YOLO(weight).export(format="onnx", imgsz=imgsz, opset=12))
    _place(out, target)
    print(f"  TAMAM  {onnx_name}  ({target.stat().st_size / 1e6:.1f} MB)")
    return True


def export_yoloe() -> None:
    target = MODELS / "yoloe-26n-seg.onnx"
    if target.exists():
        print(f"  var    {target.name}")
        return
    print("  uretim yoloe-26n-seg.onnx (sozluk gomuluyor) ...")
    ul = _ultralytics()
    m = ul.YOLOE("yoloe-26n-seg.pt")
    m.set_classes(YOLOE_CLASSES, m.get_text_pe(YOLOE_CLASSES))
    out = Path(m.export(format="onnx", imgsz=640, opset=12))
    _place(out, target)
    for junk in ROOT.glob("mobileclip*"):     # metin kodlayici, ~242 MB
        junk.unlink()
    print(f"  TAMAM  {target.name}  ({target.stat().st_size / 1e6:.1f} MB)")
    print("\n  Kullanmak icin config.yaml -> yolo bolumunde:")
    print("    model: models/yoloe-26n-seg.onnx")
    print(f"    classes: {YOLOE_TR}")


def fetch_hand() -> None:
    """El modelini MediaPipe'in resmi adresinden indirir (yalnizca eksikse)."""
    target = MODELS / HAND_FILE
    if target.exists():
        print(f"  var    {HAND_FILE}")
        return
    print(f"  indirme {HAND_FILE}  <- {HAND_URL}")
    try:
        import urllib.request
        tmp = target.with_suffix(".part")
        urllib.request.urlretrieve(HAND_URL, tmp)
        tmp.replace(target)
        print(f"  TAMAM  {HAND_FILE}  ({target.stat().st_size / 1e6:.1f} MB)")
    except Exception as exc:                            # noqa: BLE001
        print(f"  ATLANDI: {exc}")
        print("         El tanima kapali kalir, sistemin geri kalani calisir.")


def set_key(key: str, value: str) -> None:
    """config.yaml'da bir anahtari yerinde gunceller (girintiyi korur)."""
    txt = CONFIG.read_text(encoding="utf-8")
    pat = rf"(?m)^(\s*){key}:.*$"
    if not re.search(pat, txt):
        print(f"  UYARI: config.yaml'da '{key}' bulunamadi, atlandi")
        return
    CONFIG.write_text(re.sub(pat, rf"\g<1>{key}: {value}", txt, count=1),
                      encoding="utf-8")


def use_yolo26_depth() -> None:
    """Derinlik motorunu yolo26'ya cevirir. Ham degerler dav2 ile ORTAK DEGIL,
    bu yuzden kalibrasyon tablosu da sifirlanir — once yedek alinir."""
    backup = CONFIG.with_suffix(".yaml.bak")
    shutil.copy(CONFIG, backup)
    print(f"  yedek  {backup.name} (eski kalibrasyon burada)")
    set_key("backend", "yolo26")
    set_key("model", "models/yolo26n-depth.onnx")
    set_key("calib_points", "[]")
    set_key("calib_scale", "1.0")
    set_key("calib_offset", "0.0")
    print("  config.yaml: backend=yolo26, kalibrasyon sifirlandi "
          "(model dogrudan metre uretir)")


def main() -> int:
    ap = argparse.ArgumentParser(description="EdgeVision model kurulumu")
    ap.add_argument("--depth", choices=("dav2", "yolo26"), default="dav2",
                    help="dav2: Depth Anything V2 (elle indirilir, daha genis "
                         "dinamik aralik) | yolo26: ultralytics uretir, hizli")
    ap.add_argument("--yoloe", action="store_true",
                    help="acik sozluklu modeli de uret (kapi, duvar, merdiven)")
    ap.add_argument("--no-hand", action="store_true",
                    help="el modelini indirme (MediaPipe katmani kapali kalir)")
    args = ap.parse_args()

    MODELS.mkdir(exist_ok=True)
    # Mutlak yol yazilmaz: kullanici loglari/ekran goruntusu paylastiginda
    # makine yolu da paylasilmis olurdu.
    print(f"hedef klasor: {MODELS.name}/\n")

    print("Tanima modeli:")
    export_model("yolo26n-seg.onnx")
    if args.yoloe:
        export_yoloe()

    if not args.no_hand:
        print("\nEl modeli:")
        fetch_hand()

    print("\nDerinlik modeli:")
    if args.depth == "yolo26":
        export_model("yolo26n-depth.onnx")
        use_yolo26_depth()
    elif (MODELS / DAV2).exists():
        print(f"  var    {DAV2}")
    else:
        print(f"  EKSIK  {DAV2} (~95 MB)")
        print("         Ultralytics'in modeli degil, disa aktarim komutu yok.")
        print("         Iki secenek:")
        print(f"           1) Dosyayi models/ altina kopyalayin")
        print("           2) Ultralytics'in derinlik modelini kullanin:")
        print("              python setup_models.py --depth yolo26")
        return 1

    print("\nHazir:  python main.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
