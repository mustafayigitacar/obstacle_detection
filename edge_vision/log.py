"""Merkezi loglama (Python `logging`) + kare başına telemetri.

    from edge_vision.log import setup_logging, get_logger
    setup_logging(cfg.logging)
    log = get_logger(__name__)
    log.info("kamera hazir")

İki çıktı vardır:

  * İNSAN LOGU  — konsol (stderr) + dönen dosya `logs/edgevision.log`.
    Olaylar: açılış, model/sağlayıcı, zemin kilidi, sahne kesmesi, ışık
    sıçraması, komut değişimi, iş parçacığı hataları.
  * TELEMETRİ   — `logs/telemetry_<zaman>.jsonl`, kare başına tek satır.
    Sahne öznitelikleri + ölçek çapası + en yakın engel/nesne. Ani değişim
    sorunlarını SONRADAN incelemek ve gerekirse bir makine öğrenmesi modelini
    eğitmek için veri setidir (bkz. README "Ani değişimler").

stdout'a hiçbir şey yazılmaz: `--json` kipi robot köprüsünün kanalıdır.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
import threading
import time
from pathlib import Path
from typing import Optional

ROOT = "edge_vision"
_FMT = "%(asctime)s.%(msecs)03d %(levelname)-7s %(name)-24s %(message)s"
_DATEFMT = "%H:%M:%S"


def get_logger(name: str) -> logging.Logger:
    """Modül logları `edge_vision.*` altında toplanır (tek yerden ayarlanır)."""
    if not name.startswith(ROOT):
        name = f"{ROOT}.{name}"
    return logging.getLogger(name)


def setup_logging(cfg=None, level_override: Optional[str] = None) -> logging.Logger:
    """Kök `edge_vision` logger'ını kurar. Birden fazla çağrı güvenlidir."""
    level_name = (level_override or getattr(cfg, "level", "INFO") or "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    root = logging.getLogger(ROOT)
    root.setLevel(logging.DEBUG)          # filtreyi handler'lar yapar
    root.propagate = False
    for h in list(root.handlers):
        root.removeHandler(h)
        h.close()

    fmt = logging.Formatter(_FMT, _DATEFMT)
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(level)
    console.setFormatter(fmt)
    root.addHandler(console)

    log_dir = Path(getattr(cfg, "dir", "logs") or "logs")
    if getattr(cfg, "file", True):
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            log_dir / "edgevision.log",
            maxBytes=int(getattr(cfg, "max_mb", 5)) * 1024 * 1024,
            backupCount=int(getattr(cfg, "backups", 3)),
            encoding="utf-8")
        fh.setLevel(getattr(logging, str(getattr(cfg, "file_level", "DEBUG")).upper(),
                            logging.DEBUG))
        fh.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)-7s %(threadName)-12s %(name)s: %(message)s"))
        root.addHandler(fh)

    # İş parçacıklarında yakalanmamış hata sessizce ölmesin: loga düşsün.
    def _thread_hook(args):
        if args.exc_type is SystemExit:
            return
        root.critical("is parcacigi coktu: %s", args.thread.name if args.thread else "?",
                      exc_info=(args.exc_type, args.exc_value, args.exc_traceback))
    threading.excepthook = _thread_hook
    return root


class Telemetry:
    """Kare başına JSONL kaydı. Kapalıysa her çağrı boşa döner (maliyet yok)."""

    def __init__(self, cfg=None):
        self.enabled = bool(getattr(cfg, "telemetry", False))
        self.every = max(1, int(getattr(cfg, "telemetry_every", 1)))
        self._fh = None
        self._n = 0
        self.path: Optional[Path] = None
        if self.enabled:
            d = Path(getattr(cfg, "dir", "logs") or "logs")
            d.mkdir(parents=True, exist_ok=True)
            self.path = d / f"telemetry_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
            self._fh = self.path.open("w", encoding="utf-8")
            get_logger("telemetry").info("telemetri: %s", self.path)

    def write(self, record: dict) -> None:
        if self._fh is None:
            return
        self._n += 1
        if self._n % self.every:
            return
        record.setdefault("t", round(time.time(), 3))
        self._fh.write(json.dumps(record, ensure_ascii=False, default=_jsonable) + "\n")

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def _jsonable(o):
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
    except ImportError:
        pass
    if isinstance(o, float) and o != o:
        return None
    return str(o)
