"""EdgeVision giriş noktası.

    python main.py                   # kamera + debug penceresi
    python main.py --source test.mp4 # video dosyasıyla
    python main.py --json            # kare başına JSON (robot köprüsü)
    python main.py --headless        # pencere yok, konsola 1 Hz durum

Pencerede: q=çıkış, s=kare kaydet, d=sağ panel, c=kalibrasyon, tıklama=mesafe.
Kalibrasyonda: n=sonraki nesne, m=nişan kutusu, +/-=kutu boyutu,
rakam/nokta=mesafe, ENTER=kaydet, f=uygula, x=sil, r=geri al.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from edge_vision import ThreadedCamera, VisionPipeline, load_config
from edge_vision.log import Telemetry, get_logger, setup_logging

log = get_logger("main")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EdgeVision: metrik derinlik tabanlı mesafe")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--source", default=None, help="Kamera indeksi veya video dosyası")
    p.add_argument("--headless", action="store_true", help="Pencere yok, konsol durumu")
    p.add_argument("--display", action="store_true", help="Pencereyi zorla aç")
    p.add_argument("--json", action="store_true", help="Kare başına JSON (stdout)")
    p.add_argument("--scale", type=int, default=2, help="Pencere büyütme çarpanı")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--log-level", default=None,
                   help="Konsol log seviyesi: DEBUG|INFO|WARNING|ERROR (config'i ezer)")
    p.add_argument("--telemetry", action="store_true",
                   help="Kare basina JSONL telemetri (logs/telemetry_*.jsonl)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg_found = Path(args.config).exists()
    cfg = load_config(args.config if cfg_found else None)
    if args.source is not None:
        cfg.camera.source = int(args.source) if args.source.isdigit() else args.source
    if args.telemetry:
        cfg.logging.telemetry = True
    setup_logging(cfg.logging, args.log_level)
    log.info("EdgeVision basliyor (config=%s)", args.config)
    if not cfg_found:
        log.warning("config dosyasi yok (%s): varsayilan degerler kullaniliyor, "
                    "kalibrasyon kaydedilemez", args.config)
    if args.scale < 1:
        log.warning("--scale %d gecersiz, 1 kullaniliyor", args.scale)
        args.scale = 1

    try:
        pipeline = VisionPipeline(cfg).start()
    except Exception:
        log.exception("hat kurulamadi")
        return 1
    try:
        return _run(args, cfg, pipeline)
    finally:
        pipeline.stop()


def _run(args, cfg, pipeline) -> int:
    telemetry = Telemetry(cfg.logging)
    display = args.display or not (args.headless or args.json)
    try:
        return _loop(args, cfg, pipeline, telemetry, display)
    finally:
        telemetry.close()
        if display:
            import cv2
            cv2.destroyAllWindows()


def _loop(args, cfg, pipeline, telemetry, display) -> int:
    last_action = None

    win = "EdgeVision  [q=cikis  s=kaydet  d=panel  c=kalibrasyon  tik=mesafe]"
    probe = {"uv": None, "at": 0.0, "ema": None}
    right_panel = "depth"
    calib = None
    if display:
        import cv2
        from edge_vision.viz import compose_view
        from edge_vision.calib_overlay import CalibOverlay
        calib = CalibOverlay(args.config, pipeline.depth, cfg.calib)
        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow(win, 40, 40)
        cv2.setWindowProperty(win, cv2.WND_PROP_TOPMOST, 1)

        def on_mouse(event, x, y, flags, param):
            if event != cv2.EVENT_LBUTTONDOWN:
                return
            u, v = x // args.scale, y // args.scale
            if u >= pipeline.pw:
                u -= pipeline.pw
            probe["uv"] = (u, v); probe["at"] = time.time(); probe["ema"] = None
        cv2.setMouseCallback(win, on_mouse)

    processed = 0
    last_id = -1
    last_status = 0.0
    cam_stalled = False
    log.info("kamera aciliyor (kaynak=%r)...", cfg.camera.source)
    log.info("metrik derinlik agi arka planda isiniyor (ilk harita ~1 sn)...")
    try:
        cam = ThreadedCamera(cfg.camera.source, cfg.camera.capture_width,
                             cfg.camera.capture_height, cfg.camera.fps)
    except RuntimeError as exc:
        log.error("%s", exc)
        return 1
    with cam:
        t_start = time.perf_counter()
        try:
            while True:
                frame_id, frame = cam.read()
                if frame is None:
                    # Yalnızca yerel dosyada None = akışın sonu. RTSP/HTTP
                    # akışı (str ama dosya değil) ilk kareleri geç verebilir;
                    # eskiden burada hemen çıkılıyordu.
                    if cam.is_file:
                        break
                    if processed == 0 and time.perf_counter() - t_start > 10.0:
                        log.error("kameradan kare gelmedi (10 sn)")
                        return 1
                    time.sleep(0.005); continue
                if frame_id == last_id:
                    # Canlı kamera koptu: son kare kopyalanmaya devam eder ama
                    # kimliği artmaz. Robot köprüsü sessizce eski komutla
                    # kalmasın — bir kez STOP yayınlanır.
                    if not cam_stalled and cam.age_s > 2.0:
                        cam_stalled = True
                        log.error("kamera: %.1f sn'dir yeni kare yok", cam.age_s)
                        if args.json:
                            print(json.dumps({"command": {
                                "action": "STOP", "steer_deg": 0.0, "speed_scale": 0.0,
                                "nearest_m": None, "reason": "kamera akisi durdu"}}),
                                flush=True)
                    time.sleep(0.002); continue
                if cam_stalled:
                    log.info("kamera: yeni kareler geliyor")
                    cam_stalled = False
                last_id = frame_id
                if processed == 0:
                    log.info("kamera hazir. %s",
                             "Pencere: q=cikis." if display else "Cikis: Ctrl+C.")

                report, command, dbg = pipeline.process(
                    frame_id, frame, want_edges=display and right_panel == "edges")
                processed += 1

                if command.action != last_action:
                    n = report.nearest
                    log.info("komut: %s -> %s%s", last_action, command.action,
                             "" if n is None else
                             f"  (engel {n.distance_m:.2f} m @ {n.bearing_deg:+.0f} derece)")
                    last_action = command.action
                if telemetry.enabled:
                    telemetry.write(pipeline.telemetry_record(report, command, dbg["fps"]))

                if args.json:
                    print(json.dumps({"command": command.to_dict(), **report.to_dict()}),
                          flush=True)
                if not display and not args.json:
                    now = time.perf_counter()
                    if now - last_status >= 1.0:
                        last_status = now
                        n = report.nearest
                        eng = "yok" if n is None else f"{n.distance_m:.2f}m @ {n.bearing_deg:+.0f}"
                        nn = report.nearest_named
                        nes = "yok" if nn is None else f"{nn.label} {nn.distance_m:.2f}m"
                        d = dbg["depth"]
                        y = dbg["yolo"]
                        ytxt = "" if y is None else \
                            f" yolo={y.update_rate:.0f}fps({y.infer_ms:.0f}ms)"
                        sc = dbg["scene"]
                        stxt = "" if sc.stable else f" sahne={','.join(sc.events) or 'oturuyor'}"
                        log.info("[%-9s] engel=%-18s nesne=%-20s fps=%.1f "
                                 "derinlik=%.0ffps(%.0fms)%s k=%.2f%s",
                                 command.action, eng, nes, dbg["fps"],
                                 d.update_rate, d.infer_ms, ytxt, dbg["scale_k"], stxt)

                if display:
                    if calib.active:
                        calib.update(dbg["depth_m"], pipeline.depth.get_raw_depth(),
                                     report.detections, dbg["ground_mask"])
                    view = compose_view(dbg["proc"], dbg["edges"], report, command,
                                        dbg["roi_top"], dbg["fps"], scale=args.scale,
                                        depth=dbg["depth"], depth_m=dbg["depth_m"],
                                        right_panel=right_panel,
                                        max_range_m=cfg.avoidance.max_range_m,
                                        show_obstacles=not calib.active,
                                        yolo=dbg["yolo"])
                    if probe["uv"] is not None and time.time() - probe["at"] < 5.0:
                        pu, pv = probe["uv"]
                        txt = "derinlik yok"
                        dm = dbg["depth_m"]
                        if dm is not None:
                            y0 = max(0, min(pv, pipeline.ph - 1) - 1)
                            x0 = max(0, min(pu, pipeline.pw - 1) - 1)
                            patch = dm[y0:y0 + 3, x0:x0 + 3]
                            vals = patch[patch > 1e-3]
                            if vals.size:
                                zq = float(np.median(vals))
                                probe["ema"] = zq if probe["ema"] is None else \
                                    probe["ema"] + 0.4 * (zq - probe["ema"])
                                rawm = pipeline.depth.get_raw_depth()
                                rtxt = ""
                                if rawm is not None:
                                    rp = rawm[y0:y0 + 3, x0:x0 + 3]
                                    rv = rp[rp > 1e-3]
                                    if rv.size:
                                        rtxt = f"  (ham {float(np.median(rv)):.2f})"
                                txt = f"{probe['ema']:.2f} m{rtxt}"
                        sx, sy = pu * args.scale, pv * args.scale
                        cv2.drawMarker(view, (sx, sy), (255, 0, 255), cv2.MARKER_CROSS, 14, 2)
                        cv2.putText(view, txt, (max(4, sx - 40), max(16, sy - 12)),
                                    cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 0, 255), 2)

                    calib.draw(view, args.scale, pipeline.pw, pipeline.ph)
                    cv2.imshow(win, view)
                    if processed == 90:
                        cv2.setWindowProperty(win, cv2.WND_PROP_TOPMOST, 0)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    elif key == ord("c"):
                        calib.toggle()
                        if calib.active:
                            right_panel = "depth"
                    elif calib.active:
                        calib.handle_key(key)
                    elif key == ord("d"):
                        right_panel = "depth" if right_panel == "edges" else "edges"
                    elif key == ord("s"):
                        cap_dir = Path("captures"); cap_dir.mkdir(exist_ok=True)
                        cv2.imwrite(str(cap_dir / f"frame_{frame_id:06d}.png"), view)
                        log.info("kaydedildi: captures/frame_%06d.png", frame_id)
                if args.max_frames and processed >= args.max_frames:
                    break
        except KeyboardInterrupt:
            log.info("Ctrl+C ile durduruldu")
        except Exception:
            log.exception("ana dongude beklenmeyen hata (kare %d)", processed)
            raise

    log.info("islenen kare: %d  sahne kesmesi: %d", processed, pipeline.scene.n_cuts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
