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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EdgeVision: metrik derinlik tabanlı mesafe")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--source", default=None, help="Kamera indeksi veya video dosyası")
    p.add_argument("--headless", action="store_true", help="Pencere yok, konsol durumu")
    p.add_argument("--display", action="store_true", help="Pencereyi zorla aç")
    p.add_argument("--json", action="store_true", help="Kare başına JSON (stdout)")
    p.add_argument("--scale", type=int, default=2, help="Pencere büyütme çarpanı")
    p.add_argument("--max-frames", type=int, default=0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config if Path(args.config).exists() else None)
    if args.source is not None:
        cfg.camera.source = int(args.source) if args.source.isdigit() else args.source

    pipeline = VisionPipeline(cfg).start()
    display = args.display or not (args.headless or args.json)

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
    print(f"Kamera aciliyor (kaynak={cfg.camera.source!r})...", file=sys.stderr)
    print("Metrik derinlik agi arka planda isiniyor (ilk harita ~1 sn)...", file=sys.stderr)
    with ThreadedCamera(cfg.camera.source, cfg.camera.capture_width,
                        cfg.camera.capture_height, cfg.camera.fps) as cam:
        t_start = time.perf_counter()
        try:
            while True:
                frame_id, frame = cam.read()
                if frame is None:
                    if isinstance(cfg.camera.source, str):
                        break
                    if processed == 0 and time.perf_counter() - t_start > 10.0:
                        print("HATA: kameradan kare gelmedi.", file=sys.stderr)
                        return 1
                    time.sleep(0.005); continue
                if frame_id == last_id:
                    time.sleep(0.002); continue
                last_id = frame_id
                if processed == 0:
                    print("Kamera hazir." + (" Pencere: q=cikis." if display else " Cikis: Ctrl+C."),
                          file=sys.stderr)

                report, command, dbg = pipeline.process(
                    frame_id, frame, want_edges=display and right_panel == "edges")
                processed += 1

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
                        print(f"[{command.action:9s}] engel={eng:18s} nesne={nes:20s} "
                              f"fps={dbg['fps']:.1f} "
                              f"derinlik={d.update_rate:.0f}fps({d.infer_ms:.0f}ms){ytxt}",
                              flush=True)

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
                        print(f"Kaydedildi: captures/frame_{frame_id:06d}.png", file=sys.stderr)
                if args.max_frames and processed >= args.max_frames:
                    break
        except KeyboardInterrupt:
            pass

    pipeline.stop()
    if display:
        import cv2
        cv2.destroyAllWindows()
    print(f"Islenen kare: {processed}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
