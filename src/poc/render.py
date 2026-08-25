"""Draw a pipeline's CSV back onto the video.

Reads either pipeline's output — the two write the same core columns — and
paints the car box, the sticker centre, the heading and the ROIs into the
frames they were measured in, with the per-ROI clearances as a readout.

Everything drawn is a projection of numbers already in the CSV, so the overlay
cannot flatter a pipeline: if the geometry is wrong, the box slides off the car
on screen. The box is projected through the *field* plane because that is where
the car's footprint physically is; through the car plane it would be a correct
outline in the wrong place, and a misplaced box still looks like a car.

    python render.py --video clip.mp4 --calibration calibration.json \
        --csv track.csv --rois rois.json --out overlay.mp4
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np

BOX = (0, 220, 0)
STICKER = (0, 255, 255)
CENTRE = (0, 0, 255)
HEADING = (247, 120, 186)
ROI_OK = (255, 190, 60)
ROI_HIT = (60, 60, 235)
TEXT = (255, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def undistort_maps(K, D, model, width, height):
    """Remap tables that straighten the lens, for whichever model was fitted.

    The projection stays the *original* camera matrix, so the image and any
    points moved with ``undistort_points`` share one pixel frame — the frame the
    stored homographies are defined on.

    The model has to be honoured rather than assumed. Fisheye coefficients fed
    to the pinhole undistorter are read as ``(k1, k2, p1, p2)``, which leaves
    the frame very nearly untouched: no error, no warning, just an uncorrected
    image that every measurement below inherits.
    """
    if model == "fisheye":
        if len(D) < 4:
            raise SystemExit(f"fisheye intrinsics need 4 distortion coefficients, got {len(D)}")
        return cv2.fisheye.initUndistortRectifyMap(
            K, D[:4].reshape(4, 1), np.eye(3), K, (width, height), cv2.CV_16SC2)
    if model not in ("pinhole", "", None):
        raise SystemExit(f"unknown camera model {model!r}; expected 'pinhole' or 'fisheye'")
    return cv2.initUndistortRectifyMap(K, D, None, K, (width, height), cv2.CV_16SC2)


def undistort_points(pts, K, D, model):
    """Move pixels into the undistorted frame the maps above produce."""
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    if model == "fisheye":
        return cv2.fisheye.undistortPoints(
            p, K, D[:4].reshape(4, 1), R=np.eye(3), P=K).reshape(-1, 2)
    return cv2.undistortPoints(p, K, D, P=K).reshape(-1, 2)


def homography_at_height(K, R, t, height_mm: float):
    """Image -> world (X, Y) for the horizontal plane Z = height_mm."""
    Hw2i = K @ np.column_stack([R[:, 0], R[:, 1], R[:, 2] * float(height_mm) + t])
    H = np.linalg.inv(Hw2i)
    return H / H[2, 2]


def chip(img, text, org, colour, k, weight=1.0):
    """Text on a solid backing, sized so it survives the final downscale.

    Everything is drawn on the full-resolution frame and the whole thing is
    resized once at the end, so a stroke laid down at 2 px comes out at 1 px in a
    half-size render. Every size here is multiplied by ``k`` for that reason. The
    backing matters as much as the size: an ROI label lands on grass in one frame
    and bright concrete in the next, and a plain stroke disappears into one of them.
    """
    scale = 1.0 * k * weight
    thick = max(1, round(2.4 * k * weight))
    (tw, th), base = cv2.getTextSize(text, FONT, scale, thick)
    pad = max(2, round(9 * k))
    # Keep the chip inside the frame: an ROI near an edge would otherwise have
    # its number clipped off, which is the one part of it that has to be read.
    H_img, W_img = img.shape[:2]
    x = int(min(max(org[0], pad + 1), W_img - tw - pad - 1))
    y = int(min(max(org[1], th + pad + 1), H_img - base - pad - 1))
    cv2.rectangle(img, (x - pad, y - th - pad), (x + tw + pad, y + base + pad), (0, 0, 0), -1)
    cv2.rectangle(img, (x - pad, y - th - pad), (x + tw + pad, y + base + pad), colour,
                  max(1, round(2 * k)))
    cv2.putText(img, text, (x, y), FONT, scale, colour, thick, cv2.LINE_AA)
    return tw + 2 * pad


def apply_h(H, pts):
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    q = np.column_stack([p, np.ones(len(p))]) @ np.asarray(H, dtype=np.float64).T
    w = q[:, 2:3]
    w = np.where(np.abs(w) < 1e-12, 1e-12, w)
    return q[:, :2] / w


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--calibration", required=True, type=Path)
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--rois", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--roi-distorted", action="store_true")
    ap.add_argument("--units", default="m", choices=("m", "cm", "mm"))
    ap.add_argument("--scale", type=float, default=0.5, help="Output size relative to the source.")
    ap.add_argument("--label-scale", type=float, default=1.0,
                    help="Extra multiplier on line widths and text (try 1.5 or 2).")
    args = ap.parse_args()

    cal = json.loads(args.calibration.read_text())
    K = np.array(cal["intrinsics"]["camera_matrix"], dtype=np.float64)
    D = np.array(cal["intrinsics"]["dist_coeffs"], dtype=np.float64).ravel()
    model = str(cal["intrinsics"].get("model", "pinhole"))
    W, H = int(cal["image_width"]), int(cal["image_height"])
    rois = []
    for item in json.loads(args.rois.read_text())["rois"]:
        pts = np.array(item["points_px"], dtype=np.float64).reshape(-1, 2)
        if args.roi_distorted:
            pts = undistort_points(pts, K, D, model)
        rois.append({"name": item["name"], "px": pts,
                     "closed": item.get("type") == "polygon"})

    rows, columns = {}, []
    with open(args.csv, newline="") as fh:
        reader = csv.DictReader(fh)
        columns = list(reader.fieldnames or [])
        for row in reader:
            rows[int(row["frame"])] = row

    # The CSV and the ROI file are two separate paths, and nothing links them.
    # Rendering a track against a different ROI set than it was measured with
    # produces a picture where every clearance label belongs to a shape that is
    # not the one drawn — so refuse rather than illustrate a lie.
    # Keyed on the "_hit" suffix, which only an ROI emits. Matching "_mm" and
    # excluding known prefixes was fragile: it silently swallowed the geometry
    # columns the moment new ones were added.
    in_csv = {c[:-4] for c in columns if c.endswith("_hit")}
    in_file = {r["name"] for r in rois}
    if in_csv != in_file:
        raise SystemExit(
            f"{args.csv} was measured against ROIs {sorted(in_csv) or '[]'} but "
            f"{args.rois} contains {sorted(in_file)}. Render with the ROI file the "
            "track was produced from, or re-run the pipeline against this one."
        )

    H_field = np.array(cal["field"]["homography"], dtype=np.float64)
    for r in rois:
        w = apply_h(H_field, r["px"])
        print(f"  {r['name']:12s} {len(w)} pt  first at "
              f"({w[0][0]:+.0f}, {w[0][1]:+.0f}) mm on the ground")
    world_to_px = np.linalg.inv(H_field)

    # The detection box is the matched template's outline, and it sits on the
    # roof — so it is projected through a plane at the marker's height, not
    # through the ground. Same centre, different plane; through the ground one
    # it would land about two metres away and still look plausible.
    pose = cal["field"].get("pose")
    heights = {float(r["sticker_height_mm"]) for r in rows.values()
               if r.get("sticker_height_mm")}
    sticker_px = {}
    if heights and pose is not None:
        R = np.array(pose["rotation"], dtype=np.float64)
        t = np.array(pose["translation_mm"], dtype=np.float64)
        for h in heights:
            sticker_px[h] = np.linalg.inv(homography_at_height(K, R, t, h))
    elif heights:
        print("  NOTE: no camera pose in the calibration, so the detection box "
              "cannot be placed on the marker's plane; drawing the footprint only.")

    # Overlays are drawn on the full-resolution frame and the result is resized
    # once at the end, so every stroke and glyph is pre-multiplied by the inverse
    # of that resize. Without it a 3 px line renders at 1.5 px and the labels are
    # unreadable — which is the whole reason this factor exists.
    k = args.label_scale / max(args.scale, 1e-6)
    lw = max(2, round(5 * k))

    divisor = {"m": 1000.0, "cm": 10.0, "mm": 1.0}[args.units]
    dp = {"m": 2, "cm": 1, "mm": 0}[args.units]

    map1, map2 = undistort_maps(K, D, model, W, H)
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"could not open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    out_size = (int(W * args.scale) // 2 * 2, int(H * args.scale) // 2 * 2)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), fps, out_size)

    idx, written = -1, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        row = rows.get(idx)
        if row is None:
            continue
        img = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        found = row["found"] == "1"

        for r in rois:
            hit = found and row.get(f"{r['name']}_hit") == "1"
            col = ROI_HIT if hit else ROI_OK
            p = r["px"].astype(np.int32)
            if len(p) == 1:
                cv2.drawMarker(img, tuple(p[0]), col, cv2.MARKER_TILTED_CROSS,
                               round(34 * k), lw)
            else:
                cv2.polylines(img, [p], r["closed"], col, lw, cv2.LINE_AA)
            for q in p:
                cv2.circle(img, tuple(q), round(7 * k), col, -1, cv2.LINE_AA)
            label = r["name"]
            gap = row.get(r["name"] + "_mm") if found else None
            if gap:
                label += f"   {float(gap) / divisor:+.{dp}f} {args.units}"
            # anchored on the shape's middle, not its first vertex, which is as
            # likely as not to be under the car or off the edge of the frame
            mid = p.mean(axis=0)
            chip(img, label, (mid[0] + 14 * k, mid[1] - 14 * k), col, k)

        if found:
            box_mm = np.array([[float(row[f"box{i}_x_mm"]), float(row[f"box{i}_y_mm"])]
                               for i in range(1, 5)])
            box = apply_h(world_to_px, box_mm)
            if np.isfinite(box).all():
                b = box.astype(np.int32)
                cv2.polylines(img, [b], True, BOX, lw, cv2.LINE_AA)
                cv2.line(img, tuple(b[0]), tuple(b[1]), BOX, round(lw * 2.2), cv2.LINE_AA)
            h = float(row.get("sticker_height_mm") or 0.0)
            if h in sticker_px and row.get("stick1_x_mm"):
                quad_mm = np.array([[float(row[f"stick{i}_x_mm"]), float(row[f"stick{i}_y_mm"])]
                                    for i in range(1, 5)])
                q = apply_h(sticker_px[h], quad_mm)
                if np.isfinite(q).all():
                    cv2.polylines(img, [q.astype(np.int32)], True, STICKER, lw, cv2.LINE_AA)

            centre_mm = np.array([[float(row["sticker_x_mm"]), float(row["sticker_y_mm"])]])
            c = apply_h(world_to_px, centre_mm)[0]
            a = math.radians(float(row["heading_deg"]))
            reach = np.array([[math.cos(a) * 2500, math.sin(a) * 2500]])
            tip = apply_h(world_to_px, centre_mm + reach)[0]
            if np.isfinite(c).all() and np.isfinite(tip).all():
                cv2.arrowedLine(img, (int(c[0]), int(c[1])), (int(tip[0]), int(tip[1])),
                                HEADING, lw, cv2.LINE_AA, tipLength=0.15)
                cv2.drawMarker(img, (int(c[0]), int(c[1])), CENTRE, cv2.MARKER_CROSS,
                               round(38 * k), lw)

        bar = f"f{idx}  t={float(row['time_s']):6.2f}s  "
        bar += (f"{row['method']}  score {float(row['score']):.3f}  "
                f"({float(row['sticker_x_mm']) / 1000:+.3f}, {float(row['sticker_y_mm']) / 1000:+.3f}) m  "
                f"hdg {float(row['heading_deg']):+.1f}°  sigma {float(row['sigma_mm']):.2f} mm"
                if found else "MISS")
        bar_h = round(40 * k)
        cv2.rectangle(img, (0, 0), (W, bar_h), (0, 0, 0), -1)
        cv2.putText(img, bar, (round(14 * k), round(28 * k)), FONT, 0.78 * k,
                    TEXT if found else ROI_HIT, max(1, round(2 * k)), cv2.LINE_AA)

        writer.write(cv2.resize(img, out_size, interpolation=cv2.INTER_AREA))
        written += 1

    cap.release()
    writer.release()
    print(f"wrote {written} frames to {args.out}")


if __name__ == "__main__":
    main()
