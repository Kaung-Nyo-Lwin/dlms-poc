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
WHEEL = (0, 165, 255)
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


def camera_from_pose(R, t):
    """Camera centre in world millimetres, and which way is up.

    The world frame is whatever the operator laid out on the tarmac, and nothing
    forces it to be right-handed: point +X along the kerb and +Y across it the
    other way, and +Z runs into the ground instead of out of it. Every 2D
    quantity survives that — the ground homography is a map, and a map does not
    care — so the survey checks out, the residuals are millimetres, and nothing
    complains. The mistake only surfaces when something is rectified to a height
    *above* the ground, because every such plane is then built on the wrong side
    of the tarmac and parallax runs backwards.

    So up is taken from the camera, which is unarguably above the ground, rather
    than from the frame's own +Z.
    """
    centre = -np.asarray(R, dtype=np.float64).T @ np.asarray(t, dtype=np.float64)
    return centre, (1.0 if centre[2] >= 0 else -1.0)


def homography_at_height(K, R, t, height_mm: float):
    """Image -> world (X, Y) for the horizontal plane Z = height_mm."""
    _, up = camera_from_pose(R, t)
    Hw2i = K @ np.column_stack([R[:, 0], R[:, 1], R[:, 2] * (up * float(height_mm)) + t])
    H = np.linalg.inv(Hw2i)
    return H / H[2, 2]


def _overlaps(a, b) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def chip(img, text, org, colour, k, weight=1.0, top=0, avoid=()):
    """Text on a solid backing, sized so it survives the final downscale.

    Everything is drawn on the full-resolution frame and the whole thing is
    resized once at the end, so a stroke laid down at 2 px comes out at 1 px in a
    half-size render. Every size here is multiplied by ``k`` for that reason. The
    backing matters as much as the size: an ROI label lands on grass in one frame
    and bright concrete in the next, and a plain stroke disappears into one of them.

    Three things can hide a number that has to be read, and staying inside the
    frame is only the first. ``top`` is the first row the chip may use: the
    status bar is painted last, over everything, so a label anchored on a shape
    near the top edge would otherwise be drawn and then buried — visible in the
    render as a chip with its top sliced off. ``avoid`` is the rectangles
    already taken by earlier chips, which a station with two lines a metre apart
    would otherwise stack into an unreadable pile; the chip walks downwards
    until it finds clear space, and stays put if it runs out of frame.

    Returns the rectangle it occupied, to be fed back in as ``avoid``.
    """
    scale = 1.0 * k * weight
    thick = max(1, round(2.4 * k * weight))
    (tw, th), base = cv2.getTextSize(text, FONT, scale, thick)
    pad = max(2, round(9 * k))
    H_img, W_img = img.shape[:2]
    x = int(min(max(org[0], pad + 1), W_img - tw - pad - 1))
    y = int(min(max(org[1], top + th + pad + 1), H_img - base - pad - 1))
    step = th + 2 * pad + max(2, round(4 * k))
    for _ in range(8):
        rect = (x - pad, y - th - pad, x + tw + pad, y + base + pad)
        if not any(_overlaps(rect, r) for r in avoid):
            break
        if y + step > H_img - base - pad - 1:
            break
        y += step
    rect = (x - pad, y - th - pad, x + tw + pad, y + base + pad)
    cv2.rectangle(img, rect[:2], rect[2:], (0, 0, 0), -1)
    cv2.rectangle(img, rect[:2], rect[2:], colour, max(1, round(2 * k)))
    cv2.putText(img, text, (x, y), FONT, scale, colour, thick, cv2.LINE_AA)
    return rect


def apply_h(H, pts):
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    q = np.column_stack([p, np.ones(len(p))]) @ np.asarray(H, dtype=np.float64).T
    w = q[:, 2:3]
    w = np.where(np.abs(w) < 1e-12, 1e-12, w)
    return q[:, :2] / w


def parse_time(text):
    """Seconds from "130", "2:10" or "1:02:03". None passes through.

    Colons because that is how anyone reads a time off a player's scrubber, and
    a render is nearly always cut to something that was watched first.
    """
    if text is None:
        return None
    parts = str(text).strip().split(":")
    if len(parts) > 3:
        raise SystemExit(f"cannot read {text!r} as a time; use s, m:ss or h:mm:ss")
    total = 0.0
    try:
        for part in parts:
            total = total * 60.0 + float(part)
    except ValueError:
        raise SystemExit(f"cannot read {text!r} as a time; use s, m:ss or h:mm:ss") from None
    return total


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--calibration", required=True, type=Path)
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--rois", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--roi-distorted", action="store_true")
    ap.add_argument("--start", default=None,
                    help="Render from this point on: seconds, or m:ss, or h:mm:ss.")
    ap.add_argument("--end", default=None, help="Render up to this point. Same formats.")
    ap.add_argument("--tyre-width-mm", type=float, default=None,
                    help="Draw each wheel as a tyre-sized footprint rather than a cross.")
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
    wheels_in_csv = all(f"wheel_{w}_x_mm" in columns for w in ("fl", "fr", "rl", "rr"))

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
    # The status bar is painted last, over everything. Its height is needed
    # before that, so the ROI labels can be kept out from under it.
    bar_h = round(40 * k)
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

    t0, t1 = parse_time(args.start), parse_time(args.end)
    if t0 is not None and t1 is not None and t1 <= t0:
        raise SystemExit(f"--end {args.end} is not after --start {args.start}")
    first = 0 if t0 is None else int(t0 * fps)
    last = None if t1 is None else int(t1 * fps)
    if first:
        # Seek rather than decode up to the window. The index is then read back
        # from the capture instead of counted from zero, so a backend that lands
        # somewhere other than where it was asked still keeps the rows lined up
        # with the pictures — and one that cannot seek at all just starts at the
        # beginning and is slow, rather than silently drawing the wrong frames.
        cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
    if t0 is not None or t1 is not None:
        span = [r for r in rows.values()
                if (t0 is None or float(r["time_s"]) >= t0)
                and (t1 is None or float(r["time_s"]) <= t1)]
        print(f"  window     {t0 if t0 is not None else 0:.2f}s to "
              f"{'end' if t1 is None else f'{t1:.2f}s'} — frames {first}"
              f"{'' if last is None else f'..{last}'}, {len(span)} row(s) in it")
        if not span:
            raise SystemExit(
                f"no rows in that window; {args.csv} covers "
                f"{min(float(r['time_s']) for r in rows.values()):.2f}s to "
                f"{max(float(r['time_s']) for r in rows.values()):.2f}s"
            )

    written = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if last is not None and idx > last:
            break
        row = rows.get(idx)
        if row is None:
            continue
        img = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        found = row["found"] == "1"

        labels = []
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
            labels.append((label, (mid[0] + 14 * k, mid[1] - 14 * k), col))

        if found:
            box_mm = np.array([[float(row[f"box{i}_x_mm"]), float(row[f"box{i}_y_mm"])]
                               for i in range(1, 5)])
            box = apply_h(world_to_px, box_mm)
            if np.isfinite(box).all():
                b = box.astype(np.int32)
                cv2.polylines(img, [b], True, BOX, lw, cv2.LINE_AA)
                # thicker along the front edge, so which way the car faces is
                # readable without following the heading arrow
                cv2.line(img, tuple(b[0]), tuple(b[1]), BOX, round(lw * 2.2), cv2.LINE_AA)

            # Contact patches, drawn through the GROUND homography like the box —
            # the tyre meets the road at z = 0, so this is the one place in the
            # frame where the car and the paint are genuinely on the same plane.
            # Each is a tyre-sized footprint rather than a dot, because the rules
            # ask whether a tyre is on a line, not whether a point is.
            if wheels_in_csv and row.get("wheel_fl_x_mm"):
                w_mm = np.array([[float(row[f"wheel_{w}_x_mm"]),
                                  float(row[f"wheel_{w}_y_mm"])]
                                 for w in ("fl", "fr", "rl", "rr")])
                wp = apply_h(world_to_px, w_mm)
                if np.isfinite(wp).all():
                    if args.tyre_width_mm:
                        # Radius in pixels from the local ground sample distance,
                        # so the mark shrinks with distance the way the car does.
                        off = np.array([args.tyre_width_mm / 2, 0.0])
                        for centre_px, corner in zip(wp, w_mm, strict=True):
                            edge = apply_h(world_to_px, (corner + off)[None, :])[0]
                            r = round(float(np.linalg.norm(edge - centre_px)))
                            cv2.circle(img, tuple(centre_px.astype(np.int32)),
                                       max(3, r), WHEEL, lw, cv2.LINE_AA)
                    for q in wp.astype(np.int32):
                        cv2.drawMarker(img, tuple(q), WHEEL, cv2.MARKER_CROSS,
                                       round(20 * k), lw)
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
        # Labels last. The footprint, the wheels and the detection quad are all
        # drawn over the ROI shapes on purpose — the car is the subject — but a
        # clearance is a number to be read, not part of the picture, so it goes
        # on top of the lot. Before this, a box edge crossing a chip struck the
        # reading through in the frames where the car was closest, which are
        # exactly the frames anyone reviews.
        placed = []
        for label, org, col in labels:
            placed.append(chip(img, label, org, col, k, top=bar_h, avoid=placed))

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
