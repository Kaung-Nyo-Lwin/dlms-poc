"""Survey tool: everything a pipeline run needs, produced in one folder.

Run the steps in order. Each one writes a file; the next one reads it. There are
no stations, no vehicle ids and no workspace — every input and output is a path
you choose, so a whole site lives in one directory.

    0  frame       grab one still to survey on
    1  intrinsics  checkerboard video      -> intrinsics.json
    2  gcp         click ground points     -> calibration.json
    3  measure     check it against a tape (no output; do this before trusting it)
    3b carplane    survey the marker's plane  -> car block in calibration.json
    4  sticker     cut the marker template -> sticker.png + sticker.json
    5  car         vehicle geometry        -> car.json
    6  roi         draw what to measure to -> rois.json

Then::

    python pipeline1_bev.py --video V --calibration calibration.json --car car.json \\
        --template sticker.png --template-mm-per-px <from sticker.json> \\
        --rois rois.json --out track.csv

One rule holds everything together: **every pixel this tool records is in the
undistorted frame.** Steps 2-6 undistort before showing you anything, using the
original camera matrix as the projection, so the homographies, the ROI pixels
and the frames the pipelines undistort all share one coordinate frame. Mixing
distorted and undistorted pixels is a smooth, plausible-looking error worth tens
of pixels at the frame edge — hundreds of millimetres on the ground — and
nothing downstream can detect it.

Standalone: numpy + opencv + the standard library. Nothing here imports dlms.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import picker

# --------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------


# OpenCV 5 moved the fisheye calibration flags onto the top-level namespace;
# 4.x has them under cv2.fisheye. Look in both so one file runs on either.
_FISHEYE_CALIB_FLAGS = (
    getattr(cv2, "CALIB_RECOMPUTE_EXTRINSIC", None)
    or getattr(cv2.fisheye, "CALIB_RECOMPUTE_EXTRINSIC", 0)
) | (
    getattr(cv2, "CALIB_FIX_SKEW", None)
    or getattr(cv2.fisheye, "CALIB_FIX_SKEW", 0)
)


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def save_json(path: Path, data: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2) + "\n")
    print(f"  wrote {path}", flush=True)


def read_frame(video: Path, at_s: float) -> np.ndarray:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"could not open {video}")
    if at_s > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, at_s * 1000.0)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"could not read a frame from {video} at {at_s}s")
    return frame


def load_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"could not read {path}")
    return img


def intrinsics_arrays(intr: dict) -> tuple[np.ndarray, np.ndarray, str]:
    return (np.array(intr["camera_matrix"], dtype=np.float64),
            np.array(intr["dist_coeffs"], dtype=np.float64).ravel(),
            str(intr.get("model", "pinhole")))


def undistort_maps(K, D, model, width, height):
    """Remap tables that straighten the lens, for whichever model was fitted.

    The projection stays the *original* camera matrix. That is not cosmetic:
    points are moved into this frame with the matching ``undistortPoints``, and
    re-framing with ``getOptimalNewCameraMatrix`` here would put the image and
    the points in different pixel frames — a smooth positional bias rather than
    an obvious failure. The cost is black wedges at the border where the
    undistorted field of view exceeds the sensor.

    The model has to be honoured rather than assumed. Fisheye coefficients fed
    to the pinhole undistorter are read as ``(k1, k2, p1, p2)``, which leaves
    the image very nearly untouched: no error, no warning, just an uncorrected
    frame that every later measurement inherits.
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


def undistort(img: np.ndarray, intr: dict, source: str = "") -> np.ndarray:
    """Straighten a frame for the model its intrinsics were fitted with.

    The frame must be the size the intrinsics describe. A resized one would
    remap without complaint and come back looking almost right, because the
    principal point and focal length would be off by exactly the scale factor.
    """
    K, D, model = intrinsics_arrays(intr)
    h, w = img.shape[:2]
    if (w, h) != (intr["image_width"], intr["image_height"]):
        raise SystemExit(
            f"{source or 'the image'} is {w}x{h} but these intrinsics describe "
            f"{intr['image_width']}x{intr['image_height']}. They do not fit this frame; "
            "re-shoot the calibration in the station's own framing."
        )
    m1, m2 = undistort_maps(K, D, model, w, h)
    return cv2.remap(img, m1, m2, cv2.INTER_LINEAR)


def apply_h(H: np.ndarray, pts) -> np.ndarray:
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    q = np.column_stack([p, np.ones(len(p))]) @ np.asarray(H, dtype=np.float64).T
    w = q[:, 2:3]
    return q[:, :2] / np.where(np.abs(w) < 1e-12, 1e-12, w)


def homography_at_height(K, R, t, height_mm: float) -> np.ndarray:
    """Image -> world (X, Y) for the horizontal plane Z = height_mm."""
    Hw2i = K @ np.column_stack([R[:, 0], R[:, 1], R[:, 2] * float(height_mm) + t])
    H = np.linalg.inv(Hw2i)
    return H / H[2, 2]


def calibration_planes(cal: dict):
    """(K, model, R, t) from a calibration file, or a clear complaint."""
    K, _, model = intrinsics_arrays(cal["intrinsics"])
    pose = cal["field"].get("pose")
    if pose is None:
        raise SystemExit("this calibration has no camera pose; re-run the `gcp` step")
    return K, model, np.array(pose["rotation"]), np.array(pose["translation_mm"])


#: No camera in this application is anywhere near this high. A fit implying more
#: means the two levels are barely separated — the signature of targets that
#: never actually left the ground. Testing the recovered height rather than the
#: raw scale keeps the check physical: k approaches 1 smoothly, so any epsilon
#: on it is arbitrary, while a 200 m mast is plainly absurd.
MAX_CAMERA_MM = 200_000.0


def fit_homology(shadow_mm, world_mm):
    """Least-squares ``shadow = k * world + b`` — linear in ``(k, bx, by)``."""
    n = len(world_mm)
    A = np.zeros((2 * n, 3), dtype=np.float64)
    A[0::2, 0], A[0::2, 1] = world_mm[:, 0], 1.0
    A[1::2, 0], A[1::2, 2] = world_mm[:, 1], 1.0
    sol, *_ = np.linalg.lstsq(A, shadow_mm.reshape(-1), rcond=None)
    return float(sol[0]), sol[1:]


def car_plane_from_field(H_field, pixels, world_mm, height_mm: float) -> dict:
    """Solve the car plane as a homology of the field plane, from as few as two targets.

    A free plane homography has eight degrees of freedom and needs four points.
    The car plane is not free: it is parallel to the ground at a known height,
    so once the field plane is known the only unknowns are a scale ``k`` and
    the camera's ground position — three numbers. A point at height ``h`` over
    world ``(X, Y)`` is seen along the ray from the camera centre, which meets
    the ground at::

        (X', Y') = C_xy + k * ((X, Y) - C_xy),      k = cz / (cz - h)

    — a uniform scaling about where the camera stands, with no rotation. So two
    targets give four equations for three unknowns, already one more than
    required, and a homology cannot represent a "car plane" tilted relative to
    the ground, which no camera can actually see.

    The trade against raising Z through the pose is real in both directions.
    This inherits whatever error the field plane carries, but it measures the
    plane the marker rides on instead of assuming the pose is exact, and the
    camera height it recovers can be checked against a tape on the mast.
    """
    H_field = np.asarray(H_field, dtype=np.float64)
    pixels = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
    world_mm = np.asarray(world_mm, dtype=np.float64).reshape(-1, 2)
    if len(pixels) != len(world_mm):
        raise SystemExit("every car target needs a world coordinate")
    if len(pixels) < 2:
        raise SystemExit(f"the car plane needs at least 2 targets, got {len(pixels)}")
    if height_mm <= 0:
        raise SystemExit(f"car plane height must be above the ground, got {height_mm} mm")

    # Where each raised target appears to stand when read through the ground.
    shadow = apply_h(H_field, pixels)
    if not np.isfinite(shadow).all():
        raise SystemExit(
            "a car target maps to infinity through the field plane — it is on the "
            "horizon, where the ground mapping diverges"
        )

    k, b = fit_homology(shadow, world_mm)
    if not np.isfinite(k) or k <= 1.0:
        raise SystemExit(
            f"the targets imply a scale of {k:.6f}; a camera looking down on a plane "
            f"{height_mm:.0f} mm up must give more than 1. Check that the targets really "
            "are at sticker height and that their world coordinates are the ground marks "
            "directly below them."
        )
    camera_height = k * height_mm / (k - 1.0)
    if camera_height > MAX_CAMERA_MM:
        raise SystemExit(
            f"the targets imply a camera {camera_height / 1000:.0f} m above the ground, so "
            f"they are barely displaced from the marks below them. Targets still lying on "
            f"the ground give exactly this — check they were raised to {height_mm:.0f} mm."
        )

    # world_car = (world_field - b) / k
    M = np.array([[1.0 / k, 0.0, -b[0] / k], [0.0, 1.0 / k, -b[1] / k], [0.0, 0.0, 1.0]])
    H_car = M @ H_field

    res = np.linalg.norm(apply_h(H_car, pixels) - world_mm, axis=1)
    camera_xy = b / (1.0 - k)
    return {
        "homography": H_car,
        "scale": k,
        "camera_ground_mm": [float(camera_xy[0]), float(camera_xy[1])],
        "camera_height_mm": float(camera_height),
        "residuals_mm": [float(d) for d in res],
        "rms_mm": float(np.sqrt(np.mean(res ** 2))),
        "max_mm": float(res.max()),
    }


def car_plane(cal: dict, height_mm: float):
    """The plane to read a marker on at ``height_mm``, and a note on where it came from.

    A surveyed car plane wins at the height it was surveyed at, because it
    measured the geometry instead of assuming the pose is exact. It is only
    right at that one height, though, so a marker at any other height falls
    back to raising Z through the pose.
    """
    K, _, R, t = calibration_planes(cal)
    stored = cal.get("car")
    if stored and abs(float(stored["height_mm"]) - float(height_mm)) <= 1.0:
        H = np.array(stored["homography"], dtype=np.float64)
        return H, f"surveyed car plane @ {float(stored['height_mm']):.0f} mm"
    note = f"synthesised @ {height_mm:.0f} mm"
    if stored:
        note += f" (surveyed plane is at {float(stored['height_mm']):.0f} mm, not this height)"
    return homography_at_height(K, R, t, height_mm), note


def bev_around(und, H_plane, centre_mm, span_mm, mm_per_px):
    """A metric bird's-eye raster of one square of ground, centred on a point.

    Sized from a point the operator chose rather than from the whole frame: on a
    tilted camera the visible ground runs to the horizon, and a raster covering
    all of it is both enormous and mostly useless.
    """
    x0, y0 = centre_mm[0] - span_mm, centre_mm[1] - span_mm
    n = math.ceil(2 * span_mm / mm_per_px)
    y_max = y0 + n * mm_per_px
    w2b = np.array([[1 / mm_per_px, 0, -x0 / mm_per_px],
                    [0, -1 / mm_per_px, y_max / mm_per_px],
                    [0, 0, 1.0]])
    Hp2b = w2b @ H_plane
    raster = cv2.warpPerspective(und, Hp2b, (n, n), flags=cv2.INTER_LINEAR)
    return raster, w2b, np.linalg.inv(w2b)


def plane_mm_per_px(H_plane: np.ndarray, px) -> float:
    """Local ground sample distance, sqrt(|det J|), by central differences."""
    p = np.asarray(px, dtype=np.float64).reshape(1, 2)
    ex, ey = np.array([[0.5, 0.0]]), np.array([[0.0, 0.5]])
    dx = apply_h(H_plane, p + ex)[0] - apply_h(H_plane, p - ex)[0]
    dy = apply_h(H_plane, p + ey)[0] - apply_h(H_plane, p - ey)[0]
    return float(math.sqrt(abs(np.linalg.det(np.column_stack([dx, dy])))))


# --------------------------------------------------------------------------
# 0. frame
# --------------------------------------------------------------------------


def cmd_frame(a) -> None:
    """Grab one still. Every survey step below works on a single frame."""
    frame = read_frame(a.video, a.at)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(a.out), frame)
    print(f"  {frame.shape[1]}x{frame.shape[0]} at {a.at}s")
    print(f"  wrote {a.out}", flush=True)


# --------------------------------------------------------------------------
# 1. intrinsics
# --------------------------------------------------------------------------


def cmd_intrinsics(a) -> None:
    """Lens calibration from a checkerboard video.

    ``--board`` counts *inner corners*, not squares: a board with 12x8 squares
    has 11x7 inner corners, and getting this wrong makes every view fail to
    detect with no other symptom.

    Two filters decide which views are kept, and both matter more than the
    number of frames. Blur is rejected outright. Then a view is skipped unless
    the board has actually moved since the last accepted one — a hundred frames
    of the same pose constrain nothing, and they make the reported RMS look
    better while the calibration gets no better at all.

    ``--model`` picks the lens model, and it has to match the lens. The pinhole
    model's two radial terms cannot represent a wide fisheye: the fit converges,
    reports a plausible RMS on the views it was given, and then bends straight
    lines near the frame edge. Every consumer in this tool reads the ``model``
    field written here and undistorts accordingly, so the choice made at this
    step follows the calibration everywhere — which is also why it must not be
    guessed. If straight edges stay straight after ``frame``, it is pinhole.
    """
    cols, rows = (int(v) for v in a.board.lower().split("x"))
    pattern = (cols, rows)
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * a.square_mm

    cap = cv2.VideoCapture(str(a.video))
    if not cap.isOpened():
        raise SystemExit(f"could not open {a.video}")
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
    term = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)

    obj_pts, img_pts, kept, size = [], [], [], None
    idx, seen, blurry = -1, 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if idx % a.every:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]
        if cv2.Laplacian(gray, cv2.CV_64F).var() < a.min_sharpness:
            blurry += 1
            continue
        found, corners = cv2.findChessboardCorners(gray, pattern, flags)
        if not found:
            continue
        seen += 1
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), term)
        c = corners.reshape(-1, 2)
        centre, area = c.mean(axis=0), cv2.contourArea(cv2.convexHull(c.astype(np.float32)))
        # Keep only views that add information: a new pose, not a new frame of
        # the same one. Distance is scaled by the board's apparent size so the
        # test means the same thing near and far.
        span = math.sqrt(max(area, 1.0))
        if any(np.hypot(*(centre - k[0])) < 0.35 * span and abs(area / max(k[1], 1) - 1) < 0.15
               for k in kept):
            continue
        kept.append((centre, area))
        obj_pts.append(objp.copy())
        img_pts.append(corners)
        if a.max_views and len(obj_pts) >= a.max_views:
            break
    cap.release()

    print(f"  {seen} detections, {blurry} rejected as blurry, {len(obj_pts)} distinct poses kept")
    if len(obj_pts) < 6:
        raise SystemExit(
            f"only {len(obj_pts)} usable views. Check --board (inner corners, not squares) "
            "and shoot the board at more angles and distances, pausing at each pose."
        )

    if a.model == "fisheye":
        # The fisheye fitter is strict where the pinhole one is forgiving: it
        # wants (1, N, ...) float64 for both point sets, and rejects the
        # (N, 1, ...) float32 arrays findChessboardCorners hands back.
        obj_f = [o.reshape(1, -1, 3).astype(np.float64) for o in obj_pts]
        img_f = [i.reshape(1, -1, 2).astype(np.float64) for i in img_pts]
        K, D = np.zeros((3, 3)), np.zeros((4, 1))
        rms, K, D, _, _ = cv2.fisheye.calibrate(
            obj_f, img_f, size, K, D,
            flags=_FISHEYE_CALIB_FLAGS,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-6))
    else:
        rms, K, D, _, _ = cv2.calibrateCamera(obj_pts, img_pts, size, None, None)
    print(f"  {a.model} model, reprojection RMS {rms:.4f} px over {len(obj_pts)} views")
    if rms > 1.0:
        print("  NOTE: RMS above 1 px. The usual cause is a rolling shutter: a board swept "
              "through the frame comes back sheared and pin-sharp, so the blur gate misses "
              "it. Pause at every pose and re-shoot.", flush=True)
    save_json(a.out, {
        "model": a.model,
        "camera_matrix": K.tolist(),
        "dist_coeffs": D.ravel().tolist(),
        "image_width": int(size[0]),
        "image_height": int(size[1]),
        "rms_reproj_px": float(rms),
        "n_views": len(obj_pts),
        "board": {"inner_corners": [cols, rows], "square_mm": a.square_mm},
    })


# --------------------------------------------------------------------------
# 2. gcp -> calibration
# --------------------------------------------------------------------------


def cmd_gcp(a) -> None:
    """Click ground control points, type their world coordinates, solve the pose.

    The world frame is yours: pick an origin and an axis direction on the tarmac
    and stay consistent. Z is zero for every point here — these are marks on the
    ground, and the plane they define is what every later measurement is read
    against.

    Four points is the minimum and is exactly determined; six or more spread
    across the working area is what makes the residuals mean anything. Points
    clustered in one corner will fit beautifully and be wrong everywhere else.
    """
    intr = load_json(a.intrinsics)
    K, _, _ = intrinsics_arrays(intr)
    und = undistort(load_image(a.image), intr, str(a.image))

    picked = picker.pick_points(
        und, "Ground control points — click a mark, type its world X,Y in mm",
        world=True, hint="4 or more, spread across the working area",
        open_browser=not a.no_open)
    if not picked:
        raise SystemExit("cancelled — nothing surveyed")
    if len(picked) < 4:
        raise SystemExit(f"need at least 4 points to solve a pose, got {len(picked)}")

    img_pts = np.array([p["px"] for p in picked], dtype=np.float64)
    world = np.array([p["world_mm"] for p in picked], dtype=np.float64)
    obj_pts = np.column_stack([world, np.zeros(len(world))])

    # The frame was already undistorted, so PnP sees an ideal pinhole camera:
    # pass K with zero distortion, or the lens correction is applied twice.
    zero = np.zeros(5)
    flag = cv2.SOLVEPNP_IPPE if len(picked) >= 4 else cv2.SOLVEPNP_ITERATIVE
    ok, rvec, tvec = cv2.solvePnP(obj_pts.astype(np.float64), img_pts, K, zero, flags=flag)
    if not ok:
        raise SystemExit("solvePnP failed; check that the world coordinates match the clicks")
    rvec, tvec = cv2.solvePnPRefineLM(obj_pts, img_pts, K, zero, rvec, tvec)
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.ravel()

    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, zero)
    res_px = np.linalg.norm(proj.reshape(-1, 2) - img_pts, axis=1)
    H_field = homography_at_height(K, R, t, 0.0)
    res_mm = np.linalg.norm(apply_h(H_field, img_pts) - world, axis=1)
    centre = (-R.T @ t)
    tilt = float(np.rad2deg(np.arccos(np.clip(-R[2, 2], -1.0, 1.0))))

    print(f"  camera at ({centre[0]:.0f}, {centre[1]:.0f}, {centre[2]:.0f}) mm, "
          f"tilt {tilt:.1f}° from nadir")
    print(f"  reprojection  rms {res_px.mean():.2f} px   max {res_px.max():.2f} px")
    print(f"  on the ground rms {res_mm.mean():.1f} mm  max {res_mm.max():.1f} mm")
    for p, r in zip(picked, res_mm, strict=True):
        print(f"    ({p['world_mm'][0]:8.1f}, {p['world_mm'][1]:8.1f}) mm   {r:6.1f} mm")
    if res_mm.max() > 50:
        print("  NOTE: a residual over 50 mm usually means a mistyped world coordinate or a "
              "click on the wrong mark, not a bad camera. Check the worst point above.",
              flush=True)

    save_json(a.out, {
        "image_width": int(und.shape[1]),
        "image_height": int(und.shape[0]),
        "intrinsics": intr,
        "field": {
            "name": "field",
            "height_mm": 0.0,
            "homography": H_field.tolist(),
            "rms_error_mm": float(res_mm.mean()),
            "max_error_mm": float(res_mm.max()),
            "gcps": [{"pixel": {"x": p["px"][0], "y": p["px"][1]},
                      "world": {"x_mm": p["world_mm"][0], "y_mm": p["world_mm"][1]},
                      "residual_mm": float(r)}
                     for p, r in zip(picked, res_mm, strict=True)],
            "pose": {
                "rotation": R.tolist(),
                "translation_mm": t.tolist(),
                "center_mm": centre.tolist(),
                "tilt_deg": tilt,
                "reproj_rms_px": float(res_px.mean()),
                "solver": "IPPE+LM",
            },
        },
    })
    print("  next: check it with `measure`, then cut the sticker template", flush=True)


# --------------------------------------------------------------------------
# 3. measure — verify before trusting
# --------------------------------------------------------------------------


def cmd_measure(a) -> None:
    """Click two points on the ground and read the distance back in millimetres.

    The residuals the `gcp` step prints only say the fit is self-consistent at
    the marks it was given. This says whether the ground between them is right,
    which is the number a tape can argue with.
    """
    cal = load_json(a.calibration)
    K, _, R, t = calibration_planes(cal)
    H = homography_at_height(K, R, t, a.height_mm)
    und = undistort(load_image(a.image), cal["intrinsics"], str(a.image))
    pts = picker.pick_points(und, f"Measure on the plane at {a.height_mm:.0f} mm — click two points",
                             n=2, hint="then compare with a tape", open_browser=not a.no_open)
    if not pts:
        raise SystemExit("cancelled")
    w = apply_h(H, np.array([p["px"] for p in pts]))
    d = float(np.linalg.norm(w[1] - w[0]))
    print(f"  A ({w[0][0]:+.1f}, {w[0][1]:+.1f}) mm")
    print(f"  B ({w[1][0]:+.1f}, {w[1][1]:+.1f}) mm")
    print(f"  distance {d:.1f} mm = {d / 1000:.3f} m", flush=True)


# --------------------------------------------------------------------------
# 3b. carplane — measure the plane the marker rides on
# --------------------------------------------------------------------------


def cmd_carplane(a) -> None:
    """Survey the marker's plane directly, instead of raising Z through the pose.

    Stand a pole at the marker height over each of two or more ground marks and
    click the top of each one. The world coordinate you type is the *ground
    mark beneath the pole*, not the pole top — the displacement between the two
    is exactly the parallax this plane exists to remove, and it is what the fit
    reads the camera out of.

    Intrinsics do the same two jobs here as everywhere else in this tool: they
    undistort the frame you click on, and with the pose they give the field
    homography this plane is a homology of. Because both are present, the fit
    can be checked in a way the survey alone cannot — the camera height it
    recovers is compared against the pose, and the two disagreeing means one of
    the surveys is wrong.

    Writes a ``car`` block into the calibration file, in place unless you pass
    ``--out``. Both pipelines pick it up on their own for a marker at this
    height.
    """
    cal = load_json(a.calibration)
    K, _, R, t = calibration_planes(cal)
    H_field = homography_at_height(K, R, t, 0.0)
    und = undistort(load_image(a.image), cal["intrinsics"], str(a.image))

    picked = picker.pick_points(
        und, f"Car-plane targets at {a.height_mm:.0f} mm — click each pole top, "
             "type the world X,Y of the mark beneath it",
        world=True, hint="2 minimum; 3 or more makes the residual a real check",
        open_browser=not a.no_open)
    if not picked:
        raise SystemExit("cancelled — nothing surveyed")

    pixels = np.array([p["px"] for p in picked], dtype=np.float64)
    world = np.array([p["world_mm"] for p in picked], dtype=np.float64)
    fit = car_plane_from_field(H_field, pixels, world, a.height_mm)

    pose_centre = (-np.asarray(R).T @ np.asarray(t))
    print(f"  scale k {fit['scale']:.6f}  ->  parallax {1000 * (fit['scale'] - 1):.0f} mm "
          f"per metre from the camera")
    print(f"  camera over ({fit['camera_ground_mm'][0]:.0f}, {fit['camera_ground_mm'][1]:.0f}) mm "
          f"at {fit['camera_height_mm']:.0f} mm")
    print(f"  pose says  ({pose_centre[0]:.0f}, {pose_centre[1]:.0f}) mm "
          f"at {pose_centre[2]:.0f} mm")
    print(f"  fit residual rms {fit['rms_mm']:.1f} mm  max {fit['max_mm']:.1f} mm")
    for p, r in zip(picked, fit["residuals_mm"], strict=True):
        print(f"    ({p['world_mm'][0]:8.1f}, {p['world_mm'][1]:8.1f}) mm   {r:6.1f} mm")

    if len(picked) < 3:
        print("  NOTE: 2 targets fit 3 parameters with almost nothing to spare, so the "
              "residual barely tests the survey. A third target makes it a real check.",
              flush=True)
    if fit["max_mm"] > a.max_residual_mm:
        raise SystemExit(
            f"car-plane residual too large: worst target off by {fit['max_mm']:.1f} mm "
            f"(limit {a.max_residual_mm:.1f} mm). Check that every target stood at the "
            "same height and above the mark it names."
        )

    drop = abs(fit["camera_height_mm"] - pose_centre[2])
    if drop > a.max_camera_disagreement_mm:
        print(f"  NOTE: this survey and the camera pose disagree about the camera height by "
              f"{drop:.0f} mm. They are independent measurements of the same mast, so one of "
              f"them is wrong — check the pole height and the ground marks before trusting "
              f"either plane.", flush=True)

    cal["car"] = {
        "name": "car",
        "height_mm": float(a.height_mm),
        "homography": fit["homography"].tolist(),
        "fit_model": "homology",
        "scale": fit["scale"],
        "camera_ground_mm": fit["camera_ground_mm"],
        "camera_height_mm": fit["camera_height_mm"],
        "pose_camera_height_mm": float(pose_centre[2]),
        "rms_error_mm": fit["rms_mm"],
        "max_error_mm": fit["max_mm"],
        "targets": [{"pixel": {"x": p["px"][0], "y": p["px"][1]},
                     "world": {"x_mm": p["world_mm"][0], "y_mm": p["world_mm"][1]},
                     "residual_mm": r}
                    for p, r in zip(picked, fit["residuals_mm"], strict=True)],
    }
    save_json(a.out or a.calibration, cal)
    print("  next: cut the sticker template at this height", flush=True)


# --------------------------------------------------------------------------
# 4. sticker template
# --------------------------------------------------------------------------


def cmd_sticker(a) -> None:
    """Cut the marker template the detector will match.

    Bird's-eye is the one both pipelines want. On a raster at a fixed
    millimetres-per-pixel the marker is the same size and shape wherever the car
    stands, so the template stays valid across the whole site — and it carries a
    physical scale, which is what lets pipeline 2 project it back into the
    camera frame for itself.

    A raw crop is offered for inspection. It is only strictly right at the place
    it was cut, because an oblique camera makes the marker smaller further away
    and shears it, so do not feed one to a pipeline expecting metric pixels.
    """
    cal = load_json(a.calibration)
    H_car, plane_note = car_plane(cal, a.height_mm)
    print(f"  marker plane: {plane_note}")
    und = undistort(load_image(a.image), cal["intrinsics"], str(a.image))

    if a.space == "raw":
        mmpp = plane_mm_per_px(H_car, [und.shape[1] / 2, und.shape[0] / 2])
        box = picker.pick_box(und, "Box the marker in the camera frame", mm_per_px=mmpp,
                              hint="raw crop — for inspection, not for a pipeline",
                              open_browser=not a.no_open)
        if box is None:
            raise SystemExit("cancelled")
        x, y, w, h = box
        crop = und[round(y):round(y + h), round(x):round(x + w)]
        mmpp = plane_mm_per_px(H_car, [x + w / 2, y + h / 2])
    else:
        seed = picker.pick_points(und, "Click roughly where the marker is", n=1,
                                  hint="just to centre the bird's-eye view",
                                  open_browser=not a.no_open)
        if not seed:
            raise SystemExit("cancelled")
        centre_mm = apply_h(H_car, np.array([seed[0]["px"]]))[0]
        mmpp = a.mm_per_px or round(plane_mm_per_px(H_car, seed[0]["px"]), 2)
        raster, _, _ = bev_around(und, H_car, centre_mm, a.span_mm, mmpp)
        print(f"  bird's-eye {raster.shape[1]}x{raster.shape[0]} px @ {mmpp:g} mm/px "
              f"around ({centre_mm[0]:.0f}, {centre_mm[1]:.0f}) mm")
        box = picker.pick_box(raster, "Box the marker on the bird's-eye raster", mm_per_px=mmpp,
                              hint="check the mm readout against the printed marker",
                              open_browser=not a.no_open)
        if box is None:
            raise SystemExit("cancelled")
        x, y, w, h = box
        crop = raster[round(y):round(y + h), round(x):round(x + w)]

    if crop.size == 0 or min(crop.shape[:2]) < 8:
        raise SystemExit(f"that box is too small ({crop.shape[1]}x{crop.shape[0]} px)")
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    if float(gray.std()) < 5:
        raise SystemExit("that region has almost no contrast; template matching needs structure")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(a.out), crop)
    meta_path = Path(a.out).with_suffix(".json")
    print(f"  wrote {a.out}  ({crop.shape[1]}x{crop.shape[0]} px = "
          f"{crop.shape[1] * mmpp:.0f}x{crop.shape[0] * mmpp:.0f} mm)")
    save_json(meta_path, {
        "template": str(a.out),
        "space": a.space,
        "mm_per_px": float(mmpp),
        "width_mm": float(crop.shape[1] * mmpp),
        "height_mm": float(crop.shape[0] * mmpp),
        "sticker_height_mm": float(a.height_mm),
    })
    print(f"  pass --template-mm-per-px {mmpp:g} to the pipelines", flush=True)


# --------------------------------------------------------------------------
# 5. car geometry
# --------------------------------------------------------------------------


def cmd_car(a) -> None:
    """Vehicle geometry, measured with a tape and oriented from the image.

    The split is deliberate. Lengths come from a tape because every offset
    transfers one-for-one into a bumper position and the image cannot see the
    ground under the car anyway. Orientation comes from two clicks, because it
    is the one quantity that is easy to get catastrophically wrong by typing.

    What is stored is *not* the car frame. It is the sticker template's frame:
    origin at the marker centre, +X along the template's width axis. The car's
    forward axis sits at ``sticker_yaw_offset_deg`` within it, and a marker
    printed across the roof rather than along it has an offset near ±90. Get
    that wrong and a pixel-perfect detection draws the box at right angles
    across the car — which still looks like a car, so nothing downstream catches
    it. Clicking the nose is what makes it impossible to get backwards.
    """
    cal = load_json(a.calibration)
    H_car, plane_note = car_plane(cal, a.sticker_height_mm)
    print(f"  marker plane: {plane_note}")
    und = undistort(load_image(a.image), cal["intrinsics"], str(a.image))

    seed = picker.pick_points(und, "Click the marker on the roof", n=1,
                              hint="just to centre the bird's-eye view",
                              open_browser=not a.no_open)
    if not seed:
        raise SystemExit("cancelled")
    approx = apply_h(H_car, np.array([seed[0]["px"]]))[0]
    mmpp = a.mm_per_px or round(plane_mm_per_px(H_car, seed[0]["px"]), 2)
    raster, _, b2w = bev_around(und, H_car, approx, a.span_mm, mmpp)

    two = picker.pick_points(
        raster, "Click 1) the marker centre, then 2) the centre of the front bumper", n=2,
        hint="the second click only sets which way the car faces",
        open_browser=not a.no_open)
    if not two:
        raise SystemExit("cancelled")
    centre_mm, front_mm = apply_h(b2w, np.array([p["px"] for p in two]))
    d = front_mm - centre_mm
    if np.linalg.norm(d) < 200:
        raise SystemExit("those two clicks are almost the same point; the heading is undefined")
    yaw = float(np.degrees(math.atan2(d[1], d[0])))
    yaw = (yaw + 180.0) % 360.0 - 180.0

    front = a.sticker_to_front_mm
    rear = front - a.length_mm
    half_track, half_w = a.track_mm / 2.0, a.width_mm / 2.0
    front_axle = front - (a.length_mm - a.wheelbase_mm) / 2.0
    rear_axle = front_axle - a.wheelbase_mm
    body_car = np.array([[front, half_w], [front, -half_w], [rear, -half_w], [rear, half_w]])
    wheels_car = {"fl": (front_axle, half_track), "fr": (front_axle, -half_track),
                  "rl": (rear_axle, half_track), "rr": (rear_axle, -half_track)}

    # Car frame (+X forward) -> template frame: the car's forward axis sits at
    # +yaw within the template frame, so the measurements rotate by exactly that.
    ang = math.radians(yaw)
    c, s = math.cos(ang), math.sin(ang)
    M = np.array([[c, -s], [s, c]])
    body = body_car @ M.T
    wheels = {k: (M @ np.array(v)) for k, v in wheels_car.items()}

    print(f"  marker at ({centre_mm[0]:.0f}, {centre_mm[1]:.0f}) mm, "
          f"car points {yaw:+.2f}° from the template's +X axis")
    if abs(abs(yaw) - 90.0) < 20:
        print("  (a quarter turn — the marker's long axis runs along the car)")
    save_json(a.out, {
        "car_id": a.name,
        "sticker_height_mm": float(a.sticker_height_mm),
        "sticker_yaw_offset_deg": yaw,
        "length_mm": float(a.length_mm),
        "width_mm": float(a.width_mm),
        "body_polygon_mm": [{"x_mm": float(p[0]), "y_mm": float(p[1])} for p in body],
        "wheels_mm": {k: {"x_mm": float(v[0]), "y_mm": float(v[1])} for k, v in wheels.items()},
        "front_bumper_mm": float(front),
        "rear_bumper_mm": float(rear),
        "note": "body_polygon_mm and wheels_mm are in the sticker TEMPLATE frame; "
                "front/rear_bumper_mm are car-frame distances",
    })


# --------------------------------------------------------------------------
# 6. ROIs
# --------------------------------------------------------------------------


def cmd_roi(a) -> None:
    """Draw the things the car gets measured against: kerbs, lines, bays, cones.

    Drawn on the undistorted frame, so the pixels written here are already in
    the frame the pipelines work in — no `--roi-distorted` needed. They are read
    on the *ground* plane, because that is where paint and kerbs are, whatever
    plane the marker is read on.
    """
    cal = load_json(a.calibration)
    und = undistort(load_image(a.image), cal["intrinsics"], str(a.image))
    shapes = picker.pick_shapes(und, "Draw what the car is measured against",
                                hint="click points, then p / l / g to finish a shape",
                                open_browser=not a.no_open)
    if not shapes:
        raise SystemExit("cancelled — nothing drawn")

    H_field = np.array(cal["field"]["homography"])
    for s in shapes:
        w = apply_h(H_field, np.array(s["points_px"]))
        span = (f", {np.linalg.norm(w[-1] - w[0]):.0f} mm end to end" if len(w) > 1 else "")
        print(f"  {s['name']:12s} {s['type']:8s} {len(w)} pt  "
              f"first at ({w[0][0]:+.0f}, {w[0][1]:+.0f}) mm{span}")
    save_json(a.out, {
        "_comment": "pixels are in the UNDISTORTED frame; read on the ground plane",
        "rois": shapes,
    })


# --------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name, fn, help_):
        p = sub.add_parser(name, help=help_, description=fn.__doc__,
                           formatter_class=argparse.RawDescriptionHelpFormatter)
        p.set_defaults(fn=fn)
        p.add_argument("--no-open", action="store_true", help="Print the picker URL, do not open a browser.")
        return p

    p = add("frame", cmd_frame, "0. grab one still to survey on")
    p.add_argument("--video", required=True, type=Path)
    p.add_argument("--at", type=float, default=0.0, help="Seconds into the video.")
    p.add_argument("--out", required=True, type=Path)

    p = add("intrinsics", cmd_intrinsics, "1. lens calibration from a checkerboard video")
    p.add_argument("--video", required=True, type=Path)
    p.add_argument("--board", required=True, help="INNER corners, e.g. 11x7 for a 12x8 board.")
    p.add_argument("--square-mm", required=True, type=float)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--every", type=int, default=5, help="Look at every Nth frame.")
    p.add_argument("--min-sharpness", type=float, default=60.0, help="Laplacian variance floor.")
    p.add_argument("--max-views", type=int, default=40)
    p.add_argument("--model", choices=("pinhole", "fisheye"), default="pinhole",
                   help="Lens model to fit. Must match the lens; see the step description.")

    p = add("gcp", cmd_gcp, "2. click ground points -> calibration.json")
    p.add_argument("--image", required=True, type=Path)
    p.add_argument("--intrinsics", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)

    p = add("measure", cmd_measure, "3. check the calibration against a tape")
    p.add_argument("--image", required=True, type=Path)
    p.add_argument("--calibration", required=True, type=Path)
    p.add_argument("--height-mm", type=float, default=0.0, help="Plane to measure on.")

    p = add("carplane", cmd_carplane, "3b. survey the plane the marker rides on")
    p.add_argument("--image", required=True, type=Path)
    p.add_argument("--calibration", required=True, type=Path)
    p.add_argument("--height-mm", required=True, type=float, help="Target height above ground.")
    p.add_argument("--out", type=Path, default=None,
                   help="Default: update the calibration file in place.")
    p.add_argument("--max-residual-mm", type=float, default=20.0)
    p.add_argument("--max-camera-disagreement-mm", type=float, default=500.0,
                   help="Warn if this survey and the pose disagree about the camera height.")

    p = add("sticker", cmd_sticker, "4. cut the marker template")
    p.add_argument("--image", required=True, type=Path)
    p.add_argument("--calibration", required=True, type=Path)
    p.add_argument("--height-mm", required=True, type=float, help="Marker height above ground.")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--space", choices=("bev", "raw"), default="bev")
    p.add_argument("--mm-per-px", type=float, default=None, help="Default: the local ground sample distance.")
    p.add_argument("--span-mm", type=float, default=6000.0, help="Half-width of the bird's-eye view.")

    p = add("car", cmd_car, "5. vehicle geometry relative to the marker")
    p.add_argument("--image", required=True, type=Path)
    p.add_argument("--calibration", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--name", default="car")
    p.add_argument("--sticker-height-mm", required=True, type=float)
    p.add_argument("--length-mm", required=True, type=float)
    p.add_argument("--width-mm", required=True, type=float)
    p.add_argument("--sticker-to-front-mm", required=True, type=float,
                   help="Marker centre to the front bumper, along the car.")
    p.add_argument("--wheelbase-mm", required=True, type=float)
    p.add_argument("--track-mm", required=True, type=float)
    p.add_argument("--mm-per-px", type=float, default=None)
    p.add_argument("--span-mm", type=float, default=6000.0)

    p = add("roi", cmd_roi, "6. draw what to measure the car against")
    p.add_argument("--image", required=True, type=Path)
    p.add_argument("--calibration", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)

    a = ap.parse_args()
    print(f"{a.cmd}", flush=True)
    a.fn(a)


if __name__ == "__main__":
    main()
