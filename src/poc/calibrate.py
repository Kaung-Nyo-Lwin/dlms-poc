"""Survey tool: everything a pipeline run needs, produced in one folder.

Run the steps in order. Each one writes a file; the next one reads it. There are
no stations, no vehicle ids and no workspace — every input and output is a path
you choose, so a whole site lives in one directory.

    0  frame       grab one still to survey on
    1  intrinsics  checkerboard video      -> intrinsics.json
    2  gcp         click ground points     -> calibration.json
    3  measure     check it against a tape (no output; do this before trusting it)
    3b carplane    survey the marker's plane  -> car block in calibration.json
    3c tape        check the survey's scale against tapes, and correct it
    4  sticker     cut the marker template -> sticker.png + sticker.json
    5  car         vehicle geometry        -> car.json  (tape-measured)
    5b outline     trace the car's box     -> car.json  (clicked; prefer this)
    6  roi         draw what to measure to -> rois.json
    7  rules       draw the DSR scoring regions -> rules.json (then `score.py`)

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
import os
import tempfile
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
    """Write a survey file, complete or not at all.

    The `carplane` step defaults its ``--out`` to the calibration it just read,
    which is the one write here that is genuinely in place. Truncating that and
    then failing would cost the whole ground survey, so the file is built beside
    itself and renamed over — a rename within one directory either happens or
    does not.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.",
                                    suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(json.dumps(data, indent=2) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
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


def homography_at_height(K, R, t, height_mm: float) -> np.ndarray:
    """Image -> world (X, Y) for the horizontal plane at ``height_mm`` above the marks."""
    centre, up = camera_from_pose(R, t)
    if float(height_mm) >= abs(float(centre[2])) - 1.0:
        raise SystemExit(
            f"a plane {height_mm:.0f} mm up is at or above the camera itself "
            f"({abs(centre[2]):.0f} mm); there is no bird's-eye view of it"
        )
    Hw2i = K @ np.column_stack([R[:, 0], R[:, 1], R[:, 2] * (up * float(height_mm)) + t])
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


#: Widest bird's-eye raster the picker will build, in pixels a side.
#:
#: Not a transport limit. Measured on a real station frame, a 7000 px raster is
#: a 17 MB PNG that encodes in a second and crosses a loopback socket instantly;
#: rectified views are mostly black margin and smooth tarmac, so they compress
#: far better than their pixel count suggests. This is the browser's own canvas
#: limit, which several engines put at 8192 px a side — past that the page has
#: nothing to draw on, and no amount of patience helps.
MAX_BEV_PX = 8000


def fit_raster(span_mm, mm_per_px, min_span_mm):
    """Trim a bird's-eye view to something the picker can actually serve.

    Sample distance is a property of the plane, not a choice: a plane close to
    the camera is finer, so a span that was comfortable on the tarmac wants many
    more pixels on a roof. Almost always that is fine — see :data:`MAX_BEV_PX`,
    the limit is the browser's canvas rather than anything to do with size — and
    this returns its arguments untouched. Only past that does something have to
    give, and which one is not arbitrary.

    **Extent goes first.** The millimetres-per-pixel is what every measurement
    downstream is expressed in — for the `sticker` step it is literally recorded
    and handed to the pipeline — so coarsening it costs precision in the answer.
    The span costs only how much spare tarmac is in shot, and by the time this
    is called the operator has already clicked where the thing is. Only when the
    view would fall below what the step actually needs to see is resolution
    given up instead.

    Returns ``(span_mm, mm_per_px, note)``; the note is None when nothing had to
    give.
    """
    if math.ceil(2 * span_mm / mm_per_px) <= MAX_BEV_PX:
        return span_mm, mm_per_px, None
    fitted = MAX_BEV_PX * mm_per_px / 2.0
    if fitted >= min_span_mm:
        return fitted, mm_per_px, (
            f"view narrowed to {2 * fitted / 1000:.1f} m across at {mm_per_px:g} mm/px "
            f"— full detail kept, and the raster stays clickable")
    coarse = 2.0 * min_span_mm / MAX_BEV_PX
    return min_span_mm, coarse, (
        f"view held at {2 * min_span_mm / 1000:.1f} m across but sampled at "
        f"{coarse:.2f} mm/px instead of {mm_per_px:g} — this plane is too fine to "
        f"show that much at full detail")


def bev_around(und, H_plane, centre_mm, span_mm, mm_per_px, y_up=True):
    """A metric bird's-eye raster of one square of ground, centred on a point.

    Sized from a point the operator chose rather than from the whole frame: on a
    tilted camera the visible ground runs to the horizon, and a raster covering
    all of it is both enormous and mostly useless.

    ``y_up`` puts +Y at the top, which is the convention everywhere a map is
    drawn and the right one for a world frame whose +Z comes out of the ground.
    A frame laid out the other way — see :func:`camera_from_pose` — is seen
    mirrored from above, and drawing it this way would hand back a raster in
    which every glyph reads backwards. That is not cosmetic: the marker template
    is cut from one of these rasters and correlated against another, and a
    mirrored template matches nothing at any rotation.
    """
    x0, y0 = centre_mm[0] - span_mm, centre_mm[1] - span_mm
    n = math.ceil(2 * span_mm / mm_per_px)
    y_max = y0 + n * mm_per_px
    w2b = (np.array([[1 / mm_per_px, 0, -x0 / mm_per_px],
                     [0, -1 / mm_per_px, y_max / mm_per_px],
                     [0, 0, 1.0]])
           if y_up else
           np.array([[1 / mm_per_px, 0, -x0 / mm_per_px],
                     [0, 1 / mm_per_px, -y0 / mm_per_px],
                     [0, 0, 1.0]]))
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
    centre, up = camera_from_pose(R, t)
    tilt = float(np.rad2deg(np.arccos(np.clip(-R[2, 2] * up, -1.0, 1.0))))

    print(f"  camera at ({centre[0]:.0f}, {centre[1]:.0f}, {centre[2] * up:.0f}) mm above "
          f"the marks, tilt {tilt:.1f}° from nadir")
    if up < 0:
        print("  NOTE: the world frame you typed is left-handed — +X cross +Y points into "
              "the tarmac, not out of it. Nothing 2D is affected and the ground plane is "
              "exact, so this is not an error to fix; every height is simply taken along "
              "the camera's side of the ground from here on.", flush=True)
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

    pose_centre, up = camera_from_pose(R, t)
    pose_centre = np.array([pose_centre[0], pose_centre[1], pose_centre[2] * up])
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
# 3c. tape — argue with the survey's own scale
# --------------------------------------------------------------------------


#: What one click on a painted mark is worth, in pixels, after sub-pixel snapping.
DEFAULT_SIGMA_PX = 0.5
#: What a tape stretched between two marks is worth, in millimetres.
DEFAULT_SIGMA_TAPE_MM = 5.0
#: How wrong a *typed* ground-control coordinate is allowed to be. This is the
#: knob that decides whether the tapes or the original survey wins; see
#: `refine_free_network` for why the adjustment is nothing without it.
DEFAULT_SIGMA_WORLD_MM = 30.0

#: chi-square 95% critical values for 1..10 degrees of freedom. Past that the
#: Wilson-Hilferty approximation takes over — it is within 1% by dof 4, and
#: nobody lays eleven tapes.
_CHI2_95 = (3.84, 5.99, 7.81, 9.49, 11.07, 12.59, 14.07, 15.51, 16.92, 18.31)


def chi2_95(dof: int) -> float:
    if 1 <= dof <= len(_CHI2_95):
        return _CHI2_95[dof - 1]
    return dof * (1.0 - 2.0 / (9 * dof) + 1.645 * math.sqrt(2.0 / (9 * dof))) ** 3


def image_from_world(K, R, t) -> np.ndarray:
    """World (X, Y) on the ground -> image pixels; the inverse of ``homography_at_height(…, 0)``.

    Written out rather than inverted back, so the adjustment below optimises
    exactly the map every other measurement in this tool is read through.
    """
    R = np.asarray(R, dtype=np.float64)
    return np.asarray(K, dtype=np.float64) @ np.column_stack(
        [R[:, 0], R[:, 1], np.asarray(t, dtype=np.float64).ravel()])


def ground_scale_along(H, p, u) -> float:
    """Millimetres of ground travel per pixel of click error at ``p``, along ``u``.

    The ground sample distance is not one number: on a station view it varies by
    a factor of five across the frame and it is anisotropic. What a length
    cares about is the component along its own direction, which is what this
    returns — and getting it wrong treats a bar in the far field, where a pixel
    is 15 mm, as if it were as trustworthy as one under the camera, where it
    is 2.
    """
    p = np.asarray(p, dtype=np.float64).reshape(2)
    J = np.column_stack([apply_h(H, p + e).ravel() - apply_h(H, p - e).ravel()
                         for e in (np.array([0.5, 0.0]), np.array([0.0, 0.5]))])
    return float(np.linalg.norm(J.T @ np.asarray(u, dtype=np.float64).reshape(2)))


def bar_observation(H, bar, sigma_px, sigma_tape_mm) -> dict:
    """One taped length read through ``H``, with the noise it actually carries.

    Three independent errors land in the residual: the tape itself, and a click
    at each end pushed through the local ground scale. A bar's whole job here is
    to outvote a survey, so its weight is not a detail — and the formula is
    checkable: it predicts a 3 m bar at 15 m determines scale to +-0.00213, and
    a Monte Carlo of the same geometry measured +-0.00214.

    Note what does *not* appear: where the bar is. Neither end needs a world
    coordinate, which is what makes these cheap enough to lay several of.
    """
    a = np.asarray(bar["a_px"], dtype=np.float64).reshape(2)
    b = np.asarray(bar["b_px"], dtype=np.float64).reshape(2)
    d = float(bar["length_mm"])
    if d <= 0:
        raise SystemExit(f"a taped length must be positive, got {d} mm")
    P = apply_h(H, np.array([a, b]))
    if not np.isfinite(P).all():
        raise SystemExit("a bar end maps to infinity through the ground plane — it is on "
                         "the horizon, where the mapping diverges")
    L = float(np.linalg.norm(P[1] - P[0]))
    if L < 1.0:
        raise SystemExit("a bar's two ends map to the same ground point; click them apart")
    u = (P[1] - P[0]) / L
    s = (ground_scale_along(H, a, u), ground_scale_along(H, b, u))
    sigma = math.hypot(sigma_tape_mm, sigma_px * math.hypot(*s))
    return {
        "taped_mm": d, "measured_mm": L, "error_mm": L - d, "sigma_mm": sigma,
        "z": (L - d) / sigma, "scale": d / L, "sigma_scale": sigma / d,
        "mm_per_px": [float(v) for v in s],
        "a_px": [float(a[0]), float(a[1])], "b_px": [float(b[0]), float(b[1])],
    }


def common_scale(obs) -> dict:
    """Do the bars agree on one scale factor? Inverse-variance mean, and its chi-square.

    A survey whose typed coordinates all came off the same stretched tape is
    wrong by exactly one number, and scaling the translation fixes it exactly —
    ``t -> a*t`` multiplies every ground coordinate by ``a`` and touches nothing
    else. A survey wrong in any other way makes the bars disagree with each
    other, and the chi-square is what tells the two cases apart. That is the
    whole reason to ask for more than one bar: a single one cannot tell you
    which of them you are looking at.
    """
    a = np.array([o["scale"] for o in obs], dtype=np.float64)
    w = 1.0 / np.array([o["sigma_scale"] for o in obs], dtype=np.float64) ** 2
    mean = float((w * a).sum() / w.sum())
    chi2 = float((w * (a - mean) ** 2).sum())
    dof = len(obs) - 1
    return {"scale": mean, "sigma": float(1.0 / math.sqrt(w.sum())), "chi2": chi2,
            "dof": dof, "consistent": dof < 1 or chi2 <= chi2_95(dof)}


def levmar(residual, x0, max_iter=60, tol=1e-10):
    """Levenberg-Marquardt with a central-difference Jacobian.

    Hand-rolled because this file is numpy + opencv + the standard library and
    stays that way. The problem is fourteen parameters against twenty-odd
    residuals, so a numeric Jacobian costs nothing worth optimising away.
    """
    x = np.array(x0, dtype=np.float64)
    r = residual(x)
    cost = float(r @ r)
    lam = 1e-3
    for _ in range(max_iter):
        J = np.empty((len(r), len(x)))
        for k in range(len(x)):
            h = 1e-6 * max(1.0, abs(x[k]))
            e = np.zeros(len(x))
            e[k] = h
            J[:, k] = (residual(x + e) - residual(x - e)) / (2 * h)
        A, g = J.T @ J, J.T @ r
        for _ in range(12):
            try:
                step = np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-12), -g)
            except np.linalg.LinAlgError:
                lam *= 10.0
                continue
            rn = residual(x + step)
            cn = float(rn @ rn)
            if cn < cost:
                x, r, cost, lam = x + step, rn, cn, max(lam * 0.3, 1e-9)
                break
            lam *= 10.0
        else:
            break
        if float(np.linalg.norm(step)) < tol * (1.0 + float(np.linalg.norm(x))):
            break
    return x, cost


def refine_free_network(K, R, t, gcp_px, gcp_world, bars, sigmas,
                        sigma_px=DEFAULT_SIGMA_PX, sigma_world_mm=DEFAULT_SIGMA_WORLD_MM):
    """Re-solve the pose *and* the ground control coordinates against the tapes.

    Why this cannot be a pose-only refinement, which is the obvious design and
    is worthless: hold the typed coordinates as truth and their eight
    reprojection residuals pin the pose completely, so one length observation
    has nowhere to push and the fit comes back unchanged. Measured on synthetic
    station geometry, a pose-only LM carrying a bar was identical to plain
    solvePnP in every error mode tried — 50 mm rms against 50, 100 against 100.
    The typed coordinates have to be allowed to be wrong, because with four
    marks and a calibrated camera they are the only thing that can be.

    So ``sigma_world_mm`` *is* the adjustment: it is how far the operator's
    tape survey may be overruled by the bars. Set it to nothing and this
    reduces to the no-op above; set it to infinity and the datum floats away,
    since lengths are blind to rotation, translation and reflection and can
    never determine more than five of the plane's eight degrees of freedom.

    The bar weights are frozen at the seed pose on purpose. Recomputed inside
    the cost they become a parameter the optimiser can game: a plane tipped
    until the local ground scale blows up carries a large sigma and therefore a
    small residual, which is a cheaper way to satisfy a tape than fitting it.
    """
    gcp_px = np.asarray(gcp_px, dtype=np.float64).reshape(-1, 2)
    typed = np.asarray(gcp_world, dtype=np.float64).reshape(-1, 2)
    n = len(typed)
    if len(gcp_px) != n:
        raise SystemExit("every ground control point needs a world coordinate")
    rvec0, _ = cv2.Rodrigues(np.asarray(R, dtype=np.float64))
    ends = np.array([[b["a_px"], b["b_px"]] for b in bars], dtype=np.float64).reshape(-1, 2)
    taped = np.array([float(b["length_mm"]) for b in bars], dtype=np.float64)
    sig_L = np.asarray(sigmas, dtype=np.float64)

    def unpack(x):
        Rx, _ = cv2.Rodrigues(x[:3].reshape(3, 1))
        return Rx, x[3:6], x[6:].reshape(n, 2)

    def residual(x):
        Rx, tx, W = unpack(x)
        Hw2i = image_from_world(K, Rx, tx)
        parts = [(apply_h(Hw2i, W) - gcp_px).ravel() / sigma_px,
                 (W - typed).ravel() / sigma_world_mm]
        if len(taped):
            P = apply_h(np.linalg.inv(Hw2i), ends).reshape(-1, 2, 2)
            parts.append((np.linalg.norm(P[:, 1] - P[:, 0], axis=1) - taped) / sig_L)
        return np.concatenate(parts)

    x0 = np.concatenate([rvec0.ravel(), np.asarray(t, dtype=np.float64).ravel(), typed.ravel()])
    x, _ = levmar(residual, x0)
    return unpack(x)


def probe_shift(K, R_old, t_old, H_new, anchors_mm, width, height, n=13, margin=0.5):
    """How far this adjustment moves the ground where the survey actually is.

    The cost function going down is not evidence the map got better. This is
    the number an operator can hold a tape against, and it is what ``--adjust``
    prints before it writes anything.

    Sweeping the whole frame asks the wrong question: near the horizon the
    ground mapping diverges, so *any* pose change moves those pixels by metres
    and the maximum ends up describing the sky. The grid here spans the marks
    and tapes that were actually surveyed, grown by ``margin``, and drops
    anything behind the camera or off the sensor.
    """
    anchors = np.asarray(anchors_mm, dtype=np.float64).reshape(-1, 2)
    lo, hi = anchors.min(axis=0), anchors.max(axis=0)
    pad = margin * np.maximum(hi - lo, 1.0)
    probes = np.stack(np.meshgrid(np.linspace(lo[0] - pad[0], hi[0] + pad[0], n),
                                  np.linspace(lo[1] - pad[1], hi[1] + pad[1], n)), -1).reshape(-1, 2)
    # The third homogeneous coordinate of K[r1 r2 t] is the camera-frame depth,
    # because K's last row is [0, 0, 1]. A point behind the camera still lands
    # on the sensor after the divide, and it is not ground anyone can measure.
    q = np.column_stack([probes, np.ones(len(probes))]) @ image_from_world(K, R_old, t_old).T
    px = q[:, :2] / np.where(np.abs(q[:, 2:3]) < 1e-12, 1e-12, q[:, 2:3])
    ok = (np.isfinite(px).all(axis=1) & (q[:, 2] > 0)
          & (px[:, 0] >= 0) & (px[:, 0] < width) & (px[:, 1] >= 0) & (px[:, 1] < height))
    if not ok.any():
        return np.zeros(0)
    return np.linalg.norm(apply_h(H_new, px[ok]) - probes[ok], axis=1)


def record_tape_check(cal: dict, bars, obs, fit) -> None:
    """Keep the tapes in the calibration so ``--reuse`` can re-run them.

    Written whether or not anything was corrected: clicking a tape twice is the
    one part of this step that costs the operator a walk across the tarmac.
    Assigned rather than merged, so a run that corrects nothing cannot leave
    the verdict of an earlier one standing next to it.
    """
    cal["tape"] = {"bars": bars, "checked_only": True, "observations": obs,
                   "common_scale": fit}
    print("  the tapes are stored in the calibration for --reuse; nothing else about it "
          "changed.", flush=True)


def cmd_tape(a) -> None:
    """Check the survey's scale against tapes whose endpoints you never surveyed — and fix it.

    Lay a tape between two marks on the tarmac, read the millimetres, click both
    ends. Neither end needs a world coordinate: the length is the whole
    observation, which is what makes these cheap enough to lay several of, out
    where the four ground control points are not.

    What this can and cannot see is worth being blunt about. With correct GCP
    coordinates, four points against a six-parameter pose is already
    over-determined and `solvePnPRefineLM` has the click noise handled — 2 mm
    rms on synthetic station geometry, and *every* bar-based adjustment made it
    worse. A tape is not a fix for a noisy pose. It is the only fix for a wrong
    *survey*: coordinates typed off a stretched tape, a mis-paced offset, a
    transposed digit. Nothing in the image can see that, because the fit is
    perfectly self-consistent with the wrong answer, and the `gcp` step's
    residuals will be millimetres while the map is centimetres out.

    So the default prints and changes nothing. Three stages, and the gate
    between them is the point:

    * every bar's implied length, its error, and how many sigma that is;
    * if the bars agree on one scale factor, ``--adjust`` applies it in closed
      form as ``t -> a*t``, which multiplies every ground coordinate by ``a``
      and is exact;
    * if they disagree, the error has shape rather than scale, and the pose and
      the typed coordinates are re-solved together against the tapes.

    Measured on synthetic station geometry, rms error in ground distances over
    an 8x20 m working area::

        GCP coordinates       4-pt PnP   rescale   free net (1 / 2 / 3 bars)
        exact                     2 mm     14 mm     14 /  10 /   8 mm
        0.5% scale error         50 mm     14 mm     37 /  30 /  31 mm
        30 mm random error      100 mm     75 mm     59 /  46 /  26 mm

    Read the first row before reaching for ``--adjust``: when the survey is
    right, adjusting costs a factor of seven. And read the third: the free
    network needs three bars before it clearly earns its keep.

    Lay them **long**. A bar's noise divides by its length, so one 10 m tape is
    worth nine 3 m ones — length beats count, and it is free.

    A bar cannot tell "the pose is wrong" from "the ground is not flat here".
    On a sloped pad a tape laid across the fall reads long, and ``--adjust``
    will bend the pose to absorb a non-planarity no pose can represent. That is
    the argument for leaving the check as the default.
    """
    cal = load_json(a.calibration)
    K, _, R, t = calibration_planes(cal)
    H_field = homography_at_height(K, R, t, 0.0)
    gcps = cal["field"].get("gcps") or []
    if len(gcps) < 4:
        raise SystemExit("this calibration has no stored ground control points; re-run `gcp`")
    gcp_px = np.array([[g["pixel"]["x"], g["pixel"]["y"]] for g in gcps], dtype=np.float64)
    typed = np.array([[g["world"]["x_mm"], g["world"]["y_mm"]] for g in gcps], dtype=np.float64)

    if a.reuse:
        bars = (cal.get("tape") or {}).get("bars")
        if not bars:
            raise SystemExit("--reuse needs a `tape` block in the calibration; run without it once")
        print(f"  reusing {len(bars)} bar(s) already in the calibration")
    else:
        if a.image is None:
            raise SystemExit("--image is required unless you pass --reuse")
        und = undistort(load_image(a.image), cal["intrinsics"], str(a.image))
        bars = picker.pick_bars(
            und, "Taped lengths — type the millimetres, then click both ends",
            hint="neither end needs a world coordinate · lay them long, and spread them out",
            open_browser=not a.no_open)
        if not bars:
            raise SystemExit("cancelled — nothing measured")

    # ---- stage 0: what each tape says about the calibration it was laid on
    obs = [bar_observation(H_field, b, a.sigma_px, a.sigma_tape_mm) for b in bars]
    print(f"  clicks worth {a.sigma_px:.2f} px, tape worth {a.sigma_tape_mm:.1f} mm")
    print(f"  {'taped':>9} {'reads':>9} {'error':>8} {'sigma':>7} {'z':>7}   mm/px")
    for o in obs:
        print(f"  {o['taped_mm']:9.0f} {o['measured_mm']:9.1f} {o['error_mm']:+8.1f} "
              f"{o['sigma_mm']:7.1f} {o['z']:+7.1f}   {o['mm_per_px'][0]:.1f}/{o['mm_per_px'][1]:.1f}")
    worst = max(abs(o["z"]) for o in obs)
    fit = common_scale(obs)
    res_now = np.linalg.norm(apply_h(H_field, gcp_px) - typed, axis=1)
    print(f"  ground control marks still fit their typed coordinates to "
          f"rms {res_now.mean():.1f} mm, worst {res_now.max():.1f} mm")
    if res_now.max() > a.max_gcp_residual_mm:
        i = int(np.argmax(res_now))
        print(f"  NOTE: the mark typed at ({typed[i][0]:.0f}, {typed[i][1]:.0f}) mm sits "
              f"{res_now[i]:.0f} mm off where the fit puts it. That is a mistyped coordinate "
              "or a click on the wrong mark, and no tape can absorb it — re-run `gcp`.")
    print(f"  implied scale {fit['scale']:.5f} +- {fit['sigma']:.5f}  "
          f"({1000 * (fit['scale'] - 1):+.1f} mm per metre)")
    if fit["dof"] >= 1:
        print(f"  bars agree with each other: chi2 {fit['chi2']:.1f} on {fit['dof']} dof "
              f"(95% limit {chi2_95(fit['dof']):.1f}) — {'yes' if fit['consistent'] else 'NO'}")

    # ---- stage 1: the gate
    if worst < a.gate_z and res_now.max() > a.max_gcp_residual_mm:
        i = int(np.argmax(res_now))
        raise SystemExit(
            f"the tapes agree with this calibration (worst bar {worst:.1f} sigma) but the mark "
            f"typed at ({typed[i][0]:.0f}, {typed[i][1]:.0f}) mm sits {res_now[i]:.0f} mm off "
            f"it. Scale is not the fault here — a single mark is. That is a mistyped "
            "coordinate or a click on the wrong mark in the `gcp` step, and re-running `gcp` "
            "is the fix; no scale correction can absorb it.")
    if worst < a.gate_z:
        print(f"  worst bar is {worst:.1f} sigma out; the tapes agree with this calibration.")
        print("  Nothing to adjust — and adjusting anyway costs accuracy, because the "
              "correction would be fitting the bars' own click noise.", flush=True)
        if a.adjust:
            print("  (--adjust ignored. Lower --gate-z if you are sure the tapes are better "
                  "than the survey.)", flush=True)
        record_tape_check(cal, bars, obs, fit)
        save_json(a.out or a.calibration, cal)
        return

    print(f"  worst bar is {worst:.1f} sigma out — the survey and the tapes disagree.")
    if not a.adjust:
        print("  This is a check only. Re-run with --adjust to correct the calibration, but "
              "look for a mistyped ground control coordinate first: that is the usual cause "
              "and re-typing it is a better fix than bending the pose around it.", flush=True)
        record_tape_check(cal, bars, obs, fit)
        save_json(a.out or a.calibration, cal)
        return

    # ---- stage 2: correct it, one of two ways
    if len(bars) < 3:
        print(f"  NOTE: {len(bars)} bar(s). Measured on synthetic geometry, a correction from "
              "one or two tapes is a coin toss — it helps as often as it hurts, because "
              "there is not enough to tell a real error from the bars' own click noise. "
              "Three is where it starts winning consistently.")
    if fit["consistent"]:
        method = "rescale"
        print(f"  the bars agree on one scale, so this is a scale error: t -> {fit['scale']:.5f} * t")
        print("  the typed coordinates are scaled to match. They were measured with the tape "
              "this correction says was wrong, so leaving them is what would be inconsistent — "
              "and scaling is about the world origin, so a mark used as the origin does not move.")
        R_new = R
        t_new = fit["scale"] * np.asarray(t, dtype=np.float64)
        world_new = fit["scale"] * typed
    else:
        method = "free-network"
        print(f"  the bars disagree with each other, so the error has shape and not just "
              f"scale: re-solving the pose and all {len(gcps)} ground control coordinates "
              f"together (--sigma-world-mm {a.sigma_world_mm:.0f})")
        R_new, t_new, world_new = refine_free_network(
            K, R, t, gcp_px, typed, bars, [o["sigma_mm"] for o in obs],
            sigma_px=a.sigma_px, sigma_world_mm=a.sigma_world_mm)

    H_new = homography_at_height(K, R_new, t_new, 0.0)
    obs_new = [bar_observation(H_new, b, a.sigma_px, a.sigma_tape_mm) for b in bars]
    res_px = np.linalg.norm(apply_h(image_from_world(K, R_new, t_new), world_new) - gcp_px, axis=1)
    res_mm = np.linalg.norm(apply_h(H_new, gcp_px) - world_new, axis=1)
    moved = np.linalg.norm(world_new - typed, axis=1)
    centre, up = camera_from_pose(R_new, t_new)
    tilt = float(np.rad2deg(np.arccos(np.clip(-R_new[2, 2] * up, -1.0, 1.0))))

    print(f"  bar error   rms {np.sqrt(np.mean([o['error_mm'] ** 2 for o in obs])):7.1f} mm "
          f"->{np.sqrt(np.mean([o['error_mm'] ** 2 for o in obs_new])):7.1f} mm")
    print(f"  reprojection    {cal['field']['pose']['reproj_rms_px']:7.2f} px "
          f"->{res_px.mean():7.2f} px")
    print(f"  camera      at ({centre[0]:.0f}, {centre[1]:.0f}, {centre[2] * up:.0f}) mm, "
          f"tilt {tilt:.1f}°  (was {cal['field']['pose']['center_mm'][2]:.0f} mm up, "
          f"{cal['field']['pose']['tilt_deg']:.1f}°)")
    if moved.max() > 1e-6:
        print(f"  ground control coordinates moved by up to {moved.max():.1f} mm:")
        for g, w, d in zip(gcps, world_new, moved, strict=True):
            print(f"    ({g['world']['x_mm']:8.1f}, {g['world']['y_mm']:8.1f}) -> "
                  f"({w[0]:8.1f}, {w[1]:8.1f}) mm   {d:6.1f} mm")

    anchors = np.vstack([typed, apply_h(H_field, np.array([b["a_px"] for b in bars])),
                         apply_h(H_field, np.array([b["b_px"] for b in bars]))])
    shift = probe_shift(K, R, t, H_new, anchors, cal["image_width"], cal["image_height"])
    if len(shift):
        print(f"  the ground over the surveyed area moves by {np.median(shift):.0f} mm typical, "
              f"{shift.max():.0f} mm worst")

    # Three refusals, and none of them is "the correction was large". A survey
    # wrong by 30 mm at the marks is legitimately hundreds of millimetres wrong
    # out where nothing was surveyed, so capping the movement rejects exactly the
    # corrections worth making. A blunder looks like something else entirely.
    #
    # First: a rescale is the one correction nothing internal can falsify.
    # Scaling t and the typed coordinates together is an exact re-labelling of
    # the world, so the reprojection residuals come back untouched however wrong
    # the factor is — and a single tape refits itself perfectly by construction,
    # one observation fitting one free parameter. The only check left is
    # physical: a tape stretches or gets misread by a fraction of a percent, and
    # a digit transposed in the length typed into the picker is what 6% looks
    # like.
    if method == "rescale" and abs(fit["scale"] - 1.0) > a.max_scale_error:
        raise SystemExit(
            f"the tapes imply the survey's scale is out by {100 * (fit['scale'] - 1):+.2f}%, "
            f"past the {100 * a.max_scale_error:.1f}% a tape can plausibly be wrong by. A "
            "stretched or misread tape is tenths of a percent; this is the size of a "
            "transposed digit in a length typed into the picker, or a length read in the "
            "wrong units. Check the table above. Raise --max-scale-error only if you have "
            "confirmed the figures with a second tape.")
    # Second: a tape that still does not fit once the fit has had every chance.
    still_out = max(abs(o["z"]) for o in obs_new)
    if still_out > a.max_bar_z:
        raise SystemExit(
            f"after adjusting, a bar is still {still_out:.1f} sigma out (limit "
            f"{a.max_bar_z:.1f}). No pose explains these tapes together, so at least one of "
            "them is a mistyped length or a click on the wrong mark. Check the table above "
            "before letting this through with --max-bar-z.")
    # Third: a mark dragged further than a tape survey could plausibly be wrong.
    # Only for the free network — a rescale moves every mark on purpose, and
    # that *is* the correction.
    if method == "free-network" and moved.max() > a.max_gcp_move_sigma * a.sigma_world_mm:
        i = int(np.argmax(moved))
        g = gcps[i]["world"]
        raise SystemExit(
            f"the fit had to move the mark typed at ({g['x_mm']:.0f}, {g['y_mm']:.0f}) mm by "
            f"{moved[i]:.0f} mm, which is {moved[i] / a.sigma_world_mm:.1f}x the "
            f"{a.sigma_world_mm:.0f} mm a typed coordinate is allowed to be wrong. That is a "
            "mistyped coordinate, not survey noise — re-typing it is a better fix than "
            "bending the pose around it. Raise --sigma-world-mm if the survey really is "
            "that loose.")
    if shift.size and shift.max() > a.max_shift_mm:
        print(f"  NOTE: this moves the ground by up to {shift.max():.0f} mm, more than the "
              f"{a.max_shift_mm:.0f} mm worth flagging. The tapes fit and the marks did not "
              "move far, so nothing here is inconsistent — but run `measure` against a tape "
              "before trusting the result.")

    # ---- write it back, including everything that was derived from the old pose
    cal["field"]["homography"] = H_new.tolist()
    cal["field"]["rms_error_mm"] = float(res_mm.mean())
    cal["field"]["max_error_mm"] = float(res_mm.max())
    cal["field"]["pose"] = {
        "rotation": R_new.tolist(),
        "translation_mm": np.asarray(t_new, dtype=np.float64).ravel().tolist(),
        "center_mm": centre.tolist(),
        "tilt_deg": tilt,
        "reproj_rms_px": float(res_px.mean()),
        "solver": f"IPPE+LM, then tape {method}",
    }
    for g, w, d, r in zip(gcps, world_new, moved, res_mm, strict=True):
        if d > 1e-6:
            g.setdefault("world_typed", dict(g["world"]))
            g["world"] = {"x_mm": float(w[0]), "y_mm": float(w[1])}
        g["residual_mm"] = float(r)
    cal["tape"] = {"bars": bars, "checked_only": False, "method": method,
                   "sigma_px": a.sigma_px, "sigma_tape_mm": a.sigma_tape_mm,
                   "sigma_world_mm": a.sigma_world_mm,
                   "observations": obs, "observations_after": obs_new,
                   "common_scale": fit,
                   "ground_shift_mm": (None if not len(shift) else float(shift.max()))}

    # The car plane was fitted as a homology *of the old field plane*, so it is
    # now stale — but the pole tops that produced it were stored, so it can be
    # re-fitted rather than thrown away. Silently leaving it is the one outcome
    # that must not happen: both pipelines prefer it, and nothing downstream can
    # notice that it disagrees with the ground it sits above.
    car = cal.get("car")
    if car and car.get("targets"):
        refit = car_plane_from_field(
            H_new,
            np.array([[q["pixel"]["x"], q["pixel"]["y"]] for q in car["targets"]]),
            np.array([[q["world"]["x_mm"], q["world"]["y_mm"]] for q in car["targets"]]),
            float(car["height_mm"]))
        car.update({"homography": refit["homography"].tolist(), "scale": refit["scale"],
                    "camera_ground_mm": refit["camera_ground_mm"],
                    "camera_height_mm": refit["camera_height_mm"],
                    "pose_camera_height_mm": float(centre[2] * up),
                    "rms_error_mm": refit["rms_mm"], "max_error_mm": refit["max_mm"]})
        for q, r in zip(car["targets"], refit["residuals_mm"], strict=True):
            q["residual_mm"] = r
        print(f"  re-fitted the surveyed car plane at {car['height_mm']:.0f} mm from its stored "
              f"targets: rms {refit['rms_mm']:.1f} mm, camera {refit['camera_height_mm']:.0f} mm up")
    elif car:
        cal.pop("car")
        print("  DROPPED the surveyed car plane: it was a homology of the old ground plane and "
              "its targets were not stored, so it cannot be re-fitted. Re-run `carplane`.")

    save_json(a.out or a.calibration, cal)
    print("  written. car.json is now stale too — its body_polygon_mm and wheels_mm came from "
          "clicks through the old plane. Re-run `outline`.", flush=True)



# --------------------------------------------------------------------------
# 4. sticker template
# --------------------------------------------------------------------------


def focus_on(H_plane, centre_mm, radius_mm, mode):
    """Where to open the picker, in image pixels, for a subject of a known size.

    Only meaningful on the frame: a bird's-eye raster is already cut to the
    subject, so it opens fitted and that is the right view. On the frame the
    subject is a fifth of a 3840 px image, and a pass that starts fitted starts
    with the operator hunting for the car.

    The radius is projected as a ring rather than a pair of points because the
    plane is oblique — the same millimetres subtend different pixels in different
    directions, and the ring's widest span is the one that has to fit.
    """
    if mode != "frame":
        return None
    ring = np.array([[centre_mm[0] + radius_mm * math.cos(a),
                      centre_mm[1] + radius_mm * math.sin(a)]
                     for a in np.linspace(0, 2 * math.pi, 16, endpoint=False)])
    px = apply_h(np.linalg.inv(np.asarray(H_plane, dtype=np.float64)),
                 np.vstack([ring, np.asarray(centre_mm).reshape(1, 2)]))
    if not np.isfinite(px).all():
        return None
    span = float(np.ptp(px[:-1], axis=0).max())
    return float(px[-1][0]), float(px[-1][1]), span


def plane_view(und, H_plane, mode, centre_mm=None, mm_per_px=None, span_mm=6000.0,
               min_span_mm=1000.0, y_up=True):
    """What to click on for one plane, and the maps between it and the world.

    Two ways to put a plane in front of an operator, and the choice is not about
    taste. A bird's-eye raster resamples the plane flat, which is the only way to
    *see* it as it is — a marker the same size and shape wherever it stands, a
    box that is square when the car is square. On a plane close to the camera,
    though, it is thousands of pixels a side, and every one of them is resampled
    so that a handful of clicks can be turned back into millimetres.

    The frame is already there at the camera's own resolution. A click on it maps
    to the world through exactly the same homography, and back again for drawing.
    What it costs is obliquity: the operator judges a corner on a plane they are
    looking across rather than down at.

    Returns ``(image, to_world, to_image, note)``.
    """
    if mode == "frame":
        Hinv = np.linalg.inv(np.asarray(H_plane, dtype=np.float64))
        return (und, lambda q: apply_h(H_plane, q), lambda w: apply_h(Hinv, w), None)
    span, mm, note = fit_raster(span_mm, mm_per_px, min_span_mm)
    raster, w2b, b2w = bev_around(und, H_plane, centre_mm, span, mm, y_up=y_up)
    return (raster, lambda q: apply_h(b2w, q), lambda w: apply_h(w2b, w), note)


def metric_crop_from_quad(und, H_plane, quad_px, mm_per_px, y_up=True, pad_mm=0.0):
    """A metric template cut straight from the frame, without a raster in between.

    The four corners are clicked on the undistorted frame, where the marker sits
    at the camera's own resolution and zooming costs nothing, and are mapped to
    the plane through the same homography every other measurement uses. Their
    world-axis bounding box is then warped out of the frame directly.

    The result is the template the bird's-eye route produces — world-aligned
    axes, constant millimetres per pixel — because it is the same warp; it is
    just evaluated over the marker instead of over ten metres of tarmac either
    side of it. On a plane close to the camera that is the difference between a
    few hundred pixels a side and several thousand.

    World-axis, specifically, and not the quad's own: the box is squared to the
    world, not to the marker. A template rotated upright would carry the marker's
    survey-time bearing inside it, and every offset the `outline` step stores is
    measured against the world's axes, so the two would disagree by that bearing
    with nothing anywhere to show it.
    """
    world = apply_h(H_plane, np.asarray(quad_px, dtype=np.float64).reshape(-1, 2))
    if not np.isfinite(world).all():
        raise SystemExit("one of those corners maps to infinity on this plane — it is on "
                         "the horizon, or the plane height is wrong")
    lo, hi = world.min(axis=0) - pad_mm, world.max(axis=0) + pad_mm
    n = np.maximum(np.ceil((hi - lo) / mm_per_px), 4).astype(int)
    x0, y0 = float(lo[0]), float(lo[1])
    y_top = y0 + n[1] * mm_per_px
    w2b = (np.array([[1 / mm_per_px, 0, -x0 / mm_per_px],
                     [0, -1 / mm_per_px, y_top / mm_per_px],
                     [0, 0, 1.0]])
           if y_up else
           np.array([[1 / mm_per_px, 0, -x0 / mm_per_px],
                     [0, 1 / mm_per_px, -y0 / mm_per_px],
                     [0, 0, 1.0]]))
    crop = cv2.warpPerspective(und, w2b @ np.asarray(H_plane, dtype=np.float64),
                               (int(n[0]), int(n[1])), flags=cv2.INTER_CUBIC)
    return crop, world


def box_from_picker(image, title, mm_per_px, hint, by_clicks, open_browser):
    """A crop rectangle, either dragged in one gesture or clicked corner by corner.

    Dragging is one continuous mouse-down, so the view cannot be zoomed part way
    through it — which is fine on a small raster and awkward on a large one,
    where the marker is a few dozen pixels in several thousand. Clicking the
    corners lets each one be placed at whatever zoom suits it, and the crop is
    their bounding box.

    A marker sitting at an angle puts background in that box's corners, and that
    is not worth avoiding: the matcher masks the template to its inscribed
    circle before correlating anything, so the corners are discarded either way.
    Cutting the quad out and rotating it upright would remove them, but it would
    also make the template's own axes the marker's rather than the world's — and
    every offset the `outline` step stores is measured against the world's.
    """
    if not by_clicks:
        return picker.pick_box(image, title, mm_per_px=mm_per_px, hint=hint,
                               open_browser=open_browser)
    got = picker.pick_points(image, title + " — click its corners", n=4,
                             hint=hint + "; zoom between clicks, the crop is their extent",
                             open_browser=open_browser)
    if not got:
        return None
    q = np.array([g["px"] for g in got], dtype=np.float64)
    lo, hi = q.min(axis=0), q.max(axis=0)
    return float(lo[0]), float(lo[1]), float(hi[0] - lo[0]), float(hi[1] - lo[1])


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
    _K, _m, _R, _t = calibration_planes(cal)
    y_up = camera_from_pose(_R, _t)[1] > 0
    print(f"  marker plane: {plane_note}")
    und = undistort(load_image(a.image), cal["intrinsics"], str(a.image))

    if a.space == "frame":
        got = picker.pick_points(
            und, f"Click the marker's 4 corners on the frame (plane {a.height_mm:.0f} mm)",
            n=4, hint="zoom right in; no bird's-eye raster is built for this",
            open_browser=not a.no_open)
        if not got:
            raise SystemExit("cancelled")
        quad = np.array([g["px"] for g in got], dtype=np.float64)
        mmpp = a.mm_per_px or round(plane_mm_per_px(H_car, quad.mean(axis=0)), 2)
        crop, world = metric_crop_from_quad(und, H_car, quad, mmpp, y_up=y_up,
                                            pad_mm=a.pad_mm)
        centre_mm = world.mean(axis=0)
        print(f"  corners span {world.max(axis=0)[0] - world.min(axis=0)[0]:.0f} x "
              f"{world.max(axis=0)[1] - world.min(axis=0)[1]:.0f} mm on the plane, "
              f"around ({centre_mm[0]:.0f}, {centre_mm[1]:.0f}) mm")
        print(f"  warped straight out of the frame at {mmpp:g} mm/px — no raster built")
    elif a.space == "raw":
        mmpp = plane_mm_per_px(H_car, [und.shape[1] / 2, und.shape[0] / 2])
        box = box_from_picker(und, "Box the marker in the camera frame", mmpp,
                              "raw crop — for inspection, not for a pipeline",
                              a.clicks, not a.no_open)
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
        span, mmpp, note = fit_raster(a.span_mm, mmpp, min_span_mm=1000.0)
        if note:
            print(f"  {note}")
        raster, _, _ = bev_around(und, H_car, centre_mm, span, mmpp, y_up=y_up)
        print(f"  bird's-eye {raster.shape[1]}x{raster.shape[0]} px @ {mmpp:g} mm/px "
              f"around ({centre_mm[0]:.0f}, {centre_mm[1]:.0f}) mm")
        box = box_from_picker(raster, "Box the marker on the bird's-eye raster", mmpp,
                              "check the mm readout against the printed marker",
                              a.clicks, not a.no_open)
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
    _K, _m, _R, _t = calibration_planes(cal)
    y_up = camera_from_pose(_R, _t)[1] > 0
    print(f"  marker plane: {plane_note}")
    und = undistort(load_image(a.image), cal["intrinsics"], str(a.image))

    seed = picker.pick_points(und, "Click the marker on the roof", n=1,
                              hint="just to centre the bird's-eye view",
                              open_browser=not a.no_open)
    if not seed:
        raise SystemExit("cancelled")
    approx = apply_h(H_car, np.array([seed[0]["px"]]))[0]
    mmpp = a.mm_per_px or round(plane_mm_per_px(H_car, seed[0]["px"]), 2)
    span, mmpp, note = fit_raster(a.span_mm, mmpp, min_span_mm=4000.0)
    if note:
        print(f"  {note}")
    raster, _, b2w = bev_around(und, H_car, approx, span, mmpp, y_up=y_up)

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
# 5b. outline — trace the car's box instead of taping it
# --------------------------------------------------------------------------


def build_box_from_front(front_two, marker_mm, length_mm, width_mm=None):
    """A rectangle from the two front corners, squared off and run backwards.

    An oblique camera can see a car's near end down to the road, but its own
    body hides the far end — so the rear corners are usually a guess, and a
    guessed corner is indistinguishable from a measured one once it is in the
    file. This builds them instead: the rear is exactly ``length_mm`` behind the
    front, square to it.

    Nothing about the rear touches an image, so no plane can be wrong back
    there. The front corners come off a raster rectified to their own height,
    which makes them true horizontal positions; from there this is arithmetic
    in world millimetres, and the rear lands where the tape says whether or not
    anything is visible.

    Which way is backwards comes from the marker, not from a click. The marker
    sits on the roof, inside the footprint, so of the two perpendiculars to the
    front edge the rearward one is whichever points towards it.
    """
    fl, fr = np.asarray(front_two, dtype=np.float64)[:2]
    edge = fr - fl
    span = float(np.linalg.norm(edge))
    if span < 500.0:
        raise SystemExit(f"the two front clicks are only {span:.0f} mm apart; that is not "
                         "the width of a car — click the two front corners")
    u = edge / span
    m = np.asarray(marker_mm, dtype=np.float64).reshape(2)
    if width_mm:
        # The marker is on the car's centreline, so the front edge's midpoint is
        # where that centreline crosses it — the foot of the perpendicular from
        # the marker onto the clicked line, not the midpoint of the two clicks.
        # Taking it from the marker means a corner clicked wide or short costs
        # the edge's *direction* only; it can no longer shift the whole car
        # sideways, which a clicked midpoint would do by half the error.
        mid = fl + u * float((m - fl) @ u)
        fl, fr = mid - u * width_mm / 2.0, mid + u * width_mm / 2.0

    back = np.array([-u[1], u[0]])     # square to the front edge; sign set below
    to_marker = m - (fl + fr) / 2.0
    if np.linalg.norm(to_marker) < 200.0:
        raise SystemExit("the marker sits on the front edge, so there is nothing to say "
                         "which way the car points. Trace all four corners instead.")
    back = back if float(back @ to_marker) > 0 else -back
    return np.array([fl, fr, fr + back * length_mm, fl + back * length_mm])


def norm_deg(a: float) -> float:
    """An angle folded into (-180, 180]."""
    a = (float(a) + 180.0) % 360.0 - 180.0
    return 180.0 if a == -180.0 else a


def centre_from_corners(corners):
    """The marker's centre, where its two diagonals cross.

    A single click at the middle of a marker is a guess at a point with nothing
    under it: no corner, no edge, nothing for the picker's sub-pixel snap to
    catch, and the eye with only the surrounding shape to go on. A corner is the
    opposite — it is the one place on a marker that is unambiguous, and it is
    what the snap was written for.

    Rectified on its own plane the marker is a parallelogram, so its diagonals
    bisect each other and cross at the centre. Click the four corners going round
    and the diagonals are the 1st to the 3rd, and the 2nd to the 4th.

    Two things the construction does with a corner's error. Error **along** that
    corner's own diagonal is free: it slides the click along the line it defines,
    and a line does not care. Error **across** the diagonal tilts it, and moves
    the centre by half. Against isotropic click noise the four together land
    about 0.7 of the error a single centre click would carry -- and that is
    before the far larger gain, which is that a corner can be clicked several
    times more accurately than a centre in the first place.
    """
    q = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    if len(q) != 4:
        raise SystemExit(f"the centre needs the marker's 4 corners, got {len(q)}")
    a0, a1, b0, b1 = q[0], q[2], q[1], q[3]
    da, db = a1 - a0, b1 - b0
    na, nb = float(np.linalg.norm(da)), float(np.linalg.norm(db))
    if na < 1.0 or nb < 1.0:
        raise SystemExit("two opposite corners are on top of each other; click the "
                         "marker's four corners, going round it")
    cross = float(da[0] * db[1] - da[1] * db[0])
    angle = math.degrees(math.asin(min(1.0, abs(cross) / (na * nb))))
    if angle < 20.0:
        raise SystemExit(
            f"those diagonals cross at only {angle:.0f} degrees, which places the centre "
            "poorly. On any marker that is not a sliver they are well apart — check the "
            "clicks went round the marker rather than back and forth across it."
        )
    t = float(((b0[0] - a0[0]) * db[1] - (b0[1] - a0[1]) * db[0]) / cross)
    return a0 + t * da, angle


def detector_centre(und, H_plane, near_mm, template, template_mm_per_px, y_up=True,
                    span_mm=2500.0, min_score=0.30):
    """Where the matcher puts this template on the survey frame, in world millimetres.

    The runtime anchor is not the marker's centre — it is wherever the matcher
    settles the template, and the two differ by however far the template was cut
    off-centre. That difference is a constant of the template, so it is worth
    measuring once, here, on the same frame the survey was clicked on, rather
    than discovering it later as a box that sits beside the car.

    Imported locally: the survey tool does not otherwise depend on the runtime,
    and this one optional check should not make it.
    """
    import pipeline1_bev as runtime

    mm = float(template_mm_per_px)
    raster, _w2b, b2w = bev_around(und, H_plane, near_mm, span_mm, mm, y_up=y_up)
    det = runtime.HybridDetector(template, min_score=min_score)

    # No pyramid here, and the reason is worth keeping. The runtime searches a
    # whole raster every frame, so it starts coarse: the template is decimated
    # to `pyr` and swept, then refined where that lands. At a 112x106 px marker
    # `pyr` is 8, which searches with a 14x13 px patch masked to a 5 px radius —
    # and at that resolution a marker is a blob among blobs. Measured on B1's
    # own frame, the row of tyres stacked on the wall behind the car scored
    # 0.681 against the marker's own peak, so the fine stage refined around a
    # tyre 2.6 m away, fell under min_score, and this returned "no match" with
    # the marker sitting dead centre in the raster at 0.816.
    #
    # The runtime can afford that gamble: it pays it once, on the cold frame,
    # and every frame after has a prior that pins the search. Here there is no
    # second chance and nothing to be fast about — one frame, one small raster,
    # and an operator who has already clicked the marker. So the sweep runs at
    # full resolution, which found it at every span tried.
    det.pyr = 1
    got = det.detect(raster)
    if got is None:
        return None, None, None
    x, y, theta, score, _sigma, _method = got
    centre = apply_h(b2w, np.array([[x, y]]))[0]
    tip = apply_h(b2w, np.array([[x + math.cos(math.radians(theta)) * 8,
                                  y - math.sin(math.radians(theta)) * 8]]))[0]
    heading = math.degrees(math.atan2(*(tip - centre)[::-1]))
    return centre, float(score), norm_deg(heading)


def wheels_from_box(corners, front_to_wheel_mm, rear_to_wheel_mm, track_mm):
    """The four contact patches, placed off the box the operator traced.

    Wheels are the one part of a car an oblique camera almost never shows: the
    near side is behind the sill and the far side is under the body. So they are
    not clicked, they are measured with a tape against the two lines the box
    already has — from the front line back to the front axle, and from the rear
    line forward to the rear axle.

    Pinning each axle to its own line is why it is two numbers and not a
    wheelbase. A wheelbase measured off the front alone puts every millimetre of
    its error onto the rear wheels; this way each end carries only its own. The
    wheelbase falls out as ``length - front - rear`` and is printed, which makes
    it a check rather than an input: it is a number the operator knows, so a
    tape read from the wrong place shows up as a wheelbase that is not the car's.

    This also retires the guess in the `car` step, which split the non-wheelbase
    length evenly between the two overhangs. Real cars are not symmetric — a
    front-drive hatchback's front overhang runs 100-150 mm longer than its rear
    — so the even split put each axle 50-75 mm out of place.

    **Contact patches are at ground level**, not at the height the box was
    traced at. That costs nothing here: what is stored either way is a
    horizontal offset from the marker, and a rigid car standing level holds the
    same horizontal offsets whatever height each feature sits at. Measure both
    distances horizontally from the same lines that were clicked, though, not
    from the bumper's lowest point — on a raked bumper those are tens of
    millimetres apart.
    """
    fl, fr, rr, rl = np.asarray(corners, dtype=np.float64)
    front_mid, rear_mid = (fl + fr) / 2.0, (rl + rr) / 2.0
    fwd = front_mid - rear_mid
    fwd = fwd / np.linalg.norm(fwd)
    lat = fr - fl
    lat = lat / np.linalg.norm(lat)                   # +lat points to the car's right
    front_axle = front_mid - fwd * float(front_to_wheel_mm)
    rear_axle = rear_mid + fwd * float(rear_to_wheel_mm)
    half = float(track_mm) / 2.0
    return {"fl": front_axle - lat * half, "fr": front_axle + lat * half,
            "rl": rear_axle - lat * half, "rr": rear_axle + lat * half}


def build_box_from_front_and_side(front_two, side_two, marker_mm, length_mm, width_mm):
    """A rectangle from a front edge and a side line, each doing what it is good at.

    Two clicks on a bumper fix how far forward the car is, and they do it well:
    the bumper is a hard edge square across the car, and both clicks sit on it.
    What they are poor at is the car's *direction*, because they are only a
    car's width apart — and a direction error does not shorten the box, it
    swings the far corners sideways in proportion to their distance along it.
    That is why a survey can come back with good front-to-back numbers and bad
    left-to-right ones from the same four clicks.

    A line down the flank is four times the baseline, so it pins the direction
    four times better. It also pins something the front edge never could: where
    the side of the car actually is. Without it the box has to assume the marker
    sits on the centreline and lay the width symmetrically about it, and a
    marker stuck 60 mm off centre then throws every lateral measurement by 60 mm
    with nothing in the file to show it.

    So each measurement is used where it is strongest:

    * **direction** from the side line — the long baseline;
    * **how far forward** from the front clicks — the square, hard edge;
    * **which side, and how far across** from the side line — measured, not
      assumed;
    * length and width from the tape, as before.

    The front clicks' own separation stops mattering here, which is the point:
    it no longer feeds the direction at all.
    """
    p0, p1 = np.asarray(side_two, dtype=np.float64)[:2]
    fwd = p1 - p0
    span = float(np.linalg.norm(fwd))
    if span < 800.0:
        raise SystemExit(f"the two side clicks are only {span:.0f} mm apart; that is too "
                         "short a line to take a direction from — put them at opposite "
                         "ends of the car")
    fwd = fwd / span
    m = np.asarray(marker_mm, dtype=np.float64).reshape(2)
    front = np.asarray(front_two, dtype=np.float64)[:2]
    front_mid = front.mean(axis=0)

    # Forward is whichever way along the side line the front clicks lie.
    if float((front_mid - m) @ fwd) < 0:
        fwd = -fwd
    right = np.array([fwd[1], -fwd[0]])           # the car's right, given that forward

    # Where the front edge sits, along the car. Both front clicks should give the
    # same answer; their spread is reported by the caller as a check.
    s_front = float(front_mid @ fwd)

    # The clicked flank is one side of the box. Which one is settled by the
    # marker: the body lies between the flank and the roof.
    c_near = float(p0 @ right)
    c_far = c_near + (width_mm if float(m @ right) > c_near else -width_mm)
    c_left, c_right = min(c_near, c_far), max(c_near, c_far)

    def at(s, c):
        return fwd * s + right * c

    return np.array([at(s_front, c_left), at(s_front, c_right),
                     at(s_front - length_mm, c_right), at(s_front - length_mm, c_left)])


def cmd_outline(a) -> None:
    """The car's box, traced on the plane the box actually sits at.

    The `car` step takes five tape measurements and a nose click and builds a
    rectangle from them. Every one of those numbers transfers one-for-one into
    a bumper position, and a rectangle is not what a car is. This traces the
    box instead: four clicks on a metric bird's-eye raster, stored as
    millimetre offsets from the marker.

    **The plane is the whole point.** A bird's-eye raster rectified to height
    ``p`` shows where things *appear* on that plane, not where they are. A
    corner at height ``h`` traced on the wrong plane comes back displaced by
    roughly ``|h - p| * r / (H_camera - h)``, which at B7 is about a millimetre
    per millimetre of mismatch — some 950 mm of error for tracing bumpers on
    the marker's 1450 mm plane. That error is a constant in the car's frame, so
    it rotates with the car and tracks it perfectly; it never announces itself,
    it is simply wrong everywhere by the same amount. Rectifying to the
    corners' own height instead makes the click read their true horizontal
    position, which is the number a footprint wants.

    Nothing is lost by moving the plane. Both rasters resample the same
    undistorted pixels through different homographies, so the corners are
    exactly as visible either way — only the millimetres assigned to a click
    change.

    Two ways to give it the box. Click all four corners **in order, going round
    the car** — front-left, front-right, rear-right, rear-left — or, when the
    far end is hidden under the car's own body as it usually is on an oblique
    camera, pass ``--length-mm`` and click only the two front corners. The rear
    is then built square to the front edge at that distance, by
    :func:`build_box_from_front`, and never touches an image at all.

    ``--width-mm`` is worth passing in either case. In the two-click mode it
    replaces the clicked width with a span of exactly that, laid half either
    side of **the marker**, not either side of the clicked midpoint: the car is
    symmetric about its centreline and the marker sits on it, so the front
    edge's midpoint is where that centreline crosses it. A corner clicked wide
    or short then costs the edge's direction only — it can no longer drag the
    whole car sideways, which a clicked midpoint does by half the error. A
    corner slid 300 mm along the edge leaves the footprint exact.

    Either way the clicked width is compared against ``--width-mm``, and that
    comparison is the only test that can see a wrong ``--height-mm`` here,
    because a built box is square and the right length whatever plane it was
    traced on. A plane set too high returns everything too small, and the width
    is where it shows; the step reports the gap as the box error it implies and
    solves back for the height the clicks really mean.

    **The front edge and the flank need not sit at the same height.** A bumper
    corner is near 500 mm, a sill runs lower, a shoulder crease higher, and
    clicking one on the other's plane displaces it. ``--front-height-mm`` and
    ``--side-height-mm`` give each line its own raster; both default to
    ``--height-mm``, so a command that does not use them is unchanged.

    Nothing has to be reconciled afterwards, which is the part worth
    understanding. A raster rectified to height ``p`` reports where a ray
    crosses ``p``, so a feature *at* ``p`` reads its true horizontal position —
    and marker, front edge and flank therefore arrive as the same world
    millimetres on the same ground, whatever heights they were read at. They
    compose directly. Measured on a synthetic car with bumpers at 500 mm and a
    sill at 300 mm, per-plane clicking returns the box to **0.00 mm**; forcing
    both onto one plane costs 22 mm at 500, 347 mm at 300 and 589 mm at 160.

    A wrong ``--side-height-mm`` is the milder of the two mistakes, and provably
    so: a homology maps a line to a *parallel* line, so the direction the flank
    gives is exact however wrong its height is — 0.000000 deg over a 600 mm
    error in testing. All that moves is where the flank sits across the car,
    and the step prints how much per 10 mm for the geometry in front of it.

    With three levels a question becomes askable that two could not answer: how
    far the marker sits from the flank, against half of ``--width-mm``. What is
    left over is the marker's own offset from the centreline plus, when the
    planes differ, the body's taper between them — two causes in one number, so
    it is quoted and never corrected.

    Everything is shown for approval before anything is written, and each
    overlay is drawn on the plane that makes it mean something. The clicked
    lines go back on their own levels, where they still land on the clicks that
    made them. **The box is drawn on the lowest of the three** — its corners are
    horizontal positions carrying no height of their own, so putting them on an
    image is a choice, and the lowest level is the one that sets the outline on
    the car's base rather than up at bumper height, standing proud of it by
    exactly the parallax this step exists to remove. The gap between the box and
    the front line is that parallax, and it is worth seeing. Nothing written
    changes: this moves an overlay, not a millimetre.

    When those overlays end up on different planes the check moves onto the
    frame, since a flank drawn on the front plane's bird's-eye lands where it
    *appears* from there rather than where it was clicked. Cancelling re-runs
    the clicks; ``--no-preview`` skips it.

    The printed box size is the check that matters. A wrong ``--height-mm``
    leaves the trace perfectly square and tracking the car perfectly — the
    squareness test cannot see it — and shows up only as a box of the wrong
    size. Tracing 500 mm bumpers on the marker's 1450 mm plane at B7 returns a
    3345 x 1422 mm box for a 4000 x 1700 mm car, and a footprint 964 mm out.
    """
    cal = load_json(a.calibration)
    K, _, R, t = calibration_planes(cal)
    y_up = camera_from_pose(R, t)[1] > 0
    H_stick, plane_note = car_plane(cal, a.sticker_height_mm)
    # Three levels, and each click is read on the plane its own feature sits at.
    # A bird's-eye raster rectified to height p reports where a ray crosses p —
    # so a feature *at* p reads its true horizontal position and needs no
    # correction at all. That is why the levels reconcile with no arithmetic:
    # marker, front edge and flank come back as the same world millimetres,
    # measured on the same ground, and compose directly.
    h_front = a.height_mm if a.front_height_mm is None else a.front_height_mm
    h_side = a.height_mm if a.side_height_mm is None else a.side_height_mm
    # The flank pass only exists in the two-click mode, so a second plane is only
    # real when that pass will run. Otherwise --side-height-mm would quietly cost
    # a raster and push the preview onto the frame for a line nobody clicks.
    side_pass = bool(a.side and a.length_mm)
    split = side_pass and h_side != h_front

    # The box is *drawn* on the lowest of the three levels. Its corners are
    # horizontal positions and carry no height of their own, so putting them
    # back on an image is a choice, and the lowest plane is the one that puts
    # them nearest the tarmac the footprint is finally measured on. Drawing at
    # the front's height instead sits the outline up at bumper level, where it
    # stands proud of the car's base by exactly the parallax this step exists to
    # remove — pleasant to look at, and the wrong thing to check against.
    # Nothing written changes: this moves an overlay, not a millimetre.
    h_draw = min(a.sticker_height_mm, h_front, h_side if side_pass else h_front)
    H_out = homography_at_height(K, R, t, h_front)
    H_side = homography_at_height(K, R, t, h_side) if split else H_out
    H_draw = (H_out if h_draw == h_front else
              H_side if h_draw == h_side else homography_at_height(K, R, t, h_draw))
    # Layers on different planes cannot share a raster: each would land where it
    # *appears* from the other's viewpoint rather than where it belongs. The
    # frame is the one surface they all project onto honestly.
    on_frame = len({h_front, h_draw} | ({h_side} if side_pass else set())) > 1
    if side_pass and not a.width_mm:
        raise SystemExit("--side needs --width-mm: the flank fixes which side of the car "
                         "the box starts from, and the tape gives how wide it runs.")
    if a.side_height_mm is not None and not side_pass:
        print("  NOTE: --side-height-mm does nothing without --side and --length-mm; the "
              "flank is only clicked in the two-click mode. Ignoring it.", flush=True)
    und = undistort(load_image(a.image), cal["intrinsics"], str(a.image))
    print(f"  marker plane:  {plane_note}")
    print(f"  front plane:   z = {h_front:.0f} mm")
    if split:
        print(f"  side plane:    z = {h_side:.0f} mm")
    if h_draw != h_front:
        print(f"  box drawn on:  z = {h_draw:.0f} mm — the lowest of the three, so the "
              f"outline sits on the car's base and not up at bumper level")

    # 1. the marker centre, always on a bird's-eye of the marker's own plane.
    # Never on the frame, and never at the box's span: this is the one click the
    # whole file is measured from, it wants the plane shown flat, and the seed
    # has already found the marker — so a couple of metres around it is the
    # whole picture, at any plane height, for a few hundred pixels.
    seed = picker.pick_points(und, "Click the marker on the roof", n=1,
                              hint="just to centre the bird's-eye view",
                              open_browser=not a.no_open)
    if not seed:
        raise SystemExit("cancelled")
    approx = apply_h(H_stick, np.array([seed[0]["px"]]))[0]
    mm_s = a.mm_per_px or round(plane_mm_per_px(H_stick, seed[0]["px"]), 2)
    img_s, to_w_s, _to_i_s, note = plane_view(und, H_stick, "bev", approx, mm_s,
                                              a.marker_span_mm, 600.0, y_up)
    print(f"  marker view: bird's-eye {img_s.shape[1]}x{img_s.shape[0]} px @ {mm_s:g} mm/px"
          + (f" — {note}" if note else ""))
    if a.marker_corners:
        four = picker.pick_points(
            img_s, "Click the marker's 4 corners, going round", n=4,
            hint="snap is on and a corner is what it catches — let it settle each click",
            open_browser=not a.no_open)
        if not four:
            raise SystemExit("cancelled")
        quad = to_w_s(np.array([q["px"] for q in four]))
        centre_mm, cross_deg = centre_from_corners(quad)
        # For a parallelogram the four corners average to the same point, by a
        # different route. Quoting the gap turns four clicks into a measurement
        # with a residual instead of four clicks with a result.
        spread = float(np.linalg.norm(quad.mean(axis=0) - centre_mm))
        side = [float(np.linalg.norm(quad[i] - quad[(i + 1) % 4])) for i in range(4)]
        print(f"  marker centre from its 4 corners, diagonals crossing at {cross_deg:.0f} deg")
        print(f"  ({centre_mm[0]:.1f}, {centre_mm[1]:.1f}) mm — their mean agrees to "
              f"{spread:.1f} mm")
        print(f"  marker measures {(side[0] + side[2]) / 2:.0f} x {(side[1] + side[3]) / 2:.0f} mm "
              f"— check that against the printed one")
        if spread > a.max_centre_spread_mm:
            print(f"  NOTE: over {a.max_centre_spread_mm:.0f} mm apart. On a marker that "
                  f"rectifies to a parallelogram those two are the same point, so one "
                  f"corner is off — or the plane height is wrong.", flush=True)
    else:
        one = picker.pick_points(img_s, "Click the marker centre", n=1,
                                 hint="this is the origin every offset is measured from",
                                 open_browser=not a.no_open)
        if not one:
            raise SystemExit("cancelled")
        centre_mm = to_w_s(np.array([one[0]["px"]]))[0]

    # What the matcher will call the centre of this same marker. The runtime hangs
    # every box off that, not off the marker, so the gap between the two is a
    # constant worth measuring here rather than meeting later as a box beside the car.
    det_centre = det_offset = det_score = det_heading = None
    if a.template:
        tpl = cv2.imread(str(a.template), cv2.IMREAD_COLOR)
        if tpl is None:
            raise SystemExit(f"could not read template {a.template}")
        if not a.template_mm_per_px:
            raise SystemExit("--template needs --template-mm-per-px, the scale it was cut at")
        det_centre, det_score, det_heading = detector_centre(
            und, H_stick, centre_mm, tpl, a.template_mm_per_px, y_up=y_up)
        if det_centre is None:
            print("  NOTE: the template did not match on this frame, so no detector "
                  "offset was measured. The box will be built against the clicked "
                  "centre alone.", flush=True)
        else:
            # The click is ground truth for where the marker is, so a match that
            # lands further off than the template is itself wide has found some
            # other object — not a template cut off-centre, which cannot exceed
            # the template's own half-diagonal by definition. Storing one would
            # put a metre of "offset" in the car file for --detector-offset to
            # add to every frame of every run.
            hard_mm = 0.5 * math.hypot(tpl.shape[1] * a.template_mm_per_px,
                                       tpl.shape[0] * a.template_mm_per_px)
            stray = float(np.linalg.norm(centre_mm - det_centre))
            if stray > hard_mm:
                print(f"  NOTE: the matcher settled {stray:.0f} mm from the marker you "
                      f"clicked, which is further than the template is wide "
                      f"({hard_mm:.0f} mm half-diagonal). That is a lock onto something "
                      f"else, not an off-centre cut, so no detector offset is stored. "
                      f"Check that --template is this station's marker.", flush=True)
                det_centre = det_score = det_heading = None
        if a.template and det_centre is not None:
            det_offset = centre_mm - det_centre
            print(f"  detector puts it at ({det_centre[0]:.1f}, {det_centre[1]:.1f}) mm, "
                  f"score {det_score:.3f}")
            print(f"  offset true - detected = ({det_offset[0]:+.1f}, {det_offset[1]:+.1f}) "
                  f"mm, {float(np.linalg.norm(det_offset)):.1f} mm")
            if float(np.linalg.norm(det_offset)) > a.max_detector_offset_mm:
                print(f"  NOTE: over {a.max_detector_offset_mm:.0f} mm. That is the template "
                      f"cut off-centre, and every box built against the detector carries it. "
                      f"Re-cut the template, or carry the offset.", flush=True)
    cam, up = camera_from_pose(R, t)
    cam = np.array([cam[0], cam[1], cam[2] * up])
    r_mm = float(np.linalg.norm(centre_mm - cam[:2]))
    factor = r_mm / max(float(cam[2]) - h_front, 1e-6)

    # 2. the box, read on the plane the box sits at
    centre_px_out = apply_h(np.linalg.inv(H_out), np.array([centre_mm]))[0]
    if a.space == "frame":
        raster_o, to_w_o, to_i_o, note = plane_view(und, H_out, "frame")
    else:
        mm_o = a.mm_per_px or round(plane_mm_per_px(H_out, centre_px_out), 2)
        raster_o, to_w_o, to_i_o, note = plane_view(und, H_out, "bev", centre_mm, mm_o,
                                                    a.span_mm, 4000.0, y_up)
    if note:
        print(f"  box view: {note}")
    # Both box passes open on the car, sized to hold it whole with room to spare.
    focus = focus_on(H_out, centre_mm, 0.6 * (a.length_mm or 5000.0), a.space)
    if focus:
        print(f"  box view: opening on the car, {focus[2]:.0f} px across at native "
              f"resolution — wheel to zoom, r to fit the whole frame")

    # The flank gets a raster of its own when it rides at its own height.
    # Nothing about the pass changes except the millimetres a click is worth,
    # which is the entire point of asking for a second height.
    if split:
        centre_px_side = apply_h(np.linalg.inv(H_side), np.array([centre_mm]))[0]
        if a.space == "frame":
            raster_s2, to_w_s2, _to_i_s2, note_s = plane_view(und, H_side, "frame")
        else:
            mm_sd = a.mm_per_px or round(plane_mm_per_px(H_side, centre_px_side), 2)
            raster_s2, to_w_s2, _to_i_s2, note_s = plane_view(und, H_side, "bev", centre_mm,
                                                              mm_sd, a.span_mm, 4000.0, y_up)
        if note_s:
            print(f"  side view: {note_s}")
        focus_s = focus_on(H_side, centre_mm, 0.6 * (a.length_mm or 5000.0), a.space)
    else:
        raster_s2, to_w_s2, focus_s = raster_o, to_w_o, focus
    n_click = 2 if a.length_mm else 4
    while True:
        pts = picker.pick_points(
            raster_o,
            f"Car box at {h_front:.0f} mm — click "
            + ("the 2 FRONT corners: front-left, front-right" if n_click == 2 else
               "4 corners: front-left, front-right, rear-right, rear-left"),
            n=n_click,
            hint=("the rear is built from --length-mm, so it need not be visible"
                  if n_click == 2 else
                  "go round the car in that order; it is what sets which end is the front"),
            open_browser=not a.no_open, focus=focus)
        if not pts:
            raise SystemExit("cancelled — nothing saved")

        clicked = to_w_o(np.array([q["px"] for q in pts]))
        click_width = float(np.linalg.norm(clicked[0] - clicked[1]))

        side = None
        if n_click == 2 and a.side:
            spts = picker.pick_points(
                raster_s2, f"Now click 2 points along ONE side of the car, at {h_side:.0f} mm",
                n=2, hint="as far apart as you can get them — this sets the direction",
                open_browser=not a.no_open, focus=focus_s)
            if not spts:
                print("  no side line — falling back to the front clicks alone", flush=True)
            else:
                side = to_w_s2(np.array([q["px"] for q in spts]))

        if n_click == 2 and side is not None:
            corners = build_box_from_front_and_side(clicked, side, centre_mm,
                                                    a.length_mm, a.width_mm)
            fwd = side[1] - side[0]
            fwd = fwd / np.linalg.norm(fwd)
            print(f"  side line {np.linalg.norm(side[1] - side[0]):.0f} mm long; direction "
                  f"taken from it, front position from the bumper clicks")

            # Three things the two lines can say about each other, none of which
            # could be asked before: a front edge on its own has nothing to
            # disagree with.
            level = abs(float((clicked[0] - clicked[1]) @ fwd))
            print(f"  the two front clicks sit {level:.0f} mm apart along the car "
                  f"(level would be 0)")
            if level > a.max_level_mm:
                print(f"  NOTE: over {a.max_level_mm:.0f} mm out of level. One front click "
                      f"is further down the car than the other, or the side line does not "
                      f"run straight along it.", flush=True)
            edge = clicked[1] - clicked[0]
            edge = edge / np.linalg.norm(edge)
            skew = abs(90.0 - math.degrees(math.acos(min(1.0, abs(float(edge @ fwd))))))
            # Reported in millimetres of corner movement, not in degrees. A degree
            # means nothing without the length it is levered over, and this is the
            # second threshold in this step to learn that the hard way.
            arm = math.hypot(a.length_mm / 2, a.width_mm / 2)
            cost = arm * math.sin(math.radians(skew))
            print(f"  front edge is {skew:.1f} deg off square to the side line "
                  f"-> {cost:.0f} mm at a corner")
            if cost > a.max_square_mm:
                print(f"  NOTE: the two lines disagree about the car's direction by "
                      f"{skew:.1f} deg, which is {cost:.0f} mm at a corner. The side line "
                      f"is believed here because it is the longer baseline, so the box is "
                      f"still right — but a front click that far out of square usually "
                      f"means one of them is not on the bumper.", flush=True)

            # The question three levels can be asked and two could not: where the
            # flank sits across the car, against where the tape and the marker
            # say it should. What is left over is the marker's own offset from
            # the centreline, plus — when the planes differ — the body's taper
            # between them. Two causes in one number, so it is quoted and never
            # corrected; the box already takes its lateral position from the
            # flank, which is the measured one of the two.
            right_ = np.array([fwd[1], -fwd[0]])
            reach = abs(float((centre_mm - side[0]) @ right_))
            print(f"  marker sits {reach:.0f} mm from the flank across the car; half of "
                  f"--width-mm is {a.width_mm / 2:.0f} mm -> {reach - a.width_mm / 2:+.0f} mm"
                  + (" (marker off centre, and body taper between the two planes)" if split
                     else " (marker off the centreline)"))

            # A homology maps a line to a parallel line, so a wrong plane height
            # cannot bend the direction a flank gives — that much is exact
            # however wrong --side-height-mm is. What it moves is where the
            # flank sits across the car, in proportion to how far the flank
            # stands from under the camera.
            if split:
                lat = abs(float((side[0] - cam[:2]) @ right_))
                sens = lat / max(float(cam[2]) - h_side, 1e-6)
                print(f"  side-height sensitivity: 10 mm of error in --side-height-mm moves "
                      f"the flank {10 * sens:.0f} mm across the car (x{sens:.2f}). The "
                      f"direction it gives is unaffected.")
        elif n_click == 2:
            corners = build_box_from_front(clicked, centre_mm, a.length_mm, a.width_mm)
            print(f"  front edge clicked {click_width:.0f} mm wide; rear built "
                  f"{a.length_mm:.0f} mm back, square to it")
            if a.width_mm:
                # The clicked width measures the same quantity the tape gives, so
                # the two disagreeing is real information — and it is the only
                # check that can see a wrong --height-mm here. A width gap and a
                # box error are one mistake in two units: the plane rescales
                # everything by (H - plane) / (H - true), so
                #     gap / width == |true - plane| / (H - true)
                # and the box error is that times the distance from the camera,
                # which cancels to error = gap * r / width. The gap alone hides
                # the cost — 88 mm of width here is 300 mm of box.
                gap = abs(click_width - a.width_mm)
                implied = gap * r_mm / max(a.width_mm, 1e-6)
                print(f"  width check: clicked {click_width:.0f} mm vs --width-mm "
                      f"{a.width_mm:.0f} mm, differ by {gap:.0f} mm "
                      f"-> about {implied:.0f} mm of box error")
                if implied > a.max_box_error_mm:
                    # Solved the other way, the clicked width says what height
                    # would have produced it — which is the number to pass next.
                    h_implied = (float(cam[2]) - (float(cam[2]) - h_front)
                                 * a.width_mm / max(click_width, 1e-6))
                    print(f"  NOTE: that is more than {a.max_box_error_mm:.0f} mm of box "
                          f"error. Either the two clicks are not on the bumper corners, or "
                          f"--height-mm is wrong — these clicks imply the corners are "
                          f"really at about {h_implied:.0f} mm, not {h_front:.0f} mm.",
                          flush=True)
        else:
            corners = clicked
            # Four corners clicked round the car make a simple quadrilateral; four
            # clicked across a diagonal fold into a bow-tie. The corners still land
            # in the right places, so the footprint looks right and the squareness
            # test passes — but two of the polygon's edges now cut across the car,
            # and every clearance measured against them is wrong. Consistent
            # winding is what separates the two, and the reported length quietly
            # becomes the diagonal, so it is worth refusing rather than warning.
            e_ = np.roll(corners, -1, axis=0) - corners
            turn = e_[:, 0] * np.roll(e_, -1, axis=0)[:, 1] - e_[:, 1] * np.roll(e_, -1, axis=0)[:, 0]
            if not (np.all(turn > 0) or np.all(turn < 0)):
                print("  NOTE: those four clicks cross over themselves, so the box would fold "
                      "into a bow-tie. Go round the car — front-left, front-right, rear-right, "
                      "rear-left, or the mirror of it — never across a diagonal.", flush=True)
                continue

        if a.no_preview:
            break
        # Drawn from the numbers that are about to be written, never recomputed —
        # a preview that agreed with itself but not with the file would be worse
        # than none. Grey marks are the raw clicks, so an adjustment is visible.
        line_name = ("adjusted front line" if (n_click == 2 and a.width_mm) else "front edge")
        # Each overlay is drawn on the plane that makes it mean something: the
        # box down on the lowest level, and each clicked line back on its own,
        # where it still lands on the clicks that made it. Read together they
        # show both halves of the check — the lines against what was clicked,
        # the box against the car's base.
        if on_frame:
            review_img = und
            def _to(H, w):
                return apply_h(np.linalg.inv(H), w)
        else:
            review_img = raster_o
            def _to(_H, w):
                return to_i_o(w)
        f_pts = _to(H_out, corners[:2])
        b_pts = _to(H_draw, corners)
        s_pts = None if side is None else _to(H_side, side)
        m_pts = _to(H_out, clicked)
        if s_pts is not None:
            m_pts = np.vstack([m_pts, s_pts])
        layers = [
            {"name": f"{line_name} @ {h_front:.0f}", "points": f_pts.tolist(),
             "closed": False, "colour": "#58a6ff", "on": True},
            {"name": f"car box @ {h_draw:.0f}", "points": b_pts.tolist(),
             "closed": True, "colour": "#3fb950", "on": True},
        ]
        if s_pts is not None:
            layers.insert(1, {"name": f"side line @ {h_side:.0f}", "points": s_pts.tolist(),
                              "closed": False, "colour": "#d29922", "on": True})
        ok = picker.review_layers(
            review_img, "Check it, then save — or redo the clicks", layers,
            marks=m_pts.tolist(),
            hint=("grey dots are your raw clicks; the buttons toggle each overlay"
                  + (" — on the frame, because the overlays sit on different planes"
                     if on_frame else "")),
            open_browser=not a.no_open)
        if ok:
            break
        print("  redo — click again", flush=True)

    fl, fr, rr, rl = corners
    # The `sticker` step cuts its template axis-aligned out of a world-aligned
    # raster, so the template's frame IS the world frame at survey time, moved
    # to the marker. The offsets therefore need no rotation.
    body = corners - centre_mm

    fwd = (fl + fr) / 2.0 - (rl + rr) / 2.0
    if np.linalg.norm(fwd) < 200:
        raise SystemExit("the front and rear clicks are almost the same point, so the "
                         "heading is undefined — check the click order")
    yaw = float(np.degrees(math.atan2(fwd[1], fwd[0])))
    yaw = (yaw + 180.0) % 360.0 - 180.0

    length = float((np.linalg.norm(fl - rl) + np.linalg.norm(fr - rr)) / 2.0)
    width = float((np.linalg.norm(fl - fr) + np.linalg.norm(rl - rr)) / 2.0)
    front_mm = float(np.dot((fl + fr) / 2.0 - centre_mm, fwd / np.linalg.norm(fwd)))
    rear_mm = front_mm - length

    print(f"  box {length:.0f} x {width:.0f} mm, car points {yaw:+.2f} deg from the "
          f"template's +X axis")
    print(f"  marker sits {front_mm:.0f} mm behind the front, {abs(rear_mm):.0f} mm "
          f"ahead of the rear")
    if n_click == 4:
        print("  ^ check that box against the car's real length and width. A wrong "
              "--height-mm shows up here as a box the wrong size and NOWHERE else: it "
              "traces perfectly square and tracks the car perfectly, it is just wrong.")

    # A traced box should close on itself. Sides that disagree, or diagonals
    # that do, mean a mis-click — and a mis-click here is silent downstream.
    skew_side = float(abs(np.linalg.norm(fl - rl) - np.linalg.norm(fr - rr)))
    skew_diag = float(abs(np.linalg.norm(fl - rr) - np.linalg.norm(fr - rl)))
    if n_click == 4:
        print(f"  squareness: opposite sides differ by {skew_side:.0f} mm, "
              f"diagonals by {skew_diag:.0f} mm")
        if max(skew_side, skew_diag) > a.max_skew_mm:
            print(f"  NOTE: over {a.max_skew_mm:.0f} mm out of square. On a rectangular "
                  f"bumper line that is a mis-click, not a car.", flush=True)
    # A built box is square and the right length by construction, so neither
    # test can say anything about it. The width check above is the one that can.
    if not 2000.0 <= length <= 6500.0 or not 1200.0 <= width <= 2600.0:
        print("  NOTE: that is not a car-sized box. Check the clicks, and that the "
              "raster is showing the car.", flush=True)

    wheels = wheel_note = None
    if a.front_to_wheel_mm is not None and a.rear_to_wheel_mm is not None:
        if a.front_to_wheel_mm + a.rear_to_wheel_mm >= length:
            raise SystemExit(
                f"--front-to-wheel-mm {a.front_to_wheel_mm:.0f} plus --rear-to-wheel-mm "
                f"{a.rear_to_wheel_mm:.0f} is {a.front_to_wheel_mm + a.rear_to_wheel_mm:.0f} "
                f"mm, which does not fit inside the {length:.0f} mm box — the axles would "
                "cross over."
            )
        track = a.track_mm or width
        w_world = wheels_from_box(corners, a.front_to_wheel_mm, a.rear_to_wheel_mm, track)
        wheels = {k: (v - centre_mm) for k, v in w_world.items()}
        wheelbase = length - a.front_to_wheel_mm - a.rear_to_wheel_mm
        wheel_note = (f"{a.front_to_wheel_mm:.0f} mm behind the front, "
                      f"{a.rear_to_wheel_mm:.0f} mm ahead of the rear, track {track:.0f}")
        print(f"  wheels: {wheel_note}")
        print(f"          wheelbase works out at {wheelbase:.0f} mm — check that against "
              f"the car; it is derived, so a tape read from the wrong place shows up here")
        if not 1800.0 <= wheelbase <= 3600.0:
            print("  NOTE: that is not a car's wheelbase. One of the two distances is "
                  "measured from the wrong place.", flush=True)
        if not a.track_mm:
            print(f"          track defaulted to the body width ({width:.0f} mm). A real "
                  f"track is usually 150-250 mm narrower, which puts each wheel about "
                  f"75-125 mm too far out; pass --track-mm to fix it.", flush=True)
        if a.tyre_width_mm:
            print(f"          tyre {a.tyre_width_mm:.0f} mm wide — a wheel counts as on a "
                  f"line within {a.tyre_width_mm / 2:.0f} mm of it")

        # The contact patches are on the ground, so a ground-plane raster is
        # where they can be checked. Whatever the camera happens to show gets
        # clicked; each click is matched to its nearest constructed wheel, and
        # the residual is the only independent word on whether the three tape
        # numbers are right.
        if a.check_wheels:
            H_g = homography_at_height(K, R, t, 0.0)
            g_px = apply_h(np.linalg.inv(H_g), np.array([centre_mm]))[0]
            if a.space == "frame":
                raster_g, to_w_g, to_i_g, note = plane_view(und, H_g, "frame")
            else:
                mm_g = a.mm_per_px or round(plane_mm_per_px(H_g, g_px), 2)
                raster_g, to_w_g, to_i_g, note = plane_view(und, H_g, "bev", centre_mm,
                                                            mm_g, a.span_mm, 4000.0, y_up)
            if note:
                print(f"    wheel view: {note}")
            shown = [{"name": "constructed wheels",
                      "points": to_i_g(np.array(list(w_world.values()))).tolist(),
                      "closed": False, "colour": "#d29922", "on": True}]
            got = picker.pick_points(
                raster_g, "Click any wheel contact patches you can actually see",
                hint="on the ground plane — where the tyre meets the tarmac; Esc to skip",
                open_browser=not a.no_open)
            if got:
                seen = to_w_g(np.array([q["px"] for q in got]))
                for q in seen:
                    k = min(w_world, key=lambda n: np.linalg.norm(w_world[n] - q))
                    print(f"    clicked wheel near {k}: constructed position is "
                          f"{np.linalg.norm(w_world[k] - q):.0f} mm away")
                shown.append({"name": "clicked", "points": to_i_g(seen).tolist(),
                              "closed": False, "colour": "#f778ba", "on": True})
            if not a.no_preview:
                picker.review_layers(raster_g, "Wheels on the ground plane", shown,
                                     hint="orange is constructed, pink is what you clicked",
                                     open_browser=not a.no_open)
    elif a.front_to_wheel_mm is not None or a.rear_to_wheel_mm is not None:
        raise SystemExit("wheels need both --front-to-wheel-mm and --rear-to-wheel-mm, "
                         "or neither")
    else:
        print("  no wheels: pass --front-to-wheel-mm and --rear-to-wheel-mm. Ten DSR "
              "rules are written about wheels and none of them can be scored without "
              "these.", flush=True)

    # What a wrong height costs here. The height is the one number this step
    # takes on trust, and parallax scales with distance from the camera, so the
    # factor is a property of where the car is standing, not of the car.
    which_h = "--front-height-mm" if a.front_height_mm is not None else "--height-mm"
    print(f"  height sensitivity: 10 mm of error in {which_h} moves the box "
          f"{10 * factor:.0f} mm here (x{factor:.2f})")

    save_json(a.out, {
        "car_id": a.name,
        "sticker_height_mm": float(a.sticker_height_mm),
        "outline_height_mm": float(h_front),
        "front_height_mm": float(h_front),
        "side_height_mm": (float(h_side) if side is not None else None),
        "built_from": "front-2" if n_click == 2 else "corners-4",
        "sticker_yaw_offset_deg": yaw,
        "length_mm": length,
        "width_mm": width,
        "body_polygon_mm": [{"x_mm": float(q[0]), "y_mm": float(q[1])} for q in body],
        "front_bumper_mm": front_mm,
        "rear_bumper_mm": rear_mm,
        "detector_centre_mm": (None if det_centre is None else
                               {"x_mm": float(det_centre[0]), "y_mm": float(det_centre[1])}),
        "detector_offset_mm": (None if det_offset is None else
                               {"x_mm": float(det_offset[0]), "y_mm": float(det_offset[1])}),
        "detector_score": det_score,
        "detector_heading_deg": det_heading,
        "wheels_mm": (None if wheels is None else
                      {k: {"x_mm": float(v[0]), "y_mm": float(v[1])}
                       for k, v in wheels.items()}),
        "front_to_wheel_mm": a.front_to_wheel_mm,
        "rear_to_wheel_mm": a.rear_to_wheel_mm,
        "track_mm": (a.track_mm or width) if wheels else None,
        "wheelbase_mm": (length - a.front_to_wheel_mm - a.rear_to_wheel_mm
                         if wheels else None),
        "tyre_width_mm": a.tyre_width_mm,
        "traced": {
            "corner_order": ["front_left", "front_right", "rear_right", "rear_left"],
            "world_mm": [{"x_mm": float(q[0]), "y_mm": float(q[1])} for q in corners],
            "marker_centre_mm": {"x_mm": float(centre_mm[0]), "y_mm": float(centre_mm[1])},
            "clicked_front_width_mm": click_width,
            "side_skew_mm": skew_side,
            "diagonal_skew_mm": skew_diag,
            "clicked_on": a.space,
            "mm_per_px": (None if a.space == "frame" else mm_o),
        },
        "note": "body_polygon_mm and wheels_mm are in the sticker TEMPLATE frame; the "
                "body was traced on the plane at front_height_mm (the flank, when one "
                "was clicked, on side_height_mm); the wheels are "
                "contact patches on the ground; front/rear_bumper_mm are car-frame "
                "distances from the marker",
    })


# --------------------------------------------------------------------------
# 7. rules — the scoring regions, drawn against the DSR catalogue
# --------------------------------------------------------------------------


def cmd_rules(a) -> None:
    """Seed a station's rules file from the DSR catalogue, then draw its regions.

    One file holds both halves of a score: the rules, copied from
    ``dsr_rules.json`` beside this script, and the regions they are measured
    against, drawn here. Keeping them together is what makes a scorecard
    reproducible — a rule that cites ``stop_line`` and a ``stop_line`` clicked
    six weeks later in another file are only related by hope.

    Run it once to create the file, and again whenever a region moves; the
    second run offers only what is still undrawn unless ``--redraw`` is passed.
    Regions are clicked on the raw undistorted frame and read on the **field**
    plane, because paint and kerbs are on the ground — the same convention the
    `roi` step uses, and the reason both can share a picker.

    ``--add`` appends a region the DSR does not name, for a station with
    something of its own to measure. It is written with no rule attached; give
    it one by editing the file, which is plain JSON on purpose.
    """
    catalogue = Path(__file__).with_name("dsr_rules.json")
    if not catalogue.exists():
        raise SystemExit(f"{catalogue} is missing; it ships beside this script")

    if a.out.exists():
        spec = load_json(a.out)
        print(f"  {a.out} exists — {len(spec['rules'])} rule(s), "
              f"{sum(1 for r in spec['rois'] if r['points_px'])} region(s) already drawn")
    else:
        cat = load_json(catalogue)
        rules = [r for r in cat["rules"] if r["station"] in (a.station, "GEN")]
        wanted = {r["params"]["roi"] for r in rules if r["params"].get("roi")}
        spec = {"schema_version": 1, "station": a.station, "source": cat["source"],
                "rois": [dict(r) for r in cat["rois"] if r["name"] in wanted],
                "rules": rules}
        n_img = sum(1 for r in rules if r["image_processing"])
        print(f"  seeded from {catalogue.name}: {len(rules)} rule(s) for {a.station} "
              f"and GEN, {n_img} of them image-processing")

    if a.add:
        if any(r["name"] == a.add for r in spec["rois"]):
            raise SystemExit(f"{a.out} already has a region called {a.add!r}")
        spec["rois"].append({"name": a.add, "type": a.type, "points_px": [],
                             "note": "added by hand; no DSR rule cites this yet"})
        print(f"  added region {a.add!r} ({a.type})")

    todo = [r for r in spec["rois"] if a.redraw or not r["points_px"]]
    if not todo:
        print("  every region is already drawn — pass --redraw to do them again")
    else:
        und = undistort(load_image(a.image), load_json(a.calibration)["intrinsics"],
                        str(a.image))
        for roi in todo:
            note = f" — {roi['note']}" if roi.get("note") else ""
            used = sorted(r["id"] for r in spec["rules"]
                          if r["params"].get("roi") == roi["name"])
            print(f"  drawing {roi['name']!r}{note}")
            if used:
                print(f"    scored by {', '.join(used)}")
            got = picker.pick_points(
                und, f"{roi['name']} — {roi['type']}{note}",
                n=1 if roi["type"] == "point" else None,
                hint=("click the two ends" if roi["type"] == "line"
                      else "click round it, in order"),
                open_browser=not a.no_open)
            if not got:
                print(f"    skipped {roi['name']!r} — nothing drawn")
                continue
            roi["points_px"] = [[p["px"][0], p["px"][1]] for p in got]

    drawn = [r for r in spec["rois"] if r["points_px"]]
    blocked = sorted({n for r in spec["rules"] for n in r["needs"]})
    save_json(a.out, spec)
    print(f"  {len(drawn)}/{len(spec['rois'])} region(s) drawn")
    undrawn = [r["name"] for r in spec["rois"] if not r["points_px"]]
    if undrawn:
        print(f"  NOTE: {', '.join(undrawn)} still undrawn — every rule citing one of "
              f"those scores as not-evaluated, not as passed.", flush=True)
    if blocked:
        print(f"  NOTE: some rules here need {', '.join(blocked)}, which this pipeline "
              f"does not produce. `score.py` lists them per run rather than passing "
              f"them silently.", flush=True)


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

    p = add("tape", cmd_tape, "3c. check the survey's scale against tapes")
    p.add_argument("--image", type=Path, default=None, help="Not needed with --reuse.")
    p.add_argument("--calibration", required=True, type=Path)
    p.add_argument("--out", type=Path, default=None,
                   help="Default: update the calibration file in place.")
    p.add_argument("--adjust", action="store_true",
                   help="Correct the calibration. Without this the step only reports, which "
                        "is the right default: when the survey is already right, adjusting "
                        "makes it worse.")
    p.add_argument("--reuse", action="store_true",
                   help="Re-run on the bars already stored in the calibration, without "
                        "clicking them again. For trying different sigmas.")
    p.add_argument("--sigma-px", type=float, default=DEFAULT_SIGMA_PX,
                   help="What one click on a mark is worth, in pixels.")
    p.add_argument("--sigma-tape-mm", type=float, default=DEFAULT_SIGMA_TAPE_MM,
                   help="What the tape itself is worth, in millimetres.")
    p.add_argument("--sigma-world-mm", type=float, default=DEFAULT_SIGMA_WORLD_MM,
                   help="How wrong a typed ground control coordinate may be. This is how "
                        "far the tapes are allowed to overrule the original survey.")
    p.add_argument("--max-gcp-residual-mm", type=float, default=50.0,
                   help="Complain when a mark misses its own typed coordinate by this much. "
                        "Same threshold the `gcp` step warns at.")
    p.add_argument("--gate-z", type=float, default=2.0,
                   help="Adjust only when some bar disagrees by more than this many sigma.")
    p.add_argument("--max-scale-error", type=float, default=0.02,
                   help="Refuse a rescale beyond this fraction; nothing internal can check "
                        "one, and a real tape error is tenths of a percent.")
    p.add_argument("--max-bar-z", type=float, default=3.0,
                   help="Refuse if a bar is still this many sigma out after adjusting; no "
                        "pose fits all of them, so one is a blunder.")
    p.add_argument("--max-gcp-move-sigma", type=float, default=4.0,
                   help="Refuse if a mark has to move this many --sigma-world-mm; that is a "
                        "mistyped coordinate rather than survey noise.")
    p.add_argument("--max-shift-mm", type=float, default=500.0,
                   help="Print a note when the ground moves further than this. Not a refusal: "
                        "a survey wrong at the marks is legitimately wronger away from them.")

    p = add("sticker", cmd_sticker, "4. cut the marker template")
    p.add_argument("--image", required=True, type=Path)
    p.add_argument("--calibration", required=True, type=Path)
    p.add_argument("--height-mm", required=True, type=float, help="Marker height above ground.")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--space", choices=("bev", "frame", "raw"), default="bev",
                   help="bev: crop a bird's-eye raster. frame: click 4 corners on the "
                        "frame and warp just those out, metric, with no raster. raw: a "
                        "plain crop, for inspection only.")
    p.add_argument("--pad-mm", type=float, default=0.0,
                   help="Extra margin around the clicked corners, in millimetres.")
    p.add_argument("--clicks", action="store_true",
                   help="Click the marker's 4 corners instead of dragging a box. Lets you "
                        "zoom between clicks, which a drag cannot.")
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

    p = add("outline", cmd_outline, "5b. trace the car's box instead of taping it")
    p.add_argument("--image", required=True, type=Path)
    p.add_argument("--calibration", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--name", default="car")
    p.add_argument("--sticker-height-mm", required=True, type=float,
                   help="Marker height above ground — the plane the marker is read on.")
    p.add_argument("--height-mm", type=float, default=500.0,
                   help="Height of the corners you are clicking. Bumper corners sit near "
                        "500 mm; measure it, because the box error tracks this about 1:1.")
    p.add_argument("--front-height-mm", type=float, default=None,
                   help="Height of the FRONT corners, when they are not at --height-mm. "
                        "Bumper corners sit near 500 mm.")
    p.add_argument("--side-height-mm", type=float, default=None,
                   help="Height of the flank feature the --side clicks follow, when it is "
                        "not at --height-mm. A sill runs lower than a bumper corner and a "
                        "shoulder crease runs higher, and clicking one on the other's "
                        "plane moves it sideways.")
    p.add_argument("--length-mm", type=float, default=None,
                   help="Car length. Given, only the 2 FRONT corners are clicked and the "
                        "rear is built square to them — for when the far end is hidden.")
    p.add_argument("--width-mm", type=float, default=None,
                   help="Car width. Used with --length-mm to override the clicked width, "
                        "and cross-checked against it either way.")
    p.add_argument("--max-box-error-mm", type=float, default=100.0,
                   help="Complain when the clicked width and --width-mm disagree by enough "
                        "to matter, measured as the box error it implies, not as millimetres "
                        "of width.")
    p.add_argument("--max-skew-mm", type=float, default=150.0,
                   help="Complain if a 4-corner trace does not close on itself.")
    p.add_argument("--no-preview", action="store_true",
                   help="Save without showing the box for approval first.")
    p.add_argument("--front-to-wheel-mm", type=float, default=None,
                   help="Front line you clicked, back to the front axle. With "
                        "--rear-to-wheel-mm, places all four contact patches.")
    p.add_argument("--rear-to-wheel-mm", type=float, default=None,
                   help="Rear line of the box, forward to the rear axle.")
    p.add_argument("--space", choices=("bev", "frame"), default="bev",
                   help="Where the BOX lines are clicked — the front edge, the side line "
                        "and the wheel check. bev: on a bird's-eye of the --height-mm "
                        "plane, which shows it flat. frame: on the camera frame, at native "
                        "resolution with no raster, judging edges across the plane rather "
                        "than down at it. The marker centre is always on a bird's-eye "
                        "either way.")
    p.add_argument("--marker-corners", action="store_true",
                   help="Find the marker centre by clicking its 4 corners and crossing the "
                        "diagonals. A corner is the one place on a marker the sub-pixel "
                        "snap can catch; the centre is not.")
    p.add_argument("--max-centre-spread-mm", type=float, default=40.0,
                   help="Complain when the diagonals' crossing point and the mean of the 4 "
                        "corners disagree — on a parallelogram they are the same point.")
    p.add_argument("--template", type=Path, default=None,
                   help="The cut marker template. Given, the matcher is run on this frame "
                        "and the offset between the clicked centre and the one it reports "
                        "is measured and stored.")
    p.add_argument("--template-mm-per-px", type=float, default=None,
                   help="The scale the template was cut at, from sticker.json.")
    p.add_argument("--max-detector-offset-mm", type=float, default=60.0,
                   help="Complain when the template's own centre is this far from the "
                        "marker's.")
    p.add_argument("--marker-span-mm", type=float, default=1500.0,
                   help="Half-width of the marker-centre view. It only has to show the "
                        "marker; the seed click has already found it.")
    p.add_argument("--side", action="store_true",
                   help="Also click 2 points down one flank. The direction and the "
                        "lateral position then come from that long line instead of from "
                        "the short front edge and an assumption about the marker.")
    p.add_argument("--max-level-mm", type=float, default=120.0,
                   help="Complain when the two front clicks are not level with each other "
                        "along the car.")
    p.add_argument("--max-square-mm", type=float, default=60.0,
                   help="Complain when the front edge and the side line disagree about the "
                        "car's direction by enough to matter, measured as the corner "
                        "movement it implies rather than as degrees.")
    p.add_argument("--track-mm", type=float, default=None,
                   help="Left wheel to right wheel. Defaults to the body width, which "
                        "puts each wheel roughly 75-125 mm too far out.")
    p.add_argument("--tyre-width-mm", type=float, default=None,
                   help="Tread width. Lets a rule ask whether a tyre is ON a line "
                        "rather than whether its centre point is.")
    p.add_argument("--check-wheels", action="store_true",
                   help="Click whatever contact patches are visible and compare them "
                        "against the constructed ones.")
    p.add_argument("--mm-per-px", type=float, default=None)
    p.add_argument("--span-mm", type=float, default=6000.0)

    p = add("roi", cmd_roi, "6. draw what to measure the car against")
    p.add_argument("--image", required=True, type=Path)
    p.add_argument("--calibration", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)

    p = add("rules", cmd_rules, "7. draw the scoring regions for a station's DSR rules")
    p.add_argument("--image", required=True, type=Path)
    p.add_argument("--calibration", required=True, type=Path)
    p.add_argument("--station", required=True, help="e.g. B7 — picks its rules plus GEN.")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--redraw", action="store_true", help="Re-draw regions already drawn.")
    p.add_argument("--add", default=None, help="Append a region the DSR does not name.")
    p.add_argument("--type", default="line", choices=("line", "polygon", "point"),
                   help="Shape for --add.")

    a = ap.parse_args()
    print(f"{a.cmd}", flush=True)
    a.fn(a)


if __name__ == "__main__":
    main()
