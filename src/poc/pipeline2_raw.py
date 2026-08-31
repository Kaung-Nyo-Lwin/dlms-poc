"""Pipeline 2 — raw first: match in the camera frame, then map only the centre.

    for each frame:
        undistort (no warp)
        hybrid-detect the sticker in camera pixels (rotation x scale NCC -> ECC)
        push the winning centre through the car-plane homography
        turn that into the car's ground box and measure it against every ROI

Same calibration, same box arithmetic, same output as pipeline 1. What differs
is *when* the warp happens, and therefore what the matcher sees.

Pipeline 1 resamples every frame, so the marker is the same size and shape
wherever the car is and rotation is the only free parameter. Here nothing is
resampled and no full-frame remap is paid for — but an oblique camera makes the
marker smaller further away and shears it, so scale becomes a second free
parameter. It is not hunted blindly: scale is a known function of *where* the
match is, predicted from the homography, and the gap between what the geometry
expects and what the correlation picked is written to the CSV as a direct
readout of how far the raw assumption has been stretched.

Angles need one piece of bookkeeping. A metric template's zero angle means
"aligned with world +X" everywhere; a camera-frame crop's zero angle means
"aligned with the image axes", which is a different world direction at every
pixel. The constant tying the two together is measured here at synthesis time.
Without it the box comes out rotated about the sticker — entirely plausible,
entirely wrong.

Standalone: numpy + opencv only. Nothing here imports dlms.

    python pipeline2_raw.py --video clip.mp4 --calibration calibration.json \
        --car car.json --template sticker.png --template-mm-per-px 9.41 \
        --rois rois.json --out track.csv
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np

# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------


def load_calibration(path: Path) -> dict:
    """The station survey: intrinsics, the ground plane, and the camera pose."""
    raw = json.loads(Path(path).read_text())
    intr = raw.get("intrinsics")
    if intr is None:
        raise SystemExit(
            f"{path} has no intrinsics. Lens distortion would be carried into every "
            "measurement, worst at the frame edges, so this pipeline refuses to guess."
        )
    pose = (raw.get("field") or {}).get("pose")
    return {
        "K": np.array(intr["camera_matrix"], dtype=np.float64),
        "D": np.array(intr["dist_coeffs"], dtype=np.float64).ravel(),
        "model": str(intr.get("model", "pinhole")),
        "width": int(raw["image_width"]),
        "height": int(raw["image_height"]),
        "H_field": np.array(raw["field"]["homography"], dtype=np.float64),
        "R": np.array(pose["rotation"], dtype=np.float64) if pose else None,
        "t": np.array(pose["translation_mm"], dtype=np.float64) if pose else None,
        "car_stored": (
            np.array(raw["car"]["homography"], dtype=np.float64) if raw.get("car") else None
        ),
        "car_height_stored": (raw.get("car") or {}).get("height_mm", 0.0),
    }


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


def homography_at_height(K: np.ndarray, R: np.ndarray, t: np.ndarray, height_mm: float):
    """Image -> world (X, Y) for the horizontal plane Z = height_mm."""
    _, up = camera_from_pose(R, t)
    Hw2i = K @ np.column_stack([R[:, 0], R[:, 1], R[:, 2] * (up * float(height_mm)) + t])
    H = np.linalg.inv(Hw2i)
    return H / H[2, 2]


#: What ``--detector-offset`` stores when given with no value: read the offset
#: `outline` measured out of the car file rather than taking one off the CLI.
USE_CAR_FILE = "<car file>"


def parse_time(text):
    """Seconds from "130", "2:10" or "1:02:03". None passes through.

    Colons because that is how anyone reads a time off a player's scrubber, and
    a clip is nearly always something that was watched first.
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


def detector_offset(arg, car):
    """The correction to add to every detected centre, in the template's frame.

    ``None`` when the option was not given, which leaves the old behaviour: the
    detector's own centre is used as the marker's.
    """
    if arg is None:
        return None
    if arg == USE_CAR_FILE:
        off = car.get("detector_offset_mm")
        if off is None:
            raise SystemExit(
                "--detector-offset with no value reads detector_offset_mm from the car "
                "file, and this one has none. `calibrate.py outline --template ... "
                "--template-mm-per-px ...` measures it, or pass X,Y in millimetres here.")
        return off
    parts = str(arg).split(",")
    try:
        if len(parts) != 2:
            raise ValueError
        return np.array([float(v) for v in parts], dtype=np.float64)
    except ValueError:
        raise SystemExit(
            f"cannot read {arg!r} as an offset; give X,Y in millimetres, or no value at "
            "all to use the car file's own measurement") from None


def load_car(path: Path) -> dict:
    """Vehicle geometry, in the *sticker template* frame.

    Origin at the sticker centre, +X along the template's width axis — the
    template's frame, not the car's. ``sticker_yaw_offset_deg`` is the car's
    forward axis measured from template +X and is applied only when reporting
    heading; a marker turned a quarter turn on the roof has an offset near ±90.
    """
    raw = json.loads(Path(path).read_text())
    yaw = float(raw.get("sticker_yaw_offset_deg", 0.0))
    poly = raw.get("body_polygon_mm") or []
    if poly:
        body = np.array([[p["x_mm"], p["y_mm"]] for p in poly], dtype=np.float64)
    else:
        length, width = float(raw["length_mm"]), float(raw["width_mm"])
        front = float(raw.get("front_bumper_mm") or length / 2.0)
        rear = float(raw.get("rear_bumper_mm") or front - length)
        car = np.array(
            [[front, width / 2], [front, -width / 2], [rear, -width / 2], [rear, width / 2]]
        )
        a = math.radians(yaw)
        c, s = math.cos(a), math.sin(a)
        body = car @ np.array([[c, -s], [s, c]]).T
    wheels = raw.get("wheels_mm") or None
    if wheels:
        wheels = {k: np.array([v["x_mm"], v["y_mm"]], dtype=np.float64)
                  for k, v in wheels.items()}
    # The template's own centre is not always the marker's: cut it a few pixels
    # off and every box hangs off a point beside the car. `outline --template`
    # runs this same matcher on the survey frame and records the gap it found.
    # It is a property of how the template was cut, so it is fixed in the
    # template's frame and turns with the car — which is why it is un-rotated
    # here by the heading the detector reported when it was measured, instead of
    # being kept as the world-frame vector it was measured as. Stored at one
    # pose and applied at another, the difference is the whole offset.
    off = raw.get("detector_offset_mm")
    det_offset = None
    if off:
        ang = -math.radians(float(raw.get("detector_heading_deg") or 0.0))
        c, s_ = math.cos(ang), math.sin(ang)
        det_offset = np.array([[c, -s_], [s_, c]]) @ np.array([off["x_mm"], off["y_mm"]])
    return {
        "detector_offset_mm": det_offset,
        "car_id": str(raw.get("car_id", "car")),
        "body_mm": body,
        "yaw_offset_deg": yaw,
        "sticker_height_mm": float(raw["sticker_height_mm"]),
        # Contact patches, in the same template frame as the body and moved by
        # the same rigid transform. Absent when the survey did not measure them,
        # and every wheel rule then scores as not-evaluated rather than as passed.
        "wheels_mm": wheels,
        "tyre_width_mm": raw.get("tyre_width_mm"),
    }


def load_rois(path: Path, H_field: np.ndarray, K, D, model, distorted: bool) -> list[dict]:
    """Clicked ground features, converted from pixels to world millimetres.

    Read through the *field* plane, because that is where paint and kerbs are.
    """
    raw = json.loads(Path(path).read_text())
    out = []
    for item in raw["rois"]:
        pts = np.array(item["points_px"], dtype=np.float64).reshape(-1, 2)
        if distorted:
            pts = undistort_points(pts, K, D, model)
        world = apply_h(H_field, pts)
        if not np.isfinite(world).all():
            raise SystemExit(
                f"ROI {item['name']!r} has a point on the ground plane's horizon, "
                "where distances are undefined"
            )
        kind = item.get("type", "point" if len(pts) == 1 else "line")
        out.append({"name": item["name"], "world_mm": world, "closed": kind == "polygon"})
    return out


# --------------------------------------------------------------------------
# the plane, seen from the camera frame
# --------------------------------------------------------------------------


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


def apply_h(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    q = np.column_stack([p, np.ones(len(p))]) @ np.asarray(H, dtype=np.float64).T
    w = q[:, 2:3]
    w = np.where(np.abs(w) < 1e-12, 1e-12, w)
    return q[:, :2] / w


def norm_deg(a: float) -> float:
    a = (float(a) + 180.0) % 360.0 - 180.0
    return 180.0 if a == -180.0 else a


class PlaneMap:
    """One plane's homography, plus the local quantities raw matching needs."""

    def __init__(self, H: np.ndarray):
        self.H = np.asarray(H, dtype=np.float64)

    def to_world(self, pts):
        return apply_h(self.H, pts)

    def to_pixel(self, pts):
        return apply_h(np.linalg.inv(self.H), pts)

    def jacobian(self, p, eps=0.5):
        """d(world mm)/d(image px) by central differences.

        Not the analytic form: the homography's quotient rule is easy to get
        subtly wrong, and at this step size the difference is far below anything
        that matters.
        """
        p = np.asarray(p, dtype=np.float64).reshape(1, 2)
        ex = np.array([[eps, 0.0]])
        ey = np.array([[0.0, eps]])
        dx = (self.to_world(p + ex)[0] - self.to_world(p - ex)[0]) / (2 * eps)
        dy = (self.to_world(p + ey)[0] - self.to_world(p - ey)[0]) / (2 * eps)
        return np.column_stack([dx, dy])

    def mm_per_px(self, p) -> float:
        """Area-preserving local scale, sqrt(|det J|) — the geometric mean of the
        two axis steps, which is what keeps a resampled template's area right."""
        return float(math.sqrt(abs(np.linalg.det(self.jacobian(p)))))

    def anisotropy(self, p) -> float:
        """How far from a similarity the local map is: ratio of singular values.

        1.0 means the marker is merely scaled and rotated there, so one scalar
        scale describes it exactly. Larger means it is also squashed, and a
        rotate-and-scale template cannot represent it — precisely the error raw
        matching pays and bird's-eye matching does not.
        """
        s = np.linalg.svd(self.jacobian(p), compute_uv=False)
        return float(s[0] / max(s[1], 1e-12))

    def world_heading(self, p, theta_deg: float, reach_px: float = 8.0) -> float:
        """World bearing of an image-space direction at one pixel.

        ``theta_deg`` is anticlockwise on screen, where rows increase downward.
        Mapping two points and taking the angle between them keeps the Y flip in
        one place instead of scattering sign conventions.
        """
        p = np.asarray(p, dtype=np.float64).reshape(2)
        a = math.radians(theta_deg)
        tip = p + np.array([math.cos(a), -math.sin(a)]) * reach_px
        base_w, tip_w = self.to_world(np.array([p, tip]))
        d = tip_w - base_w
        return float(math.degrees(math.atan2(d[1], d[0])))

    def in_front(self, pts, reference_px) -> np.ndarray:
        """Which pixels see this plane in front of the camera rather than behind.

        The projective denominator is linear in pixel coordinates: positive one
        side of the plane's horizon, negative the other. Pixels past it still
        produce perfectly finite world coordinates — mirrored through the camera
        — so a finiteness check does not catch them. Which sign means "in front"
        depends on how the survey was set up, so it is taken from a pixel known
        to be genuinely on the plane.
        """
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        h3 = self.H[2]
        ref = np.asarray(reference_px, dtype=np.float64).reshape(2)
        sign = 1.0 if float(ref @ h3[:2] + h3[2]) >= 0 else -1.0
        return ((pts @ h3[:2] + h3[2]) * sign) > 1e-9


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------


def car_box(body_template_mm, center_mm, template_heading_deg, yaw_offset_deg):
    """Sticker pose -> the car's four ground corners, in world millimetres."""
    a = math.radians(template_heading_deg)
    c, s = math.cos(a), math.sin(a)
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    world = np.asarray(body_template_mm, dtype=np.float64) @ R.T + np.asarray(center_mm).reshape(2)
    if len(world) != 4:
        world = np.array(cv2.boxPoints(cv2.minAreaRect(world.astype(np.float32))), dtype=np.float64)
    return world, norm_deg(template_heading_deg + yaw_offset_deg)


def wheels_world(wheels_mm, centre_mm, template_heading_deg, Rm=None,
                 sticker_height_mm=0.0):
    """Contact patches in world millimetres, moved by the marker's pose.

    The same rigid transform the body box gets, and for the same reason: the
    wheels are fixed in the car, so whatever moved the marker moved them. When
    a tilt was measured they go through it in three dimensions like the body,
    sitting one sticker height below the marker in its own frame.
    """
    pts = np.array([wheels_mm[n] for n in ("fl", "fr", "rl", "rr")], dtype=np.float64)
    if Rm is None:
        a = math.radians(template_heading_deg)
        c, s = math.cos(a), math.sin(a)
        return pts @ np.array([[c, -s], [s, c]], dtype=np.float64).T \
            + np.asarray(centre_mm).reshape(2)
    p3 = np.column_stack([pts, np.full(len(pts), -float(sticker_height_mm))])
    return (p3 @ np.asarray(Rm, dtype=np.float64).T)[:, :2] \
        + np.asarray(centre_mm).reshape(2)


def sticker_quad(centre_mm, heading_deg, width_mm, height_mm):
    """The matched template's own outline, in world millimetres.

    These are horizontal positions on the *marker's* plane, not the ground: the
    quad sits at ``sticker_height_mm``, and drawing it through the ground
    homography would put it a couple of metres away from the roof it is on.
    """
    a = math.radians(heading_deg)
    c, s = math.cos(a), math.sin(a)
    half = np.array([[-width_mm / 2, -height_mm / 2], [width_mm / 2, -height_mm / 2],
                     [width_mm / 2, height_mm / 2], [-width_mm / 2, height_mm / 2]])
    return half @ np.array([[c, -s], [s, c]]).T + np.asarray(centre_mm).reshape(2)


def _seg_dist(p, a, b) -> float:
    d = b - a
    L = float(d @ d)
    t = 0.0 if L < 1e-12 else float(np.clip((p - a) @ d / L, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * d)))


def _cross(p, q, r) -> float:
    """Which side of the line p->q the point r falls on."""
    return float((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))


def _segments_cross(a, b, c, d) -> bool:
    d1, d2 = _cross(c, d, a), _cross(c, d, b)
    d3, d4 = _cross(a, b, c), _cross(a, b, d)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


def _inside(poly, p) -> bool:
    return cv2.pointPolygonTest(poly.astype(np.float32), (float(p[0]), float(p[1])), False) >= 0


def line_penetration_mm(box, a, b) -> float:
    """How far the box would have to move to get off the line through ``a``-``b``.

    The shortest distance between two outlines that cross is zero, and it stays
    zero however far the car drives on — so a graze and a bumper 800 mm over a
    boundary read alike. The shortest distance that *means* something once they
    overlap is the shortest one that separates them: push the box perpendicular
    to the line until every corner is on one side, whichever side is cheaper.

    Only ever asked of an open line. A closed shape has an inside, so "which
    side" is not a question with two answers, and its zero is left alone.
    """
    e = np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)
    n = np.array([-e[1], e[0]])
    ln = float(np.linalg.norm(n))
    if ln < 1e-9:
        return 0.0
    d = (np.asarray(box, dtype=np.float64) - np.asarray(a, dtype=np.float64)) @ (n / ln)
    # Deepest corner either side; the cheaper push is the shallower of the two.
    return float(min(max(d.max(), 0.0), max(-d.min(), 0.0)))


def clearance_mm(box, roi_pts, closed) -> tuple[float, bool]:
    """Gap between the car box and one ROI, in millimetres.

    Negative when the ROI lies wholly under the car, and otherwise the shortest
    distance between the two outlines. The boolean is the thing a rule actually
    asks — "is the car on it?" — kept separate so a clearance of 0.0 is never
    confused with a near miss rounded down.

    An **open line** the box straddles is the one case where zero is not the
    useful answer, because the distance between the outlines is zero from the
    moment they touch and says nothing about how far the car went on. There the
    gap is negative and its size is how far the box would have to move to come
    off the line — see :func:`line_penetration_mm`. Polygons keep their zero:
    a closed shape has an inside, and how far a car has intruded into a region
    is a different question from how far it has crossed a boundary.
    """
    roi = np.asarray(roi_pts, dtype=np.float64).reshape(-1, 2)
    edges = [(box[i], box[(i + 1) % 4]) for i in range(4)]
    roi_segs = []
    if len(roi) >= 2:
        n = len(roi) if closed else len(roi) - 1
        roi_segs = [(roi[i], roi[(i + 1) % len(roi)]) for i in range(n)]

    crossed = [(c, d) for a, b in edges for c, d in roi_segs
               if _segments_cross(a, b, c, d)]
    if crossed:
        if closed:
            return 0.0, True
        # The deepest of the crossed segments: a polyline can clip a corner with
        # one segment and run under the whole car with the next, and the run is
        # the one worth reporting.
        return -max(line_penetration_mm(box, c, d) for c, d in crossed), True

    best = min(_seg_dist(p, a, b) for p in roi for a, b in edges)
    if roi_segs:
        best = min(best, min(_seg_dist(q, c, d) for q in box for c, d in roi_segs))
    all_in = all(_inside(box, p) for p in roi)
    return (-best if all_in else best), all_in


# --------------------------------------------------------------------------
# detection
# --------------------------------------------------------------------------


def preprocess(img: np.ndarray) -> np.ndarray:
    """clahe+unsharp, applied identically to the template and to every frame."""
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = g.shape[:2]
    grid = (max(1, round(w / 192)), max(1, round(h / 192)))
    g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=grid).apply(g)
    return cv2.addWeighted(g, 2.2, cv2.GaussianBlur(g, (0, 0), 2.0), -1.2, 0)


def circular_mask(shape) -> np.ndarray:
    """Rotation-invariant support: the same pixels are compared at every angle."""
    h, w = shape
    m = np.zeros((h, w), np.uint8)
    cv2.circle(m, (w // 2, h // 2), max(int(min(h, w) * 0.5) - 1, 1), 255, -1)
    return m


def subpixel_peak(resp, x, y) -> tuple[float, float]:
    h, w = resp.shape
    dx = dy = 0.0
    if 0 < x < w - 1:
        a, b, c = float(resp[y, x - 1]), float(resp[y, x]), float(resp[y, x + 1])
        d = a - 2 * b + c
        if abs(d) > 1e-12:
            dx = float(np.clip(0.5 * (a - c) / d, -1, 1))
    if 0 < y < h - 1:
        a, b, c = float(resp[y - 1, x]), float(resp[y, x]), float(resp[y + 1, x])
        d = a - 2 * b + c
        if abs(d) > 1e-12:
            dy = float(np.clip(0.5 * (a - c) / d, -1, 1))
    return x + dx, y + dy


def synthesise_raw_template(template, template_mm_per_px, plane: PlaneMap, ref_mm,
                            y_up=True):
    """Project the metric template onto the camera frame at one world point.

    This is what removes the need for a hand-cut camera-frame crop: the metric
    template already says what the marker looks like from directly above, and
    the plane homography says how the camera sees that patch of ground. Warping
    one through the other produces the crop the matcher needs, at a known
    reference scale, together with the yaw constant that ties the crop's image
    axes back to the metric template's world-aligned frame.
    """
    th, tw = template.shape[:2]
    # Template pixels are metric with +X right. Whether a row down the template
    # is world -Y or world +Y is whichever way the raster it was cut from ran,
    # and that follows the camera: a left-handed world frame is seen mirrored
    # from above. Getting this backwards costs nothing visible — the crop still
    # looks like a marker — and matches nothing.
    sy = -1.0 if y_up else 1.0
    corners_px = np.array([[0, 0], [tw, 0], [tw, th], [0, th]], dtype=np.float32)
    corners_mm = np.array(
        [[ref_mm[0] + (u - tw / 2) * template_mm_per_px,
          ref_mm[1] + sy * (v - th / 2) * template_mm_per_px] for u, v in corners_px]
    )
    img_corners = plane.to_pixel(corners_mm).astype(np.float32)
    x0, y0 = img_corners.min(axis=0)
    x1, y1 = img_corners.max(axis=0)
    w, h = math.ceil(x1 - x0), math.ceil(y1 - y0)
    if not (4 <= w <= 4096 and 4 <= h <= 4096):
        raise SystemExit(
            f"the marker projects to {w}x{h} px at the reference point; it is off the "
            "plane or outside the frame. Pass --ref-mm X,Y somewhere the car actually goes."
        )
    M = cv2.getPerspectiveTransform(corners_px, img_corners - [x0, y0])
    crop = cv2.warpPerspective(template, M, (w, h), flags=cv2.INTER_CUBIC)
    ref_px = plane.to_pixel(np.array([ref_mm]))[0]
    # A match at theta=0 here means the metric template's +X lies along world +X,
    # so the constant is whatever world bearing image +X has at this pixel.
    yaw_offset = -plane.world_heading(ref_px, 0.0)
    # The crop is the projected quad's bounding box, so the marker's own corners
    # sit inside it at an angle. A corner fit needs to know where.
    quad_in_crop = img_corners - [x0, y0]
    return crop, float(plane.mm_per_px(ref_px)), float(yaw_offset), quad_in_crop


def ecc_refine(pre, tpl, x, y, theta_deg):
    """Sub-pixel Euclidean refinement of one pose against one template.

    Solves for a continuous warp by gradient ascent on the correlation
    coefficient, so the answer is not quantised to the correlation grid the way
    an NCC peak is. This is where the millimetres are won.
    """
    th, tw = tpl.shape[:2]
    # Keep the padded canvas the same parity as the template, or the template
    # lands half a pixel off centre and that becomes a fixed positional bias.
    ph, pw = int(th * 1.35) // 2 * 2 + th % 2, int(tw * 1.35) // 2 * 2 + tw % 2
    a = math.radians(theta_deg)
    c, s = math.cos(a), math.sin(a)
    M = np.array([[c, s, x - (c * (pw / 2 - 0.5) + s * (ph / 2 - 0.5))],
                  [-s, c, y - (-s * (pw / 2 - 0.5) + c * (ph / 2 - 0.5))]], dtype=np.float64)
    patch = cv2.warpAffine(pre, M, (pw, ph),
                           flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP, borderValue=0)
    if patch.size == 0 or float(patch.std()) < 1e-3:
        return None
    canvas = np.zeros((ph, pw), tpl.dtype)
    y0, x0 = (ph - th) // 2, (pw - tw) // 2
    canvas[y0:y0 + th, x0:x0 + tw] = tpl
    mask = np.zeros((ph, pw), np.uint8)
    mask[y0:y0 + th, x0:x0 + tw] = 255
    try:
        cc, warp = cv2.findTransformECC(
            canvas.astype(np.float32), patch.astype(np.float32),
            np.eye(2, 3, dtype=np.float32), cv2.MOTION_EUCLIDEAN,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-7), mask, 5)
    except cv2.error:
        return None                        # ECC raises when it cannot converge
    if not np.isfinite(warp).all():
        return None
    T = np.array([[1.0, 0, pw / 2 - 0.5], [0, 1.0, ph / 2 - 0.5], [0, 0, 1.0]])
    prior = np.array([[c, s, x], [-s, c, y], [0, 0, 1.0]])
    total = prior @ np.linalg.inv(T) @ np.vstack([warp, [0, 0, 1.0]]) @ T
    return (float(cc), float(total[0, 2]), float(total[1, 2]),
            norm_deg(math.degrees(math.atan2(-total[1, 0], total[0, 0]))))


#: Short-side pixels below which the eight-parameter corner fit stops being
#: trustworthy. Measured, not guessed: at 36 px a level marker read as 1.5 deg
#: tilted, while at 63 px the same scene read within 0.2 deg and tracked a real
#: tilt to better than 0.2 deg.
CORNER_PNP_MIN_PX = 60


def ecc_corners(pre, tpl, quad_in_tpl, x, y, theta_deg):
    """The marker's four corners in image pixels, refit with tilt allowed.

    ``ecc_refine`` solves three numbers — two of position and one of rotation —
    which describes a marker lying flat at exactly the surveyed height. A marker
    on a braking car does not lie flat: the roof dips, and its image stops being
    the shape the plane predicts. Three numbers cannot represent that, so the
    dip is absorbed into position instead, which is the error a pose solve
    exists to remove.

    Seeded from the Euclidean fit, because ECC converges only from a good start
    and eight parameters from cold on a small marker will wander.
    """
    th, tw = tpl.shape[:2]
    ph, pw = int(th * 1.35) // 2 * 2 + th % 2, int(tw * 1.35) // 2 * 2 + tw % 2
    a = math.radians(theta_deg)
    c, s = math.cos(a), math.sin(a)
    M = np.array([[c, s, x - (c * (pw / 2 - 0.5) + s * (ph / 2 - 0.5))],
                  [-s, c, y - (-s * (pw / 2 - 0.5) + c * (ph / 2 - 0.5))]], dtype=np.float64)
    patch = cv2.warpAffine(pre, M, (pw, ph),
                           flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP, borderValue=0)
    if patch.size == 0 or float(patch.std()) < 1e-3:
        return None
    canvas = np.zeros((ph, pw), tpl.dtype)
    y0, x0 = (ph - th) // 2, (pw - tw) // 2
    canvas[y0:y0 + th, x0:x0 + tw] = tpl
    mask = np.zeros((ph, pw), np.uint8)
    mask[y0:y0 + th, x0:x0 + tw] = 255
    try:
        _cc, warp = cv2.findTransformECC(
            canvas.astype(np.float32), patch.astype(np.float32),
            np.eye(2, 3, dtype=np.float32), cv2.MOTION_EUCLIDEAN,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-7), mask, 5)
        seed = np.vstack([warp, [0, 0, 1]]).astype(np.float32)
        cc, warp = cv2.findTransformECC(
            canvas.astype(np.float32), patch.astype(np.float32),
            seed, cv2.MOTION_HOMOGRAPHY,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-7), mask, 5)
    except cv2.error:
        return None                        # ECC raises when it cannot converge
    if not np.isfinite(warp).all():
        return None
    # The crop holds the projected quad at an angle inside its bounding box, so
    # the corners come from where it was drawn, not from the crop's own edges.
    corners = np.asarray(quad_in_tpl, dtype=np.float64) + np.array([x0, y0], dtype=np.float64)
    q = np.column_stack([corners, np.ones(len(corners))]) @ np.asarray(warp, dtype=np.float64).T
    q = q[:, :2] / q[:, 2:]
    out = np.column_stack([q, np.ones(len(q))]) @ M.T
    # synthesise_raw_template lists corners +X right, +Y down, i.e. world -Y;
    # marker_pose wants them from (-w/2, -h/2) round. That is the reverse order.
    return (out[::-1], float(cc)) if np.isfinite(out).all() else None


def marker_pose(K, R, t, corners_px, width_mm, height_mm):
    """Marker -> world rotation and translation, from the four corners in pixels.

    IPPE is used for the same reason the survey uses it, and with the same
    caveat: a planar target has two poses that reproject almost equally well,
    so both are asked for and the better one kept.
    """
    obj = np.array([[-width_mm / 2, -height_mm / 2, 0.0],
                    [width_mm / 2, -height_mm / 2, 0.0],
                    [width_mm / 2, height_mm / 2, 0.0],
                    [-width_mm / 2, height_mm / 2, 0.0]], dtype=np.float64)
    img = np.asarray(corners_px, dtype=np.float64).reshape(-1, 1, 2)
    zero = np.zeros(5)
    try:
        n, rvecs, tvecs, err = cv2.solvePnPGeneric(
            obj.reshape(-1, 1, 3), img, K, zero, flags=cv2.SOLVEPNP_IPPE)
    except cv2.error:
        return None
    if not n:
        return None
    best = int(np.argmin([float(np.ravel(e)[0]) for e in err]))
    rvec, tvec = cv2.solvePnPRefineLM(obj.reshape(-1, 1, 3), img, K, zero,
                                      rvecs[best], tvecs[best])
    Rm, _ = cv2.Rodrigues(rvec)
    return R.T @ Rm, R.T @ (tvec.ravel() - t)


def car_box_from_tilt(body_template_mm, centre_mm, Rm, sticker_height_mm):
    """Car footprint and template heading, in world millimetres.

    Position comes from the planar read and orientation from the corner fit,
    because each is the better measurement of its own quantity. Pinning the
    marker to its surveyed height is strong information, and a pose solved from
    four corners alone throws it away: tilt and depth trade off along the
    viewing ray, so the solved centre drifts by tens of millimetres while the
    planar read holds a few. Measured here, the full pose put the footprint out
    by 77 mm where the planar centre held 6 mm.

    What the planar read cannot see is the tilt, and tilt is what moves the
    footprint out from under a marker no longer sitting square above it.
    """
    body = np.asarray(body_template_mm, dtype=np.float64)
    pts = np.column_stack([body, np.full(len(body), -float(sticker_height_mm))])
    xy = (pts @ np.asarray(Rm, dtype=np.float64).T)[:, :2] + np.asarray(centre_mm).reshape(2)
    if len(xy) != 4:
        xy = np.array(cv2.boxPoints(cv2.minAreaRect(xy.astype(np.float32))), dtype=np.float64)
    fwd = np.asarray(Rm)[:, 0]
    return xy, math.degrees(math.atan2(fwd[1], fwd[0]))


def pose_tilt_deg(Rm):
    """Pitch and roll of the marker's plane, in degrees, for the record."""
    Rm = np.asarray(Rm, dtype=np.float64)
    return (math.degrees(math.asin(float(np.clip(-Rm[2, 0], -1.0, 1.0)))),
            math.degrees(math.asin(float(np.clip(Rm[2, 1], -1.0, 1.0)))))


def polish_where_it_is(pre, plane, template, template_mm_per_px, x, y, theta_deg,
                       y_up=True):
    """Re-cut the crop at the position just found, then refine once more.

    A camera-frame crop is only strictly right at the place it was cut. Away
    from there the marker is a different size and, worse, sheared — and a
    rotate-and-scale template cannot represent shear at all, which shows up as a
    biased *angle*, not as a failure to match. Synthesising a fresh crop through
    the homography at the position just found makes the template exact for that
    spot. It also makes the bookkeeping fall out: the crop is world-aligned
    there, so the angle ECC returns needs only that point's own yaw constant.

    Returns ``(x, y, template_frame_heading_deg, cc)`` or None.
    """
    centre_mm = plane.to_world(np.array([[x, y]]))[0]
    try:
        crop, _, yaw, _quad = synthesise_raw_template(
            template, template_mm_per_px, plane, centre_mm, y_up)
    except SystemExit:
        return None                        # marker projects off-frame here
    got = ecc_refine(pre, preprocess(crop), x, y, theta_deg)
    if got is None:
        return None
    cc, nx, ny, ntheta = got
    if math.hypot(nx - x, ny - y) > 8.0 or abs(norm_deg(ntheta - theta_deg)) > 12.0:
        return None                        # it has locked onto something else
    return nx, ny, norm_deg(plane.world_heading(np.array([nx, ny]), ntheta) + yaw), cc


class RawHybridDetector:
    """Rotation x scale NCC to find it, ECC to place it — in camera pixels.

    Scale is quantised into bands rather than swept continuously: it is a known
    function of position, so a handful of bands spanning what the plane predicts
    over the search region covers every place the marker can be.
    """

    def __init__(self, raw_template, bands, coarse_step=12.0, fine_step=2.0, min_score=0.40):
        self.base = preprocess(raw_template)
        #: (scale, (x0, y0, x1, y1)) — each band searches only the strip of the
        #: frame where the plane predicts that scale, which is what keeps a
        #: multi-band search from costing a multiple of a single-scale one.
        self.bands = list(bands)
        self.coarse_step, self.fine_step, self.min_score = coarse_step, fine_step, min_score
        self._bank: dict[tuple[float, float, int], tuple[np.ndarray, np.ndarray]] = {}
        self._scaled: dict[float, np.ndarray] = {}

    def scaled(self, band: float) -> np.ndarray:
        if band not in self._scaled:
            h, w = self.base.shape[:2]
            size = (max(6, round(w * band)), max(6, round(h * band)))
            interp = cv2.INTER_AREA if band < 1 else cv2.INTER_CUBIC
            self._scaled[band] = cv2.resize(self.base, size, interpolation=interp)
        return self._scaled[band]

    def radius(self, band: float) -> float:
        return 0.5 * math.hypot(*self.scaled(band).shape[:2])

    def _rotated(self, angle, band, pyr):
        key = (round(angle, 3), band, pyr)
        if key not in self._bank:
            tpl = self.scaled(band)
            if pyr > 1:
                tpl = cv2.resize(tpl, (max(4, tpl.shape[1] // pyr), max(4, tpl.shape[0] // pyr)),
                                 interpolation=cv2.INTER_AREA)
            mask = circular_mask(tpl.shape[:2])
            h, w = tpl.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2 - 0.5, h / 2 - 0.5), angle, 1.0)
            rot = cv2.warpAffine(cv2.bitwise_and(tpl, tpl, mask=mask), M, (w, h), cv2.INTER_CUBIC)
            self._bank[key] = (rot, mask)
        return self._bank[key]

    def _best_at(self, win, angle, band, pyr):
        tpl, mask = self._rotated(angle, band, pyr)
        th, tw = tpl.shape[:2]
        if win.shape[0] < th or win.shape[1] < tw:
            return None
        resp = cv2.matchTemplate(win, tpl, cv2.TM_CCOEFF_NORMED, mask=mask)
        resp = np.nan_to_num(resp, nan=-1.0, posinf=-1.0, neginf=-1.0)
        _, score, _, loc = cv2.minMaxLoc(resp)
        px, py = subpixel_peak(resp, loc[0], loc[1])
        return float(score), float(angle), px + tw / 2.0, py + th / 2.0

    def _sweep(self, win, angles, band, pyr):
        hits = [h for h in (self._best_at(win, float(a), band, pyr) for a in angles) if h]
        hits.sort(reverse=True)
        return hits

    def _pyr_for(self, band):
        side = min(self.scaled(band).shape[:2])
        return 1 << math.floor(math.log2(max(1, min(8, side // 12))))

    def _ecc(self, pre, x, y, theta, band):
        return ecc_refine(pre, self.scaled(band), x, y, theta)

    def detect(self, pre, prior=None, search_px=None):
        """Returns (x, y, theta_deg, score, sigma_px, band, method) in frame pixels.

        ``pre`` is the whole frame, already preprocessed: each band windows out
        of it rather than being preprocessed itself, because CLAHE is spatially
        adaptive and equalising a crop is a different operator from equalising
        the frame and cropping.
        """
        angles = list(np.arange(0.0, 360.0, self.coarse_step))
        bands = self.bands
        clip = None
        if prior is not None and search_px:
            px, py, ptheta, pband = prior
            r = search_px + self.radius(pband)
            clip = (px - r, py - r, px + r, py + r)
            # Centred on the prior, so the angle it was last found at is
            # actually sampled. An arange from ptheta-30 lands on ptheta+/-6
            # and never on ptheta itself — which on a several-hundred-pixel
            # template is enough to drop the peak under the score floor.
            angles = [ptheta + k * self.coarse_step for k in range(-2, 3)]
            near = [b for b in self.bands if abs(math.log(b[0] / pband)) < 0.35]
            bands = near or [(pband, self.bands[0][1])]

        # Coarse across every band, fine for the winner only. The coarse pass
        # is cheap (downscaled frame, downscaled template); the full-resolution
        # sweep with a masked several-hundred-pixel template is not, and only
        # one band can be right, so paying for it once is the whole difference
        # between a usable search and a hopeless one.
        best = None
        for band, rect in bands:
            x0, y0, x1, y1 = rect
            if clip is not None:
                x0, y0 = max(x0, int(clip[0])), max(y0, int(clip[1]))
                x1, y1 = min(x1, int(clip[2]) + 1), min(y1, int(clip[3]) + 1)
            if x1 - x0 < 8 or y1 - y0 < 8:
                continue
            win = pre[y0:y1, x0:x1]
            # Use the pyramid even when tracking. Pipeline 1 can afford a
            # full-resolution local sweep because its template is ~85 px; here
            # the crop runs to several hundred, and masked correlation costs the
            # product of the two areas — seconds per frame rather than
            # milliseconds. The site re-cut below recovers the precision.
            pyr = self._pyr_for(band)
            if pyr > 1:
                win = cv2.resize(win, (max(8, win.shape[1] // pyr), max(8, win.shape[0] // pyr)),
                                 interpolation=cv2.INTER_AREA)
            hit = self._sweep(win, angles, band, pyr)[:1]
            if not hit:
                continue
            sc, ba, cx, cy = hit[0]
            cand = (sc, ba, cx * pyr + x0, cy * pyr + y0, band, pyr)
            if best is None or cand[0] > best[0]:
                best = cand

        if best is not None:
            # Sharpen the angle one level up the pyramid, not at full resolution.
            # Masked correlation costs the product of the window and template
            # areas, and this template is hundreds of pixels wide — a full-res
            # sweep here is most of the frame budget. Precision is not lost:
            # polish_where_it_is re-cuts an exact template at this position and
            # refines against it, which is strictly better than a sharper peak
            # from a template that is the wrong shape anyway.
            sc, ba, cx, cy, band, pyr = best
            fpyr = max(1, pyr // 2)
            r = self.radius(band) + 3.0 * pyr
            sx, sy = max(0, int(cx - r)), max(0, int(cy - r))
            sub = pre[sy:min(pre.shape[0], int(cy + r) + 1), sx:min(pre.shape[1], int(cx + r) + 1)]
            if fpyr > 1:
                sub = cv2.resize(sub, (max(8, sub.shape[1] // fpyr), max(8, sub.shape[0] // fpyr)),
                                 interpolation=cv2.INTER_AREA)
            fine = self._sweep(sub, np.arange(ba - self.coarse_step, ba + self.coarse_step + 1e-6,
                                              self.fine_step), band, fpyr)
            best = ((fine[0][0], fine[0][1], fine[0][2] * fpyr + sx, fine[0][3] * fpyr + sy, band)
                    if fine else (sc, ba, cx, cy, band))

        if best is None or best[0] < self.min_score:
            return None
        score, theta, x, y, band = best
        theta, sigma, method = norm_deg(theta), 0.5, "ncc"
        got = self._ecc(pre, x, y, theta, band)
        if got is not None:
            cc, rx, ry, rtheta = got
            if math.hypot(rx - x, ry - y) <= 6.0 and abs(norm_deg(rtheta - theta)) <= 8.0:
                x, y, theta, sigma, method = rx, ry, rtheta, max(0.02, 0.5 * (1 - cc)), "hybrid:ecc"
        return x, y, theta, score, sigma, band, method


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--calibration", required=True, type=Path)
    ap.add_argument("--car", required=True, type=Path)
    ap.add_argument("--template", required=True, type=Path, help="Bird's-eye sticker template PNG.")
    ap.add_argument("--template-mm-per-px", required=True, type=float)
    ap.add_argument("--rois", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--ref-mm", type=str, default=None, help="X,Y mm to synthesise the crop at.")
    ap.add_argument("--margin-mm", type=float, default=5000.0, help="Search margin around the ROIs.")
    ap.add_argument("--roi-distorted", action="store_true", help="ROI pixels were clicked on the raw frame.")
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--start", default=None,
                    help="Skip to this point before tracking: seconds, m:ss or h:mm:ss. "
                         "The video is sought rather than decoded up to, so a late clip "
                         "costs no more than an early one.")
    ap.add_argument("--end", default=None, help="Stop here. Same formats.")
    ap.add_argument("--detector-offset", nargs="?", const=USE_CAR_FILE, default=None,
                    metavar="X,Y",
                    help="Correct for a template cut off-centre. With no value, use the "
                         "detector_offset_mm `outline --template` measured; or give X,Y in "
                         "millimetres. Applied in the template's frame, so it turns with "
                         "the car.")
    ap.add_argument("--min-score", type=float, default=0.40)
    ap.add_argument("--search-px", type=float, default=120.0)
    ap.add_argument("--corner-pnp", action="store_true",
                    help="Solve the marker's tilt from its four corners instead of "
                         "assuming it lies flat. Costs a second ECC fit per frame.")
    args = ap.parse_args()

    cal = load_calibration(args.calibration)
    car = load_car(args.car)
    rois = load_rois(args.rois, cal["H_field"], cal["K"], cal["D"], cal["model"],
                     args.roi_distorted)

    # The plane the sticker is read on. A surveyed car plane wins at the height
    # it was surveyed at: it measured the geometry with poles instead of
    # trusting the pose to be exact. It is right only at that one height, so a
    # marker at any other height falls back to raising Z through the pose,
    # which is exact for this car if the pose is.
    stored_h = float(cal["car_height_stored"])
    if (cal["car_stored"] is not None
            and abs(stored_h - car["sticker_height_mm"]) <= 1.0):
        H_car = cal["car_stored"]
        plane_note = f"surveyed car plane @ {stored_h:.0f} mm"
    elif cal["R"] is not None:
        H_car = homography_at_height(cal["K"], cal["R"], cal["t"], car["sticker_height_mm"])
        plane_note = f"synthesised @ {car['sticker_height_mm']:.0f} mm"
        if cal["car_stored"] is not None:
            plane_note += f" (surveyed plane is at {stored_h:.0f} mm, not this height)"
    elif cal["car_stored"] is not None:
        H_car = cal["car_stored"]
        plane_note = f"surveyed car plane @ {stored_h:.0f} mm, but this marker sits at "\
                     f"{car['sticker_height_mm']:.0f} mm"
    else:
        raise SystemExit(
            "calibration has neither a camera pose nor a car plane, so the sticker could "
            "only be read on the ground — which misplaces a roof marker by metres."
        )
    plane = PlaneMap(H_car)
    # Same handedness the template was cut with; see synthesise_raw_template.
    y_up = True if cal["R"] is None else camera_from_pose(cal["R"], cal["t"])[1] > 0
    if not y_up:
        print("  world frame is left-handed — template mapped mirrored to match the "
              "camera, and the survey", flush=True)
    if args.corner_pnp and cal["R"] is None:
        raise SystemExit(
            "--corner-pnp needs the camera pose to turn corners into a marker attitude; "
            "this calibration has none. Re-run the `gcp` survey step."
        )

    # Search region: the ROI world extent plus a margin, projected onto the car
    # plane and clipped to the frame. Anything past the plane's horizon is
    # dropped rather than clamped — world coordinates there are mirrored.
    allw = np.vstack([r["world_mm"] for r in rois])
    lo, hi = allw.min(axis=0) - args.margin_mm, allw.max(axis=0) + args.margin_mm
    ref = (np.array([float(v) for v in args.ref_mm.split(",")]) if args.ref_mm
           else (lo + hi) / 2.0)
    ref_px = plane.to_pixel(np.array([ref]))[0]

    # Sample the region in *world* space and project each sample, rather than
    # taking the pixel bounding box of the projected corners. A rectangle on the
    # ground can have a corner beyond the plane's horizon, and its pixel bbox
    # then swallows the sky — where millimetres-per-pixel runs away and the
    # scale ladder fills up with bands that describe nothing real.
    gw = np.array([[x, y] for x in np.linspace(lo[0], hi[0], 41)
                   for y in np.linspace(lo[1], hi[1], 41)])
    gpx = plane.to_pixel(gw)
    keep = (plane.in_front(gpx, ref_px)
            & (gpx[:, 0] >= 0) & (gpx[:, 0] < cal["width"])
            & (gpx[:, 1] >= 0) & (gpx[:, 1] < cal["height"]))
    grid = gpx[keep]
    if len(grid) < 4:
        raise SystemExit(
            "almost none of the ROI region is visible on the car plane in this frame; "
            "check the ROIs, or cut --margin-mm"
        )
    region = (int(grid[:, 0].min()), int(grid[:, 1].min()),
              int(grid[:, 0].max()) + 1, int(grid[:, 1].max()) + 1)
    if region[2] - region[0] < 16 or region[3] - region[1] < 16:
        raise SystemExit(f"search region {region} is degenerate; check --margin-mm and the ROIs")

    template = cv2.imread(str(args.template), cv2.IMREAD_COLOR)
    if template is None:
        raise SystemExit(f"could not read template {args.template}")
    # Physical size of the marker itself, from the metric template — not from
    # the synthesised crop, whose pixel size is whatever the camera sees there.
    tpl_w_mm = template.shape[1] * args.template_mm_per_px
    tpl_h_mm = template.shape[0] * args.template_mm_per_px
    crop, ref_mm_per_px, yaw_ref, _quad_ref = synthesise_raw_template(
        template, args.template_mm_per_px, plane, ref, y_up)

    # Scale bands: what the plane predicts the crop needs across the region.
    scales = np.array([ref_mm_per_px / max(plane.mm_per_px(p), 1e-9) for p in grid])
    ok = np.isfinite(scales) & (scales > 1e-3)
    grid, scales = grid[ok], scales[ok]
    lo_s, hi_s = float(scales.min()), float(scales.max())
    shear = float(np.max([plane.anisotropy(p) for p in grid]))

    # Each band gets the strip of frame where the plane predicts that scale.
    # Scale is not a free parameter to be hunted: it is a known function of
    # *where* the match is, so a band only has to look where it applies.
    edges = (np.geomspace(lo_s, hi_s, max(2, math.ceil(math.log(hi_s / lo_s) / math.log(1.15)) + 1))
             if hi_s > lo_s * 1.001 else np.array([lo_s, hi_s * 1.001]))
    bands = []
    for a, b in itertools.pairwise(edges):
        mid = float(math.sqrt(a * b))
        pts = grid[(scales >= a - 1e-12) & (scales <= b + 1e-12)]
        if len(pts) == 0:
            continue
        pad = 0.5 * math.hypot(*crop.shape[:2]) * mid + 4
        bands.append((mid, (
            max(region[0], int(pts[:, 0].min() - pad)), max(region[1], int(pts[:, 1].min() - pad)),
            min(region[2], int(pts[:, 0].max() + pad) + 1),
            min(region[3], int(pts[:, 1].max() + pad) + 1))))
    if not bands:
        raise SystemExit("no usable scale band over the search region")
    if len(bands) > 16:
        raise SystemExit(
            f"the marker's size varies {hi_s / lo_s:.0f}x over this region, which needs "
            f"{len(bands)} scale bands — more search than bird's-eye matching would cost. "
            "Cut --margin-mm, or use pipeline1_bev.py."
        )

    det = RawHybridDetector(crop, bands, min_score=args.min_score)
    map1, map2 = undistort_maps(cal["K"], cal["D"], cal["model"], cal["width"], cal["height"])

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"could not open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    t_start = parse_time(args.start) or 0.0
    t_end = parse_time(args.end)
    if t_end is not None and t_end <= t_start:
        raise SystemExit(f"--end {args.end} is not after --start {args.start or 0}")
    det_offset = detector_offset(args.detector_offset, car)

    # Seek rather than decode-and-discard. At 4K a frame costs ~14 ms to decode,
    # so starting five minutes in would otherwise burn two minutes before the
    # first row. Where the seek actually lands is read back rather than assumed:
    # h264 seeks to a keyframe, and a frame index guessed from the request would
    # mislabel every row in the file with no way to notice.
    first = 0
    if t_start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t_start * fps))
        first = round(cap.get(cv2.CAP_PROP_POS_FRAMES))

    print("pipeline 2 — raw first", flush=True)
    print(f"  plane      {plane_note}", flush=True)
    print(f"  crop       {crop.shape[1]}x{crop.shape[0]} px synthesised at "
          f"({ref[0]:.0f}, {ref[1]:.0f}) mm, {ref_mm_per_px:.2f} mm/px there")
    print(f"  yaw const  {yaw_ref:+.2f}° (image +X -> world bearing at that pixel)", flush=True)
    print(f"  region     x {region[0]}-{region[2]}, y {region[1]}-{region[3]} px", flush=True)
    print(f"  scale      {len(bands)} band(s) over {lo_s:.2f}-{hi_s:.2f}; "
          f"worst shear {shear:.2f}")
    if shear > 1.25:
        print("  NOTE: the marker is squashed by more than a rotate-and-scale template can "
              "represent here; expect pipeline 1 to place it better.")
    if det_offset is not None:
        src = "car file" if args.detector_offset == USE_CAR_FILE else "command line"
        print(f"  det-offset ({det_offset[0]:+.1f}, {det_offset[1]:+.1f}) mm from the {src}, "
              f"{float(np.linalg.norm(det_offset)):.1f} mm, in the template's frame", flush=True)
    if t_start > 0 or t_end is not None:
        print(f"  clip       {t_start:.2f}s to "
              f"{'end' if t_end is None else f'{t_end:.2f}s'}"
              f"{f', sought to frame {first}' if first else ''}", flush=True)
    print(f"  rois       {', '.join(r['name'] for r in rois)}", flush=True)

    header = (["frame", "time_s", "found", "method", "score", "sigma_mm",
               "sticker_x_mm", "sticker_y_mm", "heading_deg"]
              + [f"box{i}_{ax}_mm" for i in range(1, 5) for ax in ("x", "y")]
              + ["sticker_height_mm", "pitch_deg", "roll_deg", "plane_fit"]
              + [f"stick{i}_{ax}_mm" for i in range(1, 5) for ax in ("x", "y")]
              + ([f"wheel_{w}_{ax}_mm" for w in ("fl", "fr", "rl", "rr") for ax in ("x", "y")]
                 if car.get("wheels_mm") else [])
              + [c for r in rois for c in (f"{r['name']}_mm", f"{r['name']}_hit")]
              + ["scale", "expected_scale", "scale_ratio"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    prior, hits, n, t0 = None, 0, 0, time.time()
    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        idx = first - 1
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            t = idx / fps
            if t_end is not None and t > t_end:
                break               # past the clip; nothing later can be wanted
            if idx % args.every or t < t_start:
                continue
            n += 1

            und = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
            pre = preprocess(und)
            got = det.detect(pre, prior, args.search_px if prior else None)
            if got is None and prior is not None:
                got, prior = det.detect(pre), None
            if got is None:
                w.writerow([idx, f"{t:.4f}", 0, "", "", ""] + [""] * (len(header) - 6))
                continue

            x, y, theta, score, sigma_px, band, method = got
            prior = (x, y, theta, band)
            # Two mappings, not one: the centre goes through the homography
            # directly (exact for a point on the plane), the heading through the
            # local bearing plus the constant tying the crop's axes to the
            # metric template's frame.
            centre_px = np.array([x, y])
            tpl_heading = norm_deg(plane.world_heading(centre_px, theta) + yaw_ref)
            fixed = polish_where_it_is(pre, plane, template, args.template_mm_per_px,
                                       x, y, theta, y_up)
            if fixed is not None:
                x, y, tpl_heading, cc = fixed
                centre_px = np.array([x, y])
                sigma_px, method = max(0.02, 0.5 * (1 - cc)), "hybrid:ecc@site"
            hits += 1
            centre = plane.to_world(centre_px[None, :])[0]
            if det_offset is not None:
                # Applied here, before the box, the quad, the wheels and the
                # reported position are built, so all five agree about where the
                # marker is instead of four of them agreeing with the template.
                ang = math.radians(tpl_heading)
                c_, s_ = math.cos(ang), math.sin(ang)
                centre = centre + np.array([[c_, -s_], [s_, c_]]) @ det_offset
            box, heading = car_box(car["body_mm"], centre, tpl_heading, car["yaw_offset_deg"])
            quad = sticker_quad(centre, tpl_heading, tpl_w_mm, tpl_h_mm)
            local_mm_px = plane.mm_per_px(centre_px)
            expected = ref_mm_per_px / max(local_mm_px, 1e-9)
            pitch = roll = None
            fit, Rm_used = "planar", None

            # Same detection, read two ways. The planar answer above is kept as
            # the fallback: the corner fit is the better model but the more
            # fragile one, and a frame it cannot converge on should degrade to
            # the old number rather than to no number at all.
            if args.corner_pnp and cal["R"] is not None:
                try:
                    crop_c, _mm, _yaw, quad_c = synthesise_raw_template(
                        template, args.template_mm_per_px, plane, centre, y_up)
                except SystemExit:
                    crop_c = None
                if crop_c is not None:
                    got_c = ecc_corners(pre, preprocess(crop_c), quad_c, x, y, theta)
                    if got_c is not None:
                        pose = marker_pose(cal["K"], cal["R"], cal["t"], got_c[0],
                                           tpl_w_mm, tpl_h_mm)
                        if pose is not None:
                            Rm, _tm = pose
                            box, tpl_h2 = car_box_from_tilt(
                                car["body_mm"], centre, Rm, car["sticker_height_mm"])
                            heading = norm_deg(tpl_h2 + car["yaw_offset_deg"])
                            pitch, roll = pose_tilt_deg(Rm)
                            fit, Rm_used = "corner-pnp", Rm

            row = [idx, f"{t:.4f}", 1, method, f"{score:.4f}", f"{sigma_px * local_mm_px:.3f}",
                   f"{centre[0]:.1f}", f"{centre[1]:.1f}", f"{heading:.3f}"]
            row += [f"{v:.1f}" for corner in box for v in corner]
            row += [f"{car['sticker_height_mm']:.1f}",
                    "" if pitch is None else f"{pitch:.3f}",
                    "" if roll is None else f"{roll:.3f}", fit]
            row += [f"{v:.1f}" for corner in quad for v in corner]
            if car.get("wheels_mm"):
                wpts = wheels_world(car["wheels_mm"], centre, tpl_heading, Rm_used,
                                    car["sticker_height_mm"])
                row += [f"{v:.1f}" for w in wpts for v in w]
            for r in rois:
                gap, hit = clearance_mm(box, r["world_mm"], r["closed"])
                row += [f"{gap:.1f}", int(hit)]
            row += [f"{band:.4f}", f"{expected:.4f}", f"{band / max(expected, 1e-9):.4f}"]
            w.writerow(row)

    cap.release()
    dt = time.time() - t0
    print(f"  {hits}/{n} frames detected ({hits / max(n, 1):.0%}) in {dt:.1f}s "
          f"({n / max(dt, 1e-9):.1f} fps) -> {args.out}")
    if n and hits < 0.25 * n:
        # The crop is only strictly right where it was cut, and the default cut
        # point is the middle of the ROI extent — which says nothing about where
        # the car stands. A few metres away the shear differs enough to put the
        # correlation under the score floor, and the run then reports a clean
        # zero rather than anything that points at the cause.
        print(f"  NOTE: the camera-frame crop was synthesised at "
              f"({ref[0]:.0f}, {ref[1]:.0f}) mm — the centre of the ROI extent. It only "
              "matches well near where it was cut. Re-run with --ref-mm X,Y set near "
              "where the car actually stands, or use pipeline1_bev.py, which does not "
              "care where the car is.", flush=True)


if __name__ == "__main__":
    main()
