"""Pipeline 1 — bird's-eye first: warp each frame to the car plane, then match.

    for each frame:
        warp the frame to a metric bird's-eye raster on the car plane
        hybrid-detect the sticker (rotation-bank NCC -> ECC refine)
        turn the sticker pose into the car's ground box
        measure that box against every ROI, in world millimetres

Why warp first: on the car-plane raster one millimetre is the same number of
pixels everywhere, so the sticker is the same size and shape wherever the car
is and rotation is the matcher's only free parameter. The price is a remap per
frame. Pipeline 2 makes the opposite trade.

Two planes are in play and mixing them is the mistake this file is arranged to
avoid. The *sticker* is read on the car plane (a horizontal plane at the roof
marker's height), which reports the ground position directly beneath it. The
*ROIs* are painted on the tarmac and are read on the field plane. They only
ever meet as world millimetres, never as pixels.

Standalone: numpy + opencv only. Nothing here imports dlms.

    python pipeline1_bev.py --video clip.mp4 --calibration calibration.json \
        --car car.json --template sticker.png --template-mm-per-px 9.41 \
        --rois rois.json --out track.csv
"""

from __future__ import annotations

import argparse
import csv
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
    """The station survey: intrinsics, the ground plane, and the camera pose.

    Only the fields the pipeline actually uses are pulled out. ``car`` is the
    plane the sticker is read on: preferred is one synthesised from the camera
    pose at *this vehicle's* marker height, which is exact for that car; the
    surveyed car plane is the fallback.
    """
    raw = json.loads(Path(path).read_text())
    intr = raw.get("intrinsics")
    if intr is None:
        raise SystemExit(
            f"{path} has no intrinsics. Lens distortion would be carried into every "
            "measurement, worst at the frame edges, so this pipeline refuses to guess."
        )
    K = np.array(intr["camera_matrix"], dtype=np.float64)
    D = np.array(intr["dist_coeffs"], dtype=np.float64).ravel()
    field = np.array(raw["field"]["homography"], dtype=np.float64)
    pose = (raw.get("field") or {}).get("pose")
    return {
        "K": K,
        "D": D,
        "model": str(intr.get("model", "pinhole")),
        "width": int(raw["image_width"]),
        "height": int(raw["image_height"]),
        "H_field": field,                      # undistorted px -> world mm, z = 0
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
    """Image -> world (X, Y) for the horizontal plane Z = height_mm.

    A point on that plane projects as K(R[:,0]X + R[:,1]Y + R[:,2]h + t), so
    inverting that 3x3 gives pixels back as the (X, Y) *directly below* a marker
    at that height — which is exactly what the car-box arithmetic wants.
    """
    _, up = camera_from_pose(R, t)
    Hw2i = K @ np.column_stack([R[:, 0], R[:, 1], R[:, 2] * (up * float(height_mm)) + t])
    H = np.linalg.inv(Hw2i)
    return H / H[2, 2]


def load_car(path: Path) -> dict:
    """Vehicle geometry, in the *sticker template* frame.

    Origin at the sticker centre, +X along the template's width axis. That is
    the template's frame, not the car's: ``sticker_yaw_offset_deg`` is the car's
    forward axis measured from template +X, and it is added only when reporting
    heading. A marker turned a quarter turn on the roof has an offset near ±90,
    and its stored polygon runs across +X accordingly.
    """
    raw = json.loads(Path(path).read_text())
    yaw = float(raw.get("sticker_yaw_offset_deg", 0.0))
    poly = raw.get("body_polygon_mm") or []
    if poly:
        body = np.array([[p["x_mm"], p["y_mm"]] for p in poly], dtype=np.float64)
    else:
        # No surveyed outline: build the rectangle from the scalar measurements,
        # which are car-frame distances, then rotate it into the template frame.
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
    return {
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
    ``distorted`` says the pixels were clicked on the raw frame rather than an
    undistorted one; they are straightened first, since the homography is only
    defined on undistorted pixels.
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
# geometry
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


def bev_matrices(x_min, y_min, mm_per_px, w_px, h_px, y_up=True):
    """World millimetres <-> raster pixels. With ``y_up``, rows increase as +Y decreases.

    This has to match the convention the template was cut with, which is why it
    is a parameter rather than a constant: a world frame whose +Z runs into the
    ground is seen mirrored from above (see :func:`camera_from_pose`), and a
    template cut one way will not correlate with a raster built the other way at
    any rotation. Both ends take it from the camera, so both agree.
    """
    y_max = y_min + h_px * mm_per_px
    w2b = (np.array(
        [[1.0 / mm_per_px, 0.0, -x_min / mm_per_px],
         [0.0, -1.0 / mm_per_px, y_max / mm_per_px],
         [0.0, 0.0, 1.0]], dtype=np.float64)
        if y_up else np.array(
        [[1.0 / mm_per_px, 0.0, -x_min / mm_per_px],
         [0.0, 1.0 / mm_per_px, -y_min / mm_per_px],
         [0.0, 0.0, 1.0]], dtype=np.float64))
    return w2b, np.linalg.inv(w2b)


def car_box(body_template_mm, center_mm, template_heading_deg, yaw_offset_deg):
    """Sticker pose -> the car's four ground corners, in world millimetres.

    The stored polygon is rotated by the angle the *template* was found at, not
    by the car's heading; the offset between them is applied only to the heading
    that gets reported. Reduced to four corners because that is the box every
    downstream number is measured against.
    """
    a = math.radians(template_heading_deg)
    c, s = math.cos(a), math.sin(a)
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    world = np.asarray(body_template_mm, dtype=np.float64) @ R.T + np.asarray(center_mm).reshape(2)
    if len(world) != 4:
        box = cv2.boxPoints(cv2.minAreaRect(world.astype(np.float32)))
        world = np.array(box, dtype=np.float64)
    return world, norm_deg(template_heading_deg + yaw_offset_deg)


#: Short-side pixels below which the eight-parameter corner fit stops being
#: trustworthy. Measured, not guessed: at 36 px a level marker read as 1.5 deg
#: tilted, while at 63 px the same scene read within 0.2 deg and tracked a real
#: tilt to better than 0.2 deg. Scale is what the fit is short of, so the number
#: is in pixels and not millimetres.
CORNER_PNP_MIN_PX = 60


def marker_pose(K, R, t, corners_px, width_mm, height_mm):
    """Marker -> world rotation and translation, from the four corners in pixels.

    The planar read gives a centre and a heading, and takes the marker's height
    on trust. Four corners over-determine that: solved as a pose they give the
    tilt as well, which is the whole point — a braking car dips its roof while
    its footprint stays put, and a rigid 2D transform charges the dip to the
    car's position.

    IPPE is used for the same reason the survey uses it, and with the same
    caveat: a planar target has two poses that reproject almost equally well,
    so both are asked for and the better one kept. On a marker this small the
    two are far apart, and picking the wrong one is obvious rather than subtle.
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
    # marker -> camera, then camera -> world through the station pose
    return R.T @ Rm, R.T @ (tvec.ravel() - t)


def car_box_from_tilt(body_template_mm, centre_mm, Rm, sticker_height_mm):
    """Car footprint and template heading, in world millimetres.

    Position comes from the planar read and orientation from the corner fit,
    because each is the better measurement of its own quantity. Pinning the
    marker to its surveyed height is strong information, and a pose solved from
    four corners alone throws it away: tilt and depth trade off against each
    other along the viewing ray, so the solved centre drifts by tens of
    millimetres while the planar read holds a few. Measured here, the full pose
    put the footprint out by 77 mm where the planar centre held 6 mm.

    What the planar read cannot see is the tilt, and tilt is exactly what moves
    the footprint out from under a marker no longer sitting square above it.
    So the offset from marker to footprint — one sticker height down, in the
    marker's own frame — is rotated by the measured attitude, and hung off the
    planar centre.

    With the marker level this reduces to :func:`car_box`, which is why the two
    can be swapped by a flag with nothing else changing.
    """
    body = np.asarray(body_template_mm, dtype=np.float64)
    pts = np.column_stack([body, np.full(len(body), -float(sticker_height_mm))])
    xy = (pts @ np.asarray(Rm, dtype=np.float64).T)[:, :2] + np.asarray(centre_mm).reshape(2)
    if len(xy) != 4:
        xy = np.array(cv2.boxPoints(cv2.minAreaRect(xy.astype(np.float32))), dtype=np.float64)
    fwd = np.asarray(Rm)[:, 0]                       # template +X, in world
    return xy, math.degrees(math.atan2(fwd[1], fwd[0]))


def pose_tilt_deg(Rm):
    """Pitch and roll of the marker's plane, in degrees, for the record.

    Zero means the marker is horizontal, which is what the planar read assumes
    everywhere. These are written to the CSV so a run can be judged rather than
    trusted: a car that never appears to tilt is a car whose corners are not
    actually being fitted.
    """
    Rm = np.asarray(Rm, dtype=np.float64)
    pitch = math.degrees(math.asin(float(np.clip(-Rm[2, 0], -1.0, 1.0))))
    roll = math.degrees(math.asin(float(np.clip(Rm[2, 1], -1.0, 1.0))))
    return pitch, roll


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


def clearance_mm(box, roi_pts, closed) -> tuple[float, bool]:
    """Gap between the car box and one ROI, in millimetres.

    Zero when they cross, negative when the ROI lies wholly under the car, and
    otherwise the shortest distance between the two outlines. The boolean is the
    thing a rule actually asks — "is the car on it?" — kept separate so a
    clearance of 0.0 is never confused with a near miss rounded down.
    """
    roi = np.asarray(roi_pts, dtype=np.float64).reshape(-1, 2)
    edges = [(box[i], box[(i + 1) % 4]) for i in range(4)]
    roi_segs = []
    if len(roi) >= 2:
        n = len(roi) if closed else len(roi) - 1
        roi_segs = [(roi[i], roi[(i + 1) % len(roi)]) for i in range(n)]

    for a, b in edges:
        for c, d in roi_segs:
            if _segments_cross(a, b, c, d):
                return 0.0, True

    best = min(_seg_dist(p, a, b) for p in roi for a, b in edges)
    if roi_segs:
        best = min(best, min(_seg_dist(q, c, d) for q in box for c, d in roi_segs))
    all_in = all(_inside(box, p) for p in roi)
    return (-best if all_in else best), all_in


# --------------------------------------------------------------------------
# detection
# --------------------------------------------------------------------------


def preprocess(img: np.ndarray) -> np.ndarray:
    """clahe+unsharp — the recipe that wins on real outdoor footage.

    Applied identically to the template and to every frame. Matching a CLAHE'd
    frame against a raw template quietly destroys accuracy, so there is exactly
    one of these functions and both paths go through it. The CLAHE grid is
    derived from a tile size in *pixels* so an 88 px template and a multi-
    megapixel raster get the same operator rather than the same tile count.
    """
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = g.shape[:2]
    grid = (max(1, round(w / 192)), max(1, round(h / 192)))
    g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=grid).apply(g)
    return cv2.addWeighted(g, 2.2, cv2.GaussianBlur(g, (0, 0), 2.0), -1.2, 0)


def circular_mask(shape) -> np.ndarray:
    """Rotation-invariant support.

    A square template carries triangular padding once rotated, and that padding's
    hard edge biases the correlation score by angle — the classic failure of a
    naive rotation bank. An inscribed circle compares the same pixels at every
    trial angle.
    """
    h, w = shape
    m = np.zeros((h, w), np.uint8)
    cv2.circle(m, (w // 2, h // 2), max(int(min(h, w) * 0.5) - 1, 1), 255, -1)
    return m


def subpixel_peak(resp, x, y) -> tuple[float, float]:
    """Quadratic interpolation of the correlation peak: ~10x over integer argmax."""
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


class HybridDetector:
    """Rotation-bank NCC to find it, ECC to place it.

    One method cannot have both: NCC finds the marker anywhere but is quantised
    to its correlation grid, while ECC is sub-pixel but only converges from a
    nearby start. Chaining them gets both, and a refinement that jumps further
    than it plausibly could has locked onto something else and is discarded.
    """

    def __init__(self, template, coarse_step=12.0, fine_step=2.0, min_score=0.45):
        self.tpl = preprocess(template)
        self.mask = circular_mask(self.tpl.shape[:2])
        self.coarse_step, self.fine_step, self.min_score = coarse_step, fine_step, min_score
        self.radius = 0.5 * math.hypot(*self.tpl.shape[:2])
        side = min(self.tpl.shape[:2])
        self.pyr = 1 << math.floor(math.log2(max(1, min(8, side // 12))))
        self._bank: dict[tuple[float, int], tuple[np.ndarray, np.ndarray]] = {}

    def _rotated(self, angle, scale):
        key = (round(angle, 3), scale)
        if key not in self._bank:
            tpl, mask = self.tpl, self.mask
            if scale > 1:
                size = (max(4, tpl.shape[1] // scale), max(4, tpl.shape[0] // scale))
                tpl = cv2.resize(tpl, size, interpolation=cv2.INTER_AREA)
                mask = circular_mask(tpl.shape[:2])
            h, w = tpl.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2 - 0.5, h / 2 - 0.5), angle, 1.0)
            rot = cv2.warpAffine(cv2.bitwise_and(tpl, tpl, mask=mask), M, (w, h), cv2.INTER_CUBIC)
            self._bank[key] = (rot, mask)
        return self._bank[key]

    def _best_at(self, win, angle, scale):
        tpl, mask = self._rotated(angle, scale)
        th, tw = tpl.shape[:2]
        if win.shape[0] < th or win.shape[1] < tw:
            return None
        resp = cv2.matchTemplate(win, tpl, cv2.TM_CCOEFF_NORMED, mask=mask)
        resp = np.nan_to_num(resp, nan=-1.0, posinf=-1.0, neginf=-1.0)
        _, score, _, loc = cv2.minMaxLoc(resp)
        px, py = subpixel_peak(resp, loc[0], loc[1])
        return float(score), float(angle), px + tw / 2.0, py + th / 2.0

    def _sweep(self, win, angles, scale):
        hits = [h for h in (self._best_at(win, float(a), scale) for a in angles) if h]
        hits.sort(reverse=True)
        return hits

    def _patch(self, frame, x, y, theta):
        """De-rotated crop around a candidate, and the template padded to match.

        Both refinements start here: the Euclidean one that places the marker
        and the homography one that lets it tilt. Returning the affine ``M``
        too is what lets a fitted corner be carried back to frame pixels.
        """
        th, tw = self.tpl.shape[:2]
        # Keep the padded canvas the same parity as the template, or the template
        # lands half a pixel off centre and that becomes a fixed positional bias.
        ph, pw = int(th * 1.35) // 2 * 2 + th % 2, int(tw * 1.35) // 2 * 2 + tw % 2
        a = math.radians(theta)
        c, s = math.cos(a), math.sin(a)
        M = np.array([[c, s, x - (c * (pw / 2 - 0.5) + s * (ph / 2 - 0.5))],
                      [-s, c, y - (-s * (pw / 2 - 0.5) + c * (ph / 2 - 0.5))]], dtype=np.float64)
        patch = cv2.warpAffine(frame, M, (pw, ph),
                               flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP, borderValue=0)
        if patch.size == 0 or float(patch.std()) < 1e-3:
            return None
        canvas = np.zeros((ph, pw), self.tpl.dtype)
        y0, x0 = (ph - th) // 2, (pw - tw) // 2
        canvas[y0:y0 + th, x0:x0 + tw] = self.tpl
        m = np.zeros((ph, pw), np.uint8)
        m[y0:y0 + th, x0:x0 + tw] = 255
        return patch, canvas, m, M, (ph, pw, th, tw, y0, x0), (c, s)

    def _ecc(self, frame, x, y, theta):
        """Sub-pixel Euclidean refinement; this is where the millimetres are won."""
        got = self._patch(frame, x, y, theta)
        if got is None:
            return None
        patch, canvas, m, _M, geom, (c, s) = got
        ph, pw = geom[0], geom[1]
        try:
            cc, warp = cv2.findTransformECC(
                canvas.astype(np.float32), patch.astype(np.float32),
                np.eye(2, 3, dtype=np.float32), cv2.MOTION_EUCLIDEAN,
                (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-7), m, 5)
        except cv2.error:
            return None                      # ECC raises when it cannot converge
        if not np.isfinite(warp).all():
            return None
        T = np.array([[1.0, 0, pw / 2 - 0.5], [0, 1.0, ph / 2 - 0.5], [0, 0, 1.0]])
        prior = np.array([[c, s, x], [-s, c, y], [0, 0, 1.0]])
        total = prior @ np.linalg.inv(T) @ np.vstack([warp, [0, 0, 1.0]]) @ T
        return (float(cc), float(total[0, 2]), float(total[1, 2]),
                norm_deg(math.degrees(math.atan2(-total[1, 0], total[0, 0]))))

    def corner_quad(self, frame, x, y, theta):
        """The marker's four corners in raster pixels, refit with tilt allowed.

        ``detect`` fits three numbers — two of position and one of rotation —
        which describes a marker lying flat at exactly the surveyed height. A
        marker on a braking car does not lie flat: the roof dips, and its image
        stops being a rectangle. Those three numbers cannot represent that, so
        the dip is absorbed into position instead, which is precisely the error
        a pose solve exists to remove.

        Refitting the same patch with all eight parameters lets the corners
        record the tilt. It is seeded from the Euclidean fit because ECC only
        converges from a good start, and eight parameters from a cold start on
        a small marker will wander.

        Returns the corners in the template's own order — the one
        :func:`sticker_quad` uses — or None if the fit does not converge.
        """
        got = self._patch(frame, x, y, theta)
        if got is None:
            return None
        patch, canvas, m, M, geom, _cs = got
        th, tw, y0, x0 = geom[2:]
        try:
            _cc, warp = cv2.findTransformECC(
                canvas.astype(np.float32), patch.astype(np.float32),
                np.eye(2, 3, dtype=np.float32), cv2.MOTION_EUCLIDEAN,
                (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-7), m, 5)
            seed = np.vstack([warp, [0, 0, 1]]).astype(np.float32)
            cc, warp = cv2.findTransformECC(
                canvas.astype(np.float32), patch.astype(np.float32),
                seed, cv2.MOTION_HOMOGRAPHY,
                (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-7), m, 5)
        except cv2.error:
            return None                      # ECC raises when it cannot converge
        if not np.isfinite(warp).all():
            return None

        # Template corners in canvas pixels, ordered to match sticker_quad:
        # the raster's Y runs opposite to the world's, so the template's first
        # world corner (-w/2, -h/2) is its BOTTOM-left pixel, not its top-left.
        corners = np.array([[x0, y0 + th - 1], [x0 + tw - 1, y0 + th - 1],
                            [x0 + tw - 1, y0], [x0, y0]], dtype=np.float64)
        # canvas -> patch (the fitted warp) -> frame (the crop's own affine)
        q = np.column_stack([corners, np.ones(4)]) @ np.asarray(warp, dtype=np.float64).T
        q = q[:, :2] / q[:, 2:]
        out = np.column_stack([q, np.ones(4)]) @ M.T
        return (out, float(cc)) if np.isfinite(out).all() else None

    def detect(self, bev, prior=None, search_px=None):
        """Returns (x, y, theta_deg, score, sigma_px, method) in raster pixels.

        The raster is preprocessed once and windowed afterwards, never the other
        way round: CLAHE is spatially adaptive, so equalising a crop is a
        different operator from equalising the whole frame and cropping.
        """
        pre = frame = preprocess(bev)
        ox = oy = 0
        angles = list(np.arange(0.0, 360.0, self.coarse_step))
        scale = self.pyr
        if prior is not None and search_px:
            # A known previous pose makes the search local, which is both faster
            # and less likely to lock onto something else in a big raster.
            r = search_px + self.radius
            x0, y0 = max(0, int(prior[0] - r)), max(0, int(prior[1] - r))
            x1, y1 = min(frame.shape[1], int(prior[0] + r) + 1), min(frame.shape[0], int(prior[1] + r) + 1)
            frame, ox, oy = frame[y0:y1, x0:x1], x0, y0
            # Centred on the prior, so the angle it was last found at is
            # actually sampled. An arange from theta-30 lands on theta+/-6
            # and never on theta itself, which costs real correlation score
            # on a marker that barely rotates between frames.
            angles = [prior[2] + k * self.coarse_step for k in range(-2, 3)]
            scale = 1
        if frame.size == 0:
            return None

        if scale > 1:
            small = cv2.resize(frame, (max(8, frame.shape[1] // scale), max(8, frame.shape[0] // scale)),
                               interpolation=cv2.INTER_AREA)
            coarse = self._sweep(small, angles, scale)
            if not coarse:
                return None
            _, best_a, cx, cy = coarse[0]
            cx, cy = cx * scale, cy * scale
            r = self.radius + 3.0 * scale
            sx, sy = max(0, int(cx - r)), max(0, int(cy - r))
            sub = frame[sy:min(frame.shape[0], int(cy + r) + 1), sx:min(frame.shape[1], int(cx + r) + 1)]
            fine = self._sweep(sub, np.arange(best_a - self.coarse_step, best_a + self.coarse_step + 1e-6,
                                              self.fine_step), 1)
            if not fine:
                return None
            score, best_a, rx, ry = fine[0]
            cx, cy = rx + sx, ry + sy
        else:
            hits = self._sweep(frame, angles, 1)
            if not hits:
                return None
            score, best_a, cx, cy = hits[0]
            fine = self._sweep(frame, np.arange(best_a - self.coarse_step, best_a + self.coarse_step + 1e-6,
                                                self.fine_step), 1)
            if fine:
                score, best_a, cx, cy = fine[0]

        if score < self.min_score:
            return None
        x, y, theta, sigma, method = cx + ox, cy + oy, norm_deg(best_a), 0.5, "ncc"
        got = self._ecc(pre, x, y, theta)
        if got is not None:
            cc, rx, ry, rtheta = got
            if math.hypot(rx - x, ry - y) <= 6.0 and abs(norm_deg(rtheta - theta)) <= 8.0:
                x, y, theta, sigma, method = rx, ry, rtheta, max(0.02, 0.5 * (1 - cc)), "hybrid:ecc"
        return x, y, theta, score, sigma, method


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
    ap.add_argument("--mm-per-px", type=float, default=None, help="Raster scale (default: the template's).")
    ap.add_argument("--margin-mm", type=float, default=5000.0, help="Raster margin around the ROIs.")
    ap.add_argument("--roi-distorted", action="store_true", help="ROI pixels were clicked on the raw frame.")
    ap.add_argument("--every", type=int, default=1, help="Process every Nth frame.")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--min-score", type=float, default=0.45)
    ap.add_argument("--search-px", type=float, default=140.0, help="Local search radius once tracking.")
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

    # Raster extent: the ROIs plus a margin. Sizing it from the whole frame
    # instead would run to the horizon and cost tens of millions of pixels a frame.
    allw = np.vstack([r["world_mm"] for r in rois])
    x_min, y_min = allw.min(axis=0) - args.margin_mm
    x_max, y_max = allw.max(axis=0) + args.margin_mm
    mmpp = args.mm_per_px or args.template_mm_per_px
    w_px, h_px = int(np.ceil((x_max - x_min) / mmpp)), int(np.ceil((y_max - y_min) / mmpp))
    if w_px * h_px > 40_000_000:
        raise SystemExit(f"raster would be {w_px}x{h_px} px; raise --mm-per-px or cut --margin-mm")
    # Same handedness the template was cut with. Without a pose we cannot tell,
    # and the ordinary map convention is the right guess.
    y_up = True if cal["R"] is None else camera_from_pose(cal["R"], cal["t"])[1] > 0
    if not y_up:
        print("  world frame is left-handed — raster mirrored to match the camera, "
              "and the template", flush=True)
    w2b, b2w = bev_matrices(x_min, y_min, mmpp, w_px, h_px, y_up=y_up)

    template = cv2.imread(str(args.template), cv2.IMREAD_COLOR)
    if template is None:
        raise SystemExit(f"could not read template {args.template}")
    # Physical size, taken before any rescale: the marker is the size it is,
    # whatever resolution the raster happens to sample it at.
    tpl_w_mm = template.shape[1] * args.template_mm_per_px
    tpl_h_mm = template.shape[0] * args.template_mm_per_px
    if abs(args.template_mm_per_px - mmpp) > 1e-9:
        # The template's pixel size is physical; rescale it to the raster's.
        f = args.template_mm_per_px / mmpp
        template = cv2.resize(template, (max(4, round(template.shape[1] * f)),
                                         max(4, round(template.shape[0] * f))),
                              interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC)
    det = HybridDetector(template, min_score=args.min_score)

    # Remap tables: undistort and warp to the car plane in one pass, built once.
    map1, map2 = undistort_maps(cal["K"], cal["D"], cal["model"], cal["width"], cal["height"])
    Hpix2bev = w2b @ H_car
    ys, xs = np.mgrid[0:h_px, 0:w_px].astype(np.float32)
    src = apply_h(np.linalg.inv(Hpix2bev), np.stack([xs.ravel(), ys.ravel()], 1)).astype(np.float32)
    wmap1, wmap2 = cv2.convertMaps(src[:, 0].reshape(h_px, w_px), src[:, 1].reshape(h_px, w_px), cv2.CV_16SC2)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"could not open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    print("pipeline 1 — bird's-eye first", flush=True)
    print(f"  plane      {plane_note}", flush=True)
    print(f"  raster     {w_px}x{h_px} px @ {mmpp:g} mm/px, origin ({x_min:.0f}, {y_min:.0f}) mm", flush=True)
    print(f"  template   {template.shape[1]}x{template.shape[0]} px", flush=True)
    if args.corner_pnp:
        short = min(template.shape[:2])
        print(f"  corner-pnp on, marker {short} px on its short side", flush=True)
        if short < CORNER_PNP_MIN_PX:
            print(f"  NOTE: under {CORNER_PNP_MIN_PX} px the corner fit is biased, not just "
                  f"noisy — it read a level marker as tilted by more than a degree in "
                  f"testing, which moves the footprint the wrong way. Either raise "
                  f"--mm-per-px, or use a bigger marker, or leave --corner-pnp off.",
                  flush=True)
    print(f"  rois       {', '.join(r['name'] for r in rois)}", flush=True)

    header = (["frame", "time_s", "found", "method", "score", "sigma_mm",
               "sticker_x_mm", "sticker_y_mm", "heading_deg"]
              + [f"box{i}_{ax}_mm" for i in range(1, 5) for ax in ("x", "y")]
              + ["sticker_height_mm", "pitch_deg", "roll_deg", "plane_fit"]
              + [f"stick{i}_{ax}_mm" for i in range(1, 5) for ax in ("x", "y")]
              + ([f"wheel_{w}_{ax}_mm" for w in ("fl", "fr", "rl", "rr") for ax in ("x", "y")]
                 if car.get("wheels_mm") else [])
              + [c for r in rois for c in (f"{r['name']}_mm", f"{r['name']}_hit")])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    prior, hits, n, t0 = None, 0, 0, time.time()
    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        idx = -1
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            t = idx / fps
            if idx % args.every or t < args.start or (args.end is not None and t > args.end):
                continue
            n += 1

            und = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
            bev = cv2.remap(und, wmap1, wmap2, cv2.INTER_LINEAR, borderValue=0)
            got = det.detect(bev, prior, args.search_px if prior else None)
            if got is None and prior is not None:
                got, prior = det.detect(bev), None      # lost it; fall back to a full search
            if got is None:
                w.writerow([idx, f"{t:.4f}", 0, "", "", ""] + [""] * (len(header) - 6))
                continue

            x, y, theta, score, sigma_px, method = got
            prior = (x, y, theta)
            hits += 1
            centre = apply_h(b2w, np.array([[x, y]]))[0]
            tip = apply_h(b2w, np.array([[x + math.cos(math.radians(theta)) * 8,
                                          y - math.sin(math.radians(theta)) * 8]]))[0]
            # The raster flips Y, so read the heading off two mapped points
            # rather than trusting the angle's sign through the flip.
            tpl_heading = math.degrees(math.atan2(*(tip - centre)[::-1]))
            box, heading = car_box(car["body_mm"], centre, tpl_heading, car["yaw_offset_deg"])
            quad = sticker_quad(centre, tpl_heading, tpl_w_mm, tpl_h_mm)
            pitch = roll = None
            fit, Rm_used = "planar", None

            # Same detection, read two ways. The planar answer above is kept as
            # the fallback: the corner fit is the better model but the more
            # fragile one, and a frame it cannot converge on should degrade to
            # the old number rather than to no number at all.
            if args.corner_pnp and cal["R"] is not None:
                got_c = det.corner_quad(bev, x, y, theta)
                if got_c is not None:
                    quad_bev, _cc = got_c
                    quad_px = apply_h(np.linalg.inv(Hpix2bev), quad_bev)
                    pose = marker_pose(cal["K"], cal["R"], cal["t"], quad_px,
                                       tpl_w_mm, tpl_h_mm)
                    if pose is not None:
                        Rm, _tm = pose
                        box, tpl_h2 = car_box_from_tilt(
                            car["body_mm"], centre, Rm, car["sticker_height_mm"])
                        heading = norm_deg(tpl_h2 + car["yaw_offset_deg"])
                        quad = apply_h(b2w, quad_bev)
                        pitch, roll = pose_tilt_deg(Rm)
                        fit, Rm_used = "corner-pnp", Rm

            row = [idx, f"{t:.4f}", 1, method, f"{score:.4f}", f"{sigma_px * mmpp:.3f}",
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
            w.writerow(row)

    cap.release()
    dt = time.time() - t0
    print(f"  {hits}/{n} frames detected ({hits / max(n, 1):.0%}) in {dt:.1f}s "
          f"({n / max(dt, 1e-9):.1f} fps) -> {args.out}")


if __name__ == "__main__":
    main()
