#!/usr/bin/env python3
"""Refit a borrowed lens profile's focal length from the car's own roof marker.

`nudge_intrinsics.py` re-solves the principal point when one camera wears
another's profile, and deliberately freezes `fx`, `fy`: focal length is a
property of the part number, and the same part number should mean the same
number. On two real units of one model it did not. Their profiles sat 30 px
apart in `f` -- 21 px of lens mapping at the edge of the working field, 44 mm of
position error on ROIs 3.5 m outside the surveyed square, and 129 mm on a
reported clearance. That is three times what two board sessions of one camera
disagree by, so it is the unit and not the shoot.

The survey cannot repair it. Eight control points in a 1.8 x 2.2 m square buy
the pose first; the pose then absorbs a wrong principal point almost entirely
(17 px of it cost under 7 mm), and what is left over is monotone in `f` with no
minimum at all -- the cost keeps falling towards a focal length that is plainly
wrong. Straightness cannot find it either, because a scale error leaves every
straight line straight.

**The car carries what the ground does not.** Its roof marker is a printed
rectangle of fixed shape on a plane of known height, and the car takes it across
the whole working area on every run. Read its four corners through a profile
whose `f` is too small and the rectangle *grows* as the car drives away::

    f too small    the marker reads 2.4 % larger at 8.9 m than at 1.6 m
    f right        the same size wherever the car is

so `fx` and `fy` are solved to flatten that trend, with the marker's true size
carried along as a free nuisance parameter. The free size is what makes this
safe on a template nobody measured with a ruler: a constant scale error is the
same at every distance and lands entirely on the nuisance, and only the *trend*
with distance can move `f`. Nothing is surveyed, nothing is re-shot, no tape
goes on the ground.

Measured at B7 borrowing between two units, scored on 601 frames of a drive-out
the fit never saw:

    lens mapping at 33 deg      21.1 -> 10.1 px
    ROI position, far           44.5 -> 11.1 mm
    ROI clearance p95 / max    128.6 / 148.4 -> 48.0 / 68.0 mm
    car box p95                125.1 -> 79.7 mm

About 50 mm of total expected error on a reported clearance, which is what two
board calibrations of one camera disagree by -- the borrowed profile ended up
closer to the station's own than that station's second board session was.

    python src/poc/focal_from_marker.py --calibration $S/calibration.json \\
        --track $S/detect.csv --template $S/sticker.png \\
        --out intrinsics/B7_marker.json \\
        --out-calibration $S/calibration_f.json \\
        --roi $S/roi.json --out-roi $S/roi_f.json

**What it needs.** A track written with `--corner-pnp`, which refits the marker
patch with all eight parameters and so *measures* where the corners are instead
of stamping the template's own shape back down. A clip where the car actually
travels: the trend is the whole signal, and frames bunched at one distance carry
none of it -- 10 frames spread over 1.6 to 8.9 m left 10 mm at the far ROIs
where 68 frames bunched inside 1.3 to 1.6 m left 133 mm, which is worse than
nothing. And `gcp --sticker-height-mm 1450 --pose-from all`, the same two-height
survey the nudge insists on, because four coplanar marks fit exactly at any `f`
by moving the camera to the height that `f` implies -- delete B7's marker-plane
controls and the same fit moves `f` 30 px the other way and leaves the marker
still growing.

**What it cannot do.** It moves `f` and nothing else -- `c` and the distortion
terms stay as the board fitted them. And `f` is a *frame* change, like the
nudge: `P = K` projects the undistorted image, so scaling `fx` slides every
undistorted pixel towards or away from `(cx, cy)`. The control points come
across because their sensor pixels are recoverable, and `--out-roi` carries an
ROI file the same way; the cut template and the traced outline cannot, and have
to be re-made with `sticker` and `outline` against the new calibration.

Standalone: numpy + opencv + the standard library, plus `calibrate.py` and
`nudge_intrinsics.py` beside it.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import calibrate as cal_
import cv2
import nudge_intrinsics as nudge_
import numpy as np

#: Frames are drawn band by band and not evenly along the clip. Coverage is what
#: identifies f, and a clip spends most of its frames wherever the car was slow.
BANDS_M = ((0.0, 2.0), (2.0, 4.0), (4.0, 6.0), (6.0, 1e9))

#: A marker reading this far off the median size is a detection that came apart,
#: not a lens: 3 % is 30 mm on a metre of marker, far more than a profile bends.
SIZE_OUTLIER = 0.03

#: The fit is pulled weakly back to the profile it started from, in units of
#: 50 px of f. A borrowed profile is already nearly right, and this costs about a
#: pixel while stopping a thin band of frames from dragging it somewhere silly.
PRIOR_WEIGHT, PRIOR_STEP_PX = 0.5, 50.0

#: Past this the marker is in the corner of the frame, a dozen pixels across and
#: fighting the steepest part of the distortion curve.
MAX_FIELD_DEG = 40.0

#: How far out a frame has to be to count as lever arm, and how many of those
#: the fit refuses to work without. See `--min-far-frames` for why it is a gate.
FAR_M, MIN_FAR = 4.0, 8

#: The field angle the lens gap is reported at: B7's ROIs and car sit inside it.
GAP_DEG = 33.0


# --------------------------------------------------------------------------
# reading what is already on disk
# --------------------------------------------------------------------------


def template_rect(png: Path, mm_per_px: float | None) -> tuple[float, float, np.ndarray]:
    """The rectangle the marker is believed to be, from the template `sticker` cut.

    Only the *shape* of this matters to the fit -- the size is re-scaled by the
    nuisance parameter -- but it is quoted in millimetres because the residual
    is, and because a size that comes back far from 1.0 is worth seeing.
    """
    img = cv2.imread(str(png))
    if img is None:
        raise SystemExit(f"cannot read the template {png}")
    if mm_per_px is None:
        meta = png.with_suffix(".json")
        if not meta.exists():
            raise SystemExit(f"pass --template-mm-per-px, or put the `sticker` step's "
                             f"{meta.name} beside the template")
        mm_per_px = float(json.loads(meta.read_text())["mm_per_px"])
    w, h = img.shape[1] * mm_per_px, img.shape[0] * mm_per_px
    known = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
    return w, h, known


def read_corners(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """The measured marker corners a `--corner-pnp` track carries, as world millimetres.

    Rows written without `--corner-pnp` are skipped rather than used: without it
    the pipeline puts the template's own rectangle back down at the pose it
    found, so its corners would agree with the template by construction and have
    nothing to say about the lens.
    """
    quads, tilt, heights, frames = [], [], set(), []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if r.get("found") != "1" or r.get("plane_fit") != "corner-pnp":
                continue
            quads.append([[float(r[f"stick{k}_x_mm"]), float(r[f"stick{k}_y_mm"])]
                          for k in (1, 2, 3, 4)])
            tilt.append(max(abs(float(r["pitch_deg"] or 0.0)), abs(float(r["roll_deg"] or 0.0))))
            heights.add(round(float(r["sticker_height_mm"]), 3))
            frames.append(int(r["frame"]))
    if len(quads) < 8:
        raise SystemExit(
            f"{path} has {len(quads)} corner-pnp rows and this fit wants tens of them. Re-run "
            "the pipeline with --corner-pnp on a clip where the car crosses the scene; without "
            "it the marker's corners are the template's own and carry nothing about the lens.")
    if len(heights) != 1:
        raise SystemExit(f"{path} mixes marker heights {sorted(heights)}; one clip, one car.")
    return (np.array(frames), np.array(quads, dtype=np.float64), np.array(tilt),
            float(next(iter(heights))))


def read_controls(cal: dict):
    """The control points the `gcp` step surveyed, and the refusal when they are all on the ground.

    This step needs the same two-height survey the nudge does, for a different
    reason. Four coplanar marks are eight equations against a six-parameter pose,
    so they fit *exactly* at any focal length -- by putting the camera at the
    height that focal length implies. The marker's plane is then raised through
    that pose, so its parallax scales with the height, and `f` and the height
    trade off almost exactly: the marker keeps on growing whatever `f` is set to,
    and there is nothing for the fit to flatten.

    That is measured, not argued. B7's own fit, re-run with the marker-plane
    controls deleted and nothing else changed, moved `f` by -30 px instead of
    +51, left the marker growing +0.79 % instead of -0.39 %, and fitted its
    shape half as well. Points at a second height are what pin the camera's
    height, and pinning the height is what makes `f` visible.
    """
    field = cal.get("field") or {}
    gcps = field.get("gcps") or []
    if len(gcps) < 4:
        raise SystemExit(f"this calibration has {len(gcps)} ground control point(s); the pose "
                         "alone needs 4. Re-run the `gcp` step.")
    sp = cal.get("sticker_plane") or {}
    controls = sp.get("controls") or []
    if len(controls) < 2:
        raise SystemExit(
            "this survey is all on the ground, and the marker cannot find a focal length over "
            "one plane: four coplanar marks fit exactly at any f, by moving the camera to the "
            "height that f implies, and the marker's plane is raised through that same pose. "
            "Re-run\n"
            "    calibrate.py gcp --image ... --sticker-height-mm 1450 --pose-from all\n"
            "to survey targets on the marker's plane as well -- 4 of them, which with the 4 "
            "ground marks is the 8 points this step wants.")
    if not sp.get("in_pose"):
        raise SystemExit(
            "this survey has targets on the marker's plane but did not put them in the pose, so "
            "the camera's height still comes from the ground alone and f has nothing to stand "
            "against. Re-run the `gcp` step with --pose-from all.")

    def xy(rows):
        return (np.array([[r["pixel"]["x"], r["pixel"]["y"]] for r in rows], dtype=np.float64),
                np.array([[r["world"]["x_mm"], r["world"]["y_mm"]] for r in rows],
                         dtype=np.float64))

    g_px, g_world = xy(gcps)
    s_px, s_world = xy(controls)
    return g_px, g_world, s_px, s_world, float(sp["height_mm"])


class Survey:
    """A station's control points as sensor pixels, and the pose any profile gives them.

    Every pixel in a calibration file is stored in the undistorted frame of the
    profile it was clicked through, and this fit changes that frame -- so the
    clicks go back to the sensor once, here, and are re-read through each
    candidate profile from there. The round trip is exact to 1e-13 px, so
    nothing inherits the starting profile's error.
    """

    def __init__(self, cal: dict):
        K, D, model = cal_.intrinsics_arrays(cal["intrinsics"])
        g_px, g_world, s_px, s_world, height = read_controls(cal)
        self.K0, self.D0, self.model = K, D, model
        self.sensor = nudge_.redistort(np.vstack([g_px, s_px]), K, D, model)
        self.g_world, self.s_world, self.height_mm = g_world, s_world, height
        self.in_pose = True             # read_controls has refused anything else
        self.n_ground = len(g_world)
        self.centre_mm = g_world.mean(axis=0)
        self.size = (int(cal["image_width"]), int(cal["image_height"]))

    def pose(self, K, D):
        """The pose the `gcp` step would have solved, had these clicks come through this profile."""
        und = cal_.undistort_points(self.sensor, K, D, self.model)
        g_obj = np.column_stack([self.g_world, np.zeros(self.n_ground)])
        R, t = cal_.solve_pnp(K, g_obj, und[:self.n_ground])
        _, up = cal_.camera_from_pose(R, t)
        s_obj = np.column_stack([self.s_world, np.full(len(self.s_world), up * self.height_mm)])
        if self.in_pose:
            R, t = cal_.solve_pnp(K, np.vstack([g_obj, s_obj]), und, seed=(R, t))
            if cal_.camera_from_pose(R, t)[1] != up:
                raise SystemExit("the combined solve put the camera under the ground; this "
                                 "profile cannot carry the survey")
        return und, np.vstack([g_obj, s_obj]), R, t, up


# --------------------------------------------------------------------------
# the two directions a profile reads in
# --------------------------------------------------------------------------


def to_world(px, K, D, model, R, t, h_mm) -> np.ndarray:
    und = cal_.undistort_points(np.asarray(px, dtype=np.float64).reshape(-1, 2), K, D, model)
    return cal_.apply_h(cal_.homography_at_height(K, R, t, h_mm), und)


def to_sensor(xy, K, D, model, R, t, h_mm) -> np.ndarray:
    H = cal_.homography_at_height(K, R, t, h_mm)
    und = cal_.apply_h(np.linalg.inv(H), np.asarray(xy, dtype=np.float64).reshape(-1, 2))
    return nudge_.redistort(und, K, D, model)


def field_deg(px, K, D, model) -> np.ndarray:
    und = cal_.undistort_points(np.asarray(px, dtype=np.float64).reshape(-1, 2), K, D, model)
    r = np.hypot((und[:, 0] - K[0, 2]) / K[0, 0], (und[:, 1] - K[1, 2]) / K[1, 1])
    return np.degrees(np.arctan(r))


def lens_gap_px(K1, D1, K2, D2, model, max_deg=GAP_DEG) -> float:
    """Largest distance between where two profiles put one ray, principal points aligned.

    A ray at field angle th lands `f * rho(th)` from the principal point, so this
    folds the focal length and every distortion term into the single quantity a
    measurement can actually see. Reported with c removed because the pose
    absorbs c and a measurement does not see it.
    """
    th = np.radians(np.linspace(0.0, max_deg, 331))
    phi = np.radians(np.arange(0.0, 360.0, 2.0))

    def rho(D):
        d = np.asarray(D, dtype=np.float64).ravel()
        if model == "fisheye":
            t2 = th ** 2
            return th * (1 + d[0] * t2 + d[1] * t2 ** 2 + d[2] * t2 ** 3 + d[3] * t2 ** 4)
        r = np.tan(th)
        r2 = r ** 2
        return r * (1 + d[0] * r2 + d[1] * r2 ** 2 + (d[4] if len(d) > 4 else 0.0) * r2 ** 3)

    r1, r2 = rho(D1), rho(D2)
    dx = np.outer(K1[0, 0] * r1 - K2[0, 0] * r2, np.cos(phi))
    dy = np.outer(K1[1, 1] * r1 - K2[1, 1] * r2, np.sin(phi))
    return float(np.hypot(dx, dy).max())


# --------------------------------------------------------------------------
# the fit
# --------------------------------------------------------------------------


def marker_size(read, w, h) -> np.ndarray:
    """Each frame's marker as a multiple of the rectangle it is supposed to be."""
    return 0.5 * (np.linalg.norm(read[:, 1] - read[:, 0], axis=1) / w
                  + np.linalg.norm(read[:, 3] - read[:, 0], axis=1) / h)


def growth_pct(size, d_m) -> float:
    """How much bigger the marker reads far away than near, in per cent.

    This is the signal, and it is scale-free: a template cut at the wrong
    millimetres per pixel moves the near and the far readings together.
    """
    near, far = size[d_m < BANDS_M[1][0]], size[d_m >= FAR_M]
    if not len(near) or not len(far):
        return float("nan")
    return float(100 * (np.median(far) / np.median(near) - 1))


def rectangle_residual(read, known) -> np.ndarray:
    """How far each read quad is from the known rectangle, after the best rigid placement.

    Size, squareness and aspect all live in this residual. Where the marker was
    and which way it faced do not, because the placement absorbs them -- which is
    exactly right: the car's position is not being measured here, only the shape
    the lens makes of a rectangle that is always the same.
    """
    a = read - read.mean(axis=1, keepdims=True)
    h = np.einsum("ki,nkj->nij", known, a)
    u, _s, vt = np.linalg.svd(h)
    d = np.linalg.det(np.einsum("nij,njk->nik", vt.transpose(0, 2, 1), u.transpose(0, 2, 1)))
    fix = np.zeros_like(u)
    fix[:, 0, 0] = 1.0
    fix[:, 1, 1] = d                       # a proper rotation, never a reflection
    rot = np.einsum("nij,njk,nkl->nil", vt.transpose(0, 2, 1), fix, u.transpose(0, 2, 1))
    return a - np.einsum("nij,kj->nki", rot, known)


def fit_focal(sv: Survey, px, known, h_mm, prior=PRIOR_WEIGHT) -> tuple:
    """Solve fx, fy so every marker reads as the rectangle it is, whatever its true size.

    Three unknowns: the two focal lengths, and the scale the marker really is.
    The pose is re-solved on the survey at every step, exactly as a station would
    solve it, so the fit is only ever asked for what the survey cannot supply.
    """
    K0, D0 = sv.K0.copy(), sv.D0.copy()
    flat = np.asarray(px, dtype=np.float64).reshape(-1, 2)
    x0 = np.array([K0[0, 0], K0[1, 1], 1.0])
    n_extra = 2 if prior else 0

    def unpack(x):
        K = K0.copy()
        K[0, 0], K[1, 1] = x[0], x[1]
        return K

    def residual(x):
        K = unpack(x)
        try:
            _, _, R, t, _ = sv.pose(K, D0)
            read = to_world(flat, K, D0, sv.model, R, t, h_mm).reshape(-1, 4, 2)
        except (SystemExit, cv2.error, np.linalg.LinAlgError):
            return np.full(flat.size + n_extra, 1e4)
        if not np.isfinite(read).all():
            return np.full(flat.size + n_extra, 1e4)
        out = rectangle_residual(read, known * x[2]).ravel()
        if prior:                   # a borrowed lens is already close to the one it came from
            out = np.concatenate([out, prior * (x[:2] - x0[:2]) / PRIOR_STEP_PX])
        return out

    x, cost = cal_.levmar(residual, x0, max_iter=200)
    return unpack(x), float(x[2]), float(np.sqrt(cost / max(len(flat), 1)))


def select(keep, d_m, tilt, frames) -> np.ndarray:
    """Draw the calmest frames band by band, so the fit gets reach and not repetition.

    An even draw along the clip follows the car's speed, not its position: a
    clip where the car creeps in and drives out fast hands back a fit with
    nothing but near frames, and near frames cannot see f at all.
    """
    per = max(1, frames // len(BANDS_M))
    take: list[int] = []
    for lo, hi in BANDS_M:
        pool = np.nonzero(keep & (d_m >= lo) & (d_m < hi))[0]
        if len(pool):
            take += list(pool[np.argsort(tilt[pool])[:per]])
    return np.array(sorted(take), dtype=int)


# --------------------------------------------------------------------------
# writing it out
# --------------------------------------------------------------------------


def carry_rois(src: Path, dst: Path, K_from, D_from, K_to, D_to, model) -> None:
    """Move an ROI file into the refitted profile's undistorted frame.

    An ROI has no world coordinate stored beside it, but it does not need one:
    the pixel goes back to the sensor through the profile it was clicked with --
    whatever that profile got wrong, it is the frame the click was made in -- and
    forward through the new one.
    """
    data = json.loads(src.read_text())
    moved = []
    for item in data.get("rois", []):
        pts = np.array(item.get("points_px") or [], dtype=np.float64).reshape(-1, 2)
        if not len(pts):
            continue                        # half-drawn: there is nothing to measure to
        out = cal_.undistort_points(nudge_.redistort(pts, K_from, D_from, model),
                                    K_to, D_to, model)
        moved.append({**item, "points_px": out.tolist()})
    shift = 0.0
    if moved:
        a = np.vstack([np.array(i["points_px"]) for i in moved])
        b = np.vstack([np.array(i.get("points_px")) for i in data["rois"] if i.get("points_px")])
        shift = float(np.linalg.norm(a - b, axis=1).max())
    print(f"  ROIs         {len(moved)} carried into the new frame, moved {shift:.1f} px worst")
    cal_.save_json(dst, {**data, "rois": moved})


def survey_stats(K, D, sv: Survey) -> dict:
    """How the survey sits in a profile: both planes and the reprojection, over all 8 points.

    Read the same way for the profile that came in and the one going out, so the
    two lines are comparable -- the stored `reproj_rms_px` is the ground marks
    alone and would flatter whichever profile it was written for.
    """
    und, obj, R, t, up = sv.pose(K, D)
    g_px, s_px = und[:sv.n_ground], und[sv.n_ground:]
    res_g = cal_.plane_residual_mm(K, R, t, g_px, sv.g_world, 0.0)
    res_s = cal_.plane_residual_mm(K, R, t, s_px, sv.s_world, sv.height_mm)
    rep = cal_.reproj_px(K, R, t, obj, und)
    return {"ground_mean_mm": float(res_g.mean()), "ground_max_mm": float(res_g.max()),
            "marker_mean_mm": float(res_s.mean()), "marker_max_mm": float(res_s.max()),
            "reproj_rms_px": float(np.sqrt(np.mean(rep ** 2))),
            "und": und, "obj": obj, "R": R, "t": t, "up": up,
            "res_g": res_g, "res_s": res_s, "rep": rep}


def write_calibration(cal: dict, intr: dict, K, D, model, sv: Survey, dst: Path) -> dict:
    """The survey re-solved through the refitted profile, in that profile's own frame."""
    st = survey_stats(K, D, sv)
    und, R, t, up, res_g, res_s, rep = (st["und"], st["R"], st["t"], st["up"],
                                        st["res_g"], st["res_s"], st["rep"])
    g_px, s_px = und[:sv.n_ground], und[sv.n_ground:]
    centre, _ = cal_.camera_from_pose(R, t)

    def points(px, world, res):
        return [{"pixel": {"x": float(p[0]), "y": float(p[1])},
                 "world": {"x_mm": float(w[0]), "y_mm": float(w[1])},
                 "residual_mm": float(r)}
                for p, w, r in zip(px, world, res, strict=True)]

    out = dict(cal)
    out["intrinsics"] = intr
    out["field"] = {**cal["field"],
                    "homography": cal_.homography_at_height(K, R, t, 0.0).tolist(),
                    "rms_error_mm": float(res_g.mean()), "max_error_mm": float(res_g.max()),
                    "gcps": points(g_px, sv.g_world, res_g),
                    "pose": {"rotation": R.tolist(), "translation_mm": np.ravel(t).tolist(),
                             "center_mm": centre.tolist(),
                             "tilt_deg": float(np.rad2deg(np.arccos(
                                 np.clip(-R[2, 2] * up, -1.0, 1.0)))),
                             "reproj_rms_px": float(rep[:sv.n_ground].mean()),
                             "solver": "focal_from_marker: fx, fy from the marker, then the "
                                       "`gcp` step's own solve on both planes"}}
    out["sticker_plane"] = {**cal["sticker_plane"],
                            "rms_error_mm": float(res_s.mean()),
                            "max_error_mm": float(res_s.max()),
                            "reproj_rms_px": float(rep[sv.n_ground:].mean()),
                            "controls": points(s_px, sv.s_world, res_s)}
    # A surveyed car plane and a tape check are clicks in the old frame with no
    # world coordinate to bring them across; carrying them would be silent and
    # wrong by the width of the refit.
    for key, what in (("car", "surveyed car plane (re-run `carplane`)"),
                      ("tape", "tape check (re-run `tape` -- it is the one check this "
                               "step cannot fake)")):
        if out.pop(key, None) is not None:
            print(f"  DROPPED the {what}")
    cal_.save_json(dst, out)
    return st


# --------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calibration", required=True, type=Path,
                    help="calibration.json from `gcp --sticker-height-mm ...`, carrying the "
                         "profile the track was detected with.")
    ap.add_argument("--track", required=True, type=Path,
                    help="A detection CSV written with --corner-pnp, on a clip where the car "
                         "crosses the scene.")
    ap.add_argument("--template", required=True, type=Path,
                    help="The cut marker template the track was detected with (sticker.png).")
    ap.add_argument("--template-mm-per-px", type=float, default=None,
                    help="Default: the mm_per_px in the sticker.json beside the template. Only "
                         "the template's shape matters; its size is re-fitted.")
    ap.add_argument("--out", required=True, type=Path, help="Where to write the refitted profile.")
    ap.add_argument("--out-calibration", type=Path, default=None,
                    help="Also re-solve the survey into the refitted profile's frame. Without "
                         "it the calibration still describes the old one.")
    ap.add_argument("--roi", type=Path, default=None, help="An ROI file to carry across.")
    ap.add_argument("--out-roi", type=Path, default=None, help="Where to write the carried ROIs.")
    ap.add_argument("--frames", type=int, default=160,
                    help="How many marker frames to fit, spread evenly over the distance bands.")
    ap.add_argument("--max-deg", type=float, default=MAX_FIELD_DEG,
                    help="Widest field angle a marker corner may sit at.")
    ap.add_argument("--max-tilt-deg", type=float, default=0.6,
                    help="Tilt a frame may report, or this track's own median if that is larger.")
    ap.add_argument("--min-far-frames", type=int, default=MIN_FAR,
                    help=f"Refuse the fit below this many frames beyond {FAR_M:g} m. They are "
                         "the lever arm; without them f is read off noise.")
    ap.add_argument("--min-sigma", type=float, default=2.0,
                    help="Refuse a change smaller than this many times the spread between two "
                         "independent halves of the same frames.")
    ap.add_argument("--force", action="store_true",
                    help="Write regardless of the gates. The numbers are printed either way.")
    a = ap.parse_args()
    if bool(a.roi) != bool(a.out_roi):
        raise SystemExit("--roi and --out-roi go together")
    t0 = time.time()
    print("focal_from_marker", flush=True)

    cal = cal_.load_json(a.calibration)
    sv = Survey(cal)
    K0, D0, model = sv.K0, sv.D0, sv.model
    w, h, known = template_rect(a.template, a.template_mm_per_px)
    frames, quad_mm, tilt, h_mm = read_corners(a.track)

    # The CSV's millimetres go back to the pixels they were measured at, through
    # the pose that measured them -- the one stored in the calibration, not a
    # re-solve of it. From there any candidate profile can read them.
    R_cal = np.array(cal["field"]["pose"]["rotation"])
    t_cal = np.array(cal["field"]["pose"]["translation_mm"])
    px = to_sensor(quad_mm.reshape(-1, 2), K0, D0, model, R_cal, t_cal, h_mm).reshape(-1, 4, 2)
    read0 = to_world(px.reshape(-1, 2), K0, D0, model, R_cal, t_cal, h_mm).reshape(-1, 4, 2)
    if float(np.abs(read0 - quad_mm).max()) > 0.01:
        raise SystemExit("the marker corners do not come back to where the track put them; the "
                         "calibration passed here is not the one the track was detected with.")

    d_m = np.linalg.norm(quad_mm.mean(axis=1) - sv.centre_mm, axis=1) / 1000
    size0 = marker_size(read0, w, h)
    ang = field_deg(px.reshape(-1, 2), K0, D0, model).reshape(-1, 4).max(axis=1)
    # The reported tilt is not a clean physical number -- corner-pnp solves the
    # marker's pose against the template's *claimed* size, so a template cut at
    # the wrong scale comes back as tilt. Each track's own calmest half is the
    # fair cut, and the band draw below stops it deleting the far frames.
    keep = ((tilt <= max(float(np.median(tilt)), a.max_tilt_deg)) & (ang <= a.max_deg)
            & (np.abs(size0 / np.median(size0) - 1.0) <= SIZE_OUTLIER))
    take = select(keep, d_m, tilt, a.frames)
    if not len(take):
        raise SystemExit("no marker frames survived the filters; loosen --max-tilt-deg or "
                         "--max-deg, or detect a clip with the car further out.")
    n_far = int((d_m[take] >= FAR_M).sum())

    print(f"  marker       {len(quad_mm)} corner-pnp frames in {a.track.name}, "
          f"{int(keep.sum())} usable, {len(take)} fitted ({n_far} beyond {FAR_M:g} m), "
          f"clip frames {frames[take].min()}-{frames[take].max()}")
    print(f"  coverage     {d_m[take].min():.1f} to {d_m[take].max():.1f} m from the control "
          f"square, out to {ang[take].max():.0f}° of field angle")
    print(f"  template     {w:.0f} x {h:.0f} mm as cut; the marker reads "
          f"{growth_pct(size0[take], d_m[take]):+.2f} % bigger far than near")

    K, scale, rms = fit_focal(sv, px[take], known, h_mm)
    # Two independent halves of the same frames, interleaved so both keep the
    # coverage. Their disagreement is the only honest error bar available here:
    # the survey never saw f, so its residuals cannot be one.
    halves = [fit_focal(sv, px[take[i::2]], known, h_mm)[0][0, 0] for i in (0, 1)]
    sigma = abs(halves[0] - halves[1]) / 2
    d_fx, d_fy = float(K[0, 0] - K0[0, 0]), float(K[1, 1] - K0[1, 1])
    ratio = abs(d_fx) / sigma if sigma > 1e-9 else float("inf")

    _, _, R1, t1, _ = sv.pose(K, D0)
    read1 = to_world(px[take].reshape(-1, 2), K, D0, model, R1, t1, h_mm).reshape(-1, 4, 2)
    grew0, grew1 = growth_pct(size0[take], d_m[take]), growth_pct(
        marker_size(read1, w * scale, h * scale), d_m[take])
    gap = lens_gap_px(K0, D0, K, D0, model)

    print(f"  focal length ({K0[0, 0]:.1f}, {K0[1, 1]:.1f}) -> ({K[0, 0]:.1f}, {K[1, 1]:.1f}) px"
          f"   moved ({d_fx:+.1f}, {d_fy:+.1f})")
    print(f"  its own spread  {sigma:.2f} px between two halves of these frames  ->  the change "
          f"is {'more than 1000' if ratio > 1000 else format(ratio, '.1f')} sigma")
    print(f"  principal point ({K0[0, 2]:.1f}, {K0[1, 2]:.1f}) px, held")
    print(f"  distortion      {np.array2string(np.asarray(D0).ravel(), precision=6)}, held")
    print(f"  marker size     {scale:.4f} of the cut template, so {w * scale:.0f} x "
          f"{h * scale:.0f} mm; {rms:.1f} mm rms left in its shape")
    print(f"  marker growth   {grew0:+.2f} % -> {grew1:+.2f} % near to far, which is the fit")
    print(f"  lens mapping    moved {gap:.1f} px at {GAP_DEG:g}° of field angle")

    # ---- the gates
    refused = []
    if n_far < a.min_far_frames:
        refused.append(f"only {n_far} frames beyond {FAR_M:g} m")
    if ratio < a.min_sigma:
        refused.append(f"f moved {ratio:.1f} times the spread between two halves")
    if n_far < a.min_far_frames and not a.force:
        raise SystemExit(
            f"\n  REFUSED: only {n_far} of the fitted frames are beyond {FAR_M:g} m, and f is "
            f"read from how the marker changes with distance -- with no reach there is no "
            f"signal, and the fit will happily move f by tens of pixels the wrong way.\n"
            f"  Detect a clip where the car drives out, or loosen --max-tilt-deg / --max-deg if "
            f"the far frames are being filtered rather than missing. A template cut at the wrong "
            f"scale reports tilt it does not have and loses exactly those frames.\n"
            f"  --force writes it anyway; --min-far-frames lowers the bar.")
    if ratio < a.min_sigma and not a.force:
        raise SystemExit(
            f"\n  NOT WRITTEN: f moved {abs(d_fx):.1f} px and two halves of the same frames "
            f"disagree by {sigma:.1f} px, so this fit cannot tell {ratio:.1f} sigma from zero. "
            f"Keep the profile as it is -- it is already inside what was measured here.\n"
            f"  To sharpen it, in the order that pays: more reach (the trend is the signal), "
            f"then more frames, then a calmer marker. --min-sigma lowers the bar, --force "
            f"ignores it.")

    intr = json.loads(json.dumps(cal["intrinsics"]))       # a copy, not the loaded object
    intr["camera_matrix"] = K.tolist()
    intr["marker_fit"] = {
        "fitted": ["fx", "fy"],
        "calibration": str(a.calibration),
        "track": str(a.track),
        "template": str(a.template),
        "template_mm": [w, h],
        "frames": len(take), f"frames_beyond_{FAR_M:g}m": n_far,
        "clip_frames": [int(frames[take].min()), int(frames[take].max())],
        "span_m": [float(d_m[take].min()), float(d_m[take].max())],
        "max_field_deg": float(ang[take].max()),
        "marker_grew_pct_near_to_far": grew0,
        "marker_grew_pct_after": grew1,
        "d_fx_px": d_fx, "d_fy_px": d_fy,
        "half_split_sigma_px": sigma, "change_sigmas": ratio,
        "marker_scale": scale, "marker_mm": [w * scale, h * scale],
        "residual_mm": rms,
        "lens_gap_px": gap, "lens_gap_deg": GAP_DEG,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "trusted": not refused,
        "note": "fx and fy solved so the car's marker keeps its size across the scene, with its "
                "true size free; c and the distortion terms are the board fit's, untouched"
                + ("" if not refused else ". FORCED PAST ITS GATES (" + ", ".join(refused)
                   + "): this profile is not trustworthy, and was written only because --force "
                     "was passed"),
    }
    if refused:
        print(f"  FORCED past the gates ({'; '.join(refused)}); the profile is marked untrusted")
    cal_.save_json(a.out, intr)

    if a.out_calibration:
        was = survey_stats(K0, D0, sv)
        now = write_calibration(cal, intr, K, D0, model, sv, a.out_calibration)
        print(f"  survey       ground {was['ground_mean_mm']:.1f} -> {now['ground_mean_mm']:.1f} "
              f"mm rms, marker {was['marker_mean_mm']:.1f} -> {now['marker_mean_mm']:.1f} mm, "
              f"reproj {was['reproj_rms_px']:.2f} -> {now['reproj_rms_px']:.2f} px")
        print("  the survey never chose f, so these residuals are an independent check on the "
              "fit rather than the fit admiring itself.")
    else:
        print("  the survey in the calibration file still describes the old frame. "
              "--out-calibration carries it across; without it, re-run `gcp`.")
    if a.roi:
        carry_rois(a.roi, a.out_roi, K0, D0, K, D0, model)
    print("  the cut template and the traced outline are millimetres measured through the old "
          "profile, and no remap can restore them: re-run `sticker` and `outline` against the "
          "new calibration before tracking with it.", flush=True)
    print(f"  done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
