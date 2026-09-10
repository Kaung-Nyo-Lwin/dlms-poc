"""Borrow one camera's lens profile for another, and nudge what differs.

Cameras of the same model share their glass and their sensor, so `fx`, `fy` and
the distortion coefficients are properties of the part number and travel between
units. What does not travel is where the sensor sits behind the lens: the
principal point `(cx, cy)` is set by how the two were bolted together, and it
moves by tens of pixels from unit to unit. So a master calibration transplanted
whole is right about the lens and wrong about the assembly.

This step re-solves the assembly, and only the assembly, against control points
already surveyed on the target camera::

    python src/poc/nudge_intrinsics.py --calibration $S/calibration.json \\
        --out intrinsics/B7_nudged.json

Everything hardware-invariant is frozen — focal length, aspect, distortion — and
what is left is `(cx, cy)` plus the six numbers of the pose: eight unknowns
against two equations per control point. The `gcp` step's own 4 ground marks and
4 marker-plane targets are exactly the sixteen equations that makes solvable, so
nothing new has to be surveyed.

**The two heights are the whole basis of it.** A shift in `cx` looks almost
exactly like a pan of the camera, and on one flat plane the two are the same
picture — the fit then splits 8 unknowns across a mapping that only ever had 8
of its own, and lands anywhere. Points at a second height break the tie, because
a pan moves the two planes together and a principal-point shift does not. That
is the same parallax `gcp --pose-from all` and `carplane` live on, and it is why
this step refuses a survey that is all on the ground.

It is also why the answer comes with a `sigma` and a gate. Eight points leave
eight degrees of freedom, and the fit will spend click noise on `(cx, cy)`
happily — on a synthetic B7 (8 m up, targets at 1450 mm, true offset 125 px)
quarter-pixel clicking recovered the principal point to 59 px with sigma 66,
and half-pixel clicking to 120 px with sigma 134. The measured shift is worth
taking when it stands clear of its own sigma and not otherwise, so the profile
is written only when the shift reaches ``--min-sigma`` sigmas (2 by default).
Sigma tracked the true error closely across every configuration tested, which is
what makes it usable as the gate rather than as decoration.

Whether the nudge is worth doing at all is a question about the two cameras, and
the same synthetic answers it: on the marker's plane, a 125 px offset carried
un-nudged cost about 10.5 mm, and nudging brought it to 6.3 mm with 8 points,
4.3 mm with 12 and 3.1 mm with 24. Below roughly 50 px of true offset there is
nothing to win — the master profile is already inside the noise, and the nudge
adds about 1.7 mm of its own. More parallax is worth more than more points: the
same 8 clicks with targets at 3000 mm instead of 1450 halved sigma.

**The nudged profile defines a new undistorted frame.** Every pixel this toolkit
records lives in the frame `P = K` projects, so moving `cx` by 100 px moves that
frame by 100 px, and every click already stored — the GCPs, the ROIs, the car
outline, the cut template — describes the old one. ``--out-calibration`` remaps
what it can prove it is safe to remap and says loudly what it cannot; the files
it names have to be re-run.

Standalone: numpy + opencv + the standard library, plus `calibrate.py` beside it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import calibrate as cal_
import cv2
import numpy as np

# The flags that lock everything the two cameras share. Focal length and the
# radial terms are the lens; the principal point is the assembly, and it is the
# only intrinsic left free. FIX_ASPECT_RATIO is redundant beside FIX_FOCAL_LENGTH
# and kept because it states the intent: the pixels are square in the same way on
# both units, whatever the fit is tempted to do with them.
_BASE_FLAGS = (cv2.CALIB_USE_INTRINSIC_GUESS
               | cv2.CALIB_FIX_FOCAL_LENGTH
               | cv2.CALIB_FIX_ASPECT_RATIO
               | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3)

#: A principal point this far from the master's is not an assembly tolerance, it
#: is a wrong master or a mistyped control point. Expressed as a fraction of the
#: frame because the pixel figure means nothing without the sensor's size.
MAX_SHIFT_FRACTION = 0.05


def pinhole_flags(D: np.ndarray) -> tuple[int, np.ndarray, str]:
    """Flags and a padded distortion vector that leave ``D`` genuinely untouched.

    ``CALIB_ZERO_TANGENT_DIST`` is the flag the recipe calls for and it does two
    things, not one: it holds `p1` and `p2` still *at zero*. On a master that has
    non-zero tangential terms it would silently discard them, and the profile
    written out would no longer describe the lens either camera has. So the
    decentring terms decide which flag applies — zero them when they are already
    zero, fix them where they are otherwise — and the result is checked against
    the master afterwards either way.
    """
    D = np.asarray(D, dtype=np.float64).ravel()
    if len(D) == 4:                       # (k1, k2, p1, p2), no k3
        D = np.append(D, 0.0)
    if len(D) == 5:
        flags = _BASE_FLAGS
    elif len(D) == 8:                     # rational model
        flags = (_BASE_FLAGS | cv2.CALIB_RATIONAL_MODEL
                 | cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_K6)
    else:
        raise SystemExit(
            f"this master has {len(D)} distortion coefficients. Only the 5-term and the "
            "8-term rational pinhole models are handled here; thin-prism and tilted-sensor "
            "models have terms this step has no flag to lock.")
    if float(D[2]) == 0.0 and float(D[3]) == 0.0:
        return flags | cv2.CALIB_ZERO_TANGENT_DIST, D, "ZERO_TANGENT_DIST"
    return flags | cv2.CALIB_FIX_TANGENT_DIST, D, "FIX_TANGENT_DIST"


def redistort(px, K, D, model):
    """Undo ``calibrate.undistort_points`` — undistorted pixels back to sensor pixels.

    Every pixel this toolkit stores was clicked on an undistorted frame, and
    ``calibrateCamera`` models the lens, so feeding it those clicks straight
    would apply the correction twice. The round trip is exact to 1e-13 px: the
    survey's own projection is `P = K`, so `K` inverts the frame back to
    normalised rays and the model's own projection puts them back on the sensor.

    Exactness matters more than it looks. The master's `(cx, cy)` is the wrong
    one — that is the premise of this whole step — but it is the one the frame
    was built with, so inverting it recovers the true sensor pixel regardless of
    how wrong it was. The fit downstream is not inheriting the master's error.
    """
    p = np.asarray(px, dtype=np.float64).reshape(-1, 2)
    rays = np.column_stack([(p[:, 0] - K[0, 2]) / K[0, 0],
                            (p[:, 1] - K[1, 2]) / K[1, 1],
                            np.ones(len(p))])
    if model == "fisheye":
        out, _ = cv2.fisheye.projectPoints(rays.reshape(-1, 1, 3), np.zeros(3), np.zeros(3),
                                           K, np.asarray(D, dtype=np.float64)[:4].reshape(4, 1))
    else:
        out, _ = cv2.projectPoints(rays.reshape(-1, 1, 3), np.zeros(3), np.zeros(3),
                                   K, np.asarray(D, dtype=np.float64).ravel())
    return out.reshape(-1, 2)


def read_controls(cal: dict):
    """The two sets of control points the `gcp` step surveyed, and the height between them.

    Ground marks alone cannot do this, and the refusal is not a threshold that
    could be argued down: with every point on one plane, a principal-point shift
    and a rotation of the camera produce the same image, and no amount of care
    with the clicking separates them.
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
            "this survey is all on the ground, and a principal point cannot be recovered from "
            "one plane: shifting (cx, cy) and rotating the camera make the same picture, and "
            "only points at a second height tell them apart. Re-run\n"
            "    calibrate.py gcp --image ... --sticker-height-mm 1450 --pose-from all\n"
            "to survey targets on the marker's plane as well — 4 of them, which with the "
            "4 ground marks is the 8 points this step wants.")

    def xy(rows):
        return (np.array([[r["pixel"]["x"], r["pixel"]["y"]] for r in rows], dtype=np.float64),
                np.array([[r["world"]["x_mm"], r["world"]["y_mm"]] for r in rows], dtype=np.float64))

    g_px, g_world = xy(gcps)
    s_px, s_world = xy(controls)
    return g_px, g_world, s_px, s_world, float(sp["height_mm"])


def nudge_pinhole(K, D, size, obj, img):
    """``cv2.calibrateCamera`` with everything but the principal point locked.

    One view, so both point sets are wrapped in a list of one. The extended form
    is used for one reason: ``stdDeviationsIntrinsics`` is the only honest thing
    in the output. The reprojection RMS is not — 8 points give 16 equations
    against 8 unknowns, and a fit with that much freedom drives its residual down
    by moving the principal point wherever the noise asks it to.
    """
    flags, D5, tangent = pinhole_flags(D)
    rms, K_new, D_out, rvecs, tvecs, sd_int, _, _ = cv2.calibrateCameraExtended(
        [obj.astype(np.float32)], [img.astype(np.float32)], size,
        np.asarray(K, dtype=np.float64).copy(), D5.copy().reshape(1, -1), flags=flags)
    sd = np.asarray(sd_int, dtype=np.float64).ravel()
    return (np.asarray(K_new, dtype=np.float64), np.asarray(D_out, dtype=np.float64).ravel(),
            cv2.Rodrigues(rvecs[0])[0], np.asarray(tvecs[0], dtype=np.float64).ravel(),
            float(rms), float(sd[2]), float(sd[3]),
            f"cv2.calibrateCamera, {tangent}")


def nudge_fisheye(K, D, obj, img, R0, t0):
    """The same eight unknowns, for a lens ``cv2.calibrateCamera`` cannot describe.

    Every station here is fisheye, and the pinhole fitter has no way to represent
    that lens: handed four fisheye coefficients as a `1 x 5` array it reads them
    as `(k1, k2, p1, p2)` and returns a converged, plausible, wrong answer.
    ``cv2.fisheye.calibrate`` is the matching fitter but not the matching
    *problem* — it has no flag that frees the principal point while holding the
    focal length, and on noiseless synthetic input it stopped 26 px from the
    right answer.

    So the same estimator is written out directly: `(cx, cy)` and the pose
    against ``fisheye.projectPoints``, minimised by the Levenberg-Marquardt in
    `calibrate.py`. Run on pinhole data beside ``cv2.calibrateCamera`` it agrees
    to 0.01 px on the principal point and 0.001 mm on the camera centre, so this
    is the recipe's own solve reaching a lens it could not otherwise reach — not
    a second opinion.
    """
    K = np.asarray(K, dtype=np.float64)
    D4 = np.asarray(D, dtype=np.float64)[:4].reshape(4, 1)

    def residual(x):
        K_x = K.copy()
        K_x[0, 2], K_x[1, 2] = x[0], x[1]
        proj, _ = cv2.fisheye.projectPoints(obj.reshape(-1, 1, 3), x[2:5].reshape(3, 1),
                                            x[5:8].reshape(3, 1), K_x, D4)
        return (proj.reshape(-1, 2) - img).ravel()

    x0 = np.concatenate([[K[0, 2], K[1, 2]], cv2.Rodrigues(R0)[0].ravel(), np.ravel(t0)])
    x, cost = cal_.levmar(residual, x0, max_iter=200)

    # Covariance of the converged fit, for the same sigma the pinhole path gets
    # from OpenCV: s^2 * (J'J)^-1, with s^2 the residual variance per degree of
    # freedom. pinv rather than inv so a degenerate configuration reports a huge
    # sigma — which the gate will then refuse — instead of raising.
    r = residual(x)
    J = np.empty((len(r), len(x)))
    for k in range(len(x)):
        h = 1e-6 * max(1.0, abs(x[k]))
        e = np.zeros(len(x))
        e[k] = h
        J[:, k] = (residual(x + e) - residual(x - e)) / (2 * h)
    dof = max(len(r) - len(x), 1)
    cov = (cost / dof) * np.linalg.pinv(J.T @ J)

    K_new = K.copy()
    K_new[0, 2], K_new[1, 2] = x[0], x[1]
    rms = float(np.sqrt(cost / (len(r) / 2)))
    return (K_new, np.asarray(D, dtype=np.float64).ravel(), cv2.Rodrigues(x[2:5])[0], x[5:8],
            rms, float(np.sqrt(cov[0, 0])), float(np.sqrt(cov[1, 1])),
            "Levenberg-Marquardt on fisheye.projectPoints")


def cmd_nudge(a) -> None:
    cal = cal_.load_json(a.calibration)
    master = cal_.load_json(a.master) if a.master else cal.get("intrinsics")
    if master is None:
        raise SystemExit("this calibration carries no intrinsics block; pass --master")
    K, D, model = cal_.intrinsics_arrays(master)

    # The clicks stored in this calibration were made on a frame undistorted by
    # whatever profile the `gcp` step was given. A different master here would
    # put them in a different pixel frame, and re-distorting them through this
    # one would land them somewhere else on the sensor — quietly, and by about
    # as much as the correction being solved for.
    if a.master:
        own = cal.get("intrinsics")
        if own is not None and not np.allclose(np.array(own["camera_matrix"]), K, atol=1e-6):
            raise SystemExit(
                f"--master {a.master} is not the profile this survey was clicked through, and "
                "every pixel in it is stored in that profile's undistorted frame. Re-run the "
                "`gcp` step against this master, or drop --master to use the stored one.")

    size = (int(cal["image_width"]), int(cal["image_height"]))
    if (int(master["image_width"]), int(master["image_height"])) != size:
        raise SystemExit(
            f"the master describes {master['image_width']}x{master['image_height']} and this "
            f"survey is {size[0]}x{size[1]}. A profile does not scale across framings.")

    g_px, g_world, s_px, s_world, height_mm = read_controls(cal)
    n = len(g_px) + len(s_px)
    print(f"  {len(g_px)} ground marks + {len(s_px)} targets at {height_mm:.0f} mm "
          f"= {n} points, {2 * n} equations against 8 unknowns "
          f"(cx, cy and the pose), {2 * n - 8} degrees of freedom")

    # The pose the survey already has, which is both the seed and the thing the
    # nudge has to be compared against. Ground first because IPPE needs one
    # plane, then refined on everything — `gcp --pose-from all`'s own sequence.
    R_pre, t_pre = cal_.solve_pnp(K, np.column_stack([g_world, np.zeros(len(g_world))]), g_px)
    centre_pre, up = cal_.camera_from_pose(R_pre, t_pre)
    obj = np.vstack([np.column_stack([g_world, np.zeros(len(g_world))]),
                     np.column_stack([s_world, np.full(len(s_world), up * height_mm)])])
    und = np.vstack([g_px, s_px])
    R_pre, t_pre = cal_.solve_pnp(K, obj, und, seed=(R_pre, t_pre))
    centre_pre, up = cal_.camera_from_pose(R_pre, t_pre)

    # Onto the sensor, where the lens model lives.
    img = redistort(und, K, D, model)

    if model == "fisheye":
        K_new, D_new, R, t, rms, sig_x, sig_y, solver = nudge_fisheye(K, D, obj, img,
                                                                      R_pre, t_pre)
    else:
        K_new, D_new, R, t, rms, sig_x, sig_y, solver = nudge_pinhole(K, D, size, obj, img)

    # The recipe's promise is that the distortion comes back untouched. It is
    # cheap to check and it is the one way to tell that the flags locked what
    # they were meant to: a coefficient that moved means this fit was free
    # somewhere it should not have been, and the profile would be wrong for both
    # cameras rather than nudged for one.
    # OpenCV returns the widest array its flags could have filled — 14 terms for
    # a rational fit, 5 where the master had 4 — so the tail is checked for the
    # zeros it should be and then dropped. What is written stays the shape the
    # master was, because a profile that changes length between cameras of one
    # model invites the question of which terms were fitted, and none were.
    D_cmp = np.asarray(D, dtype=np.float64).ravel()
    head, tail = D_new[:len(D_cmp)], D_new[len(D_cmp):]
    if not (np.allclose(head, D_cmp, rtol=0, atol=1e-12) and np.all(tail == 0.0)):
        raise SystemExit(
            "the distortion coefficients moved during the fit, which the flags were supposed "
            f"to prevent:\n    master {D_cmp}\n    fitted {D_new}\n"
            "Nothing is written — this is a bug in the flag set for this model, not a survey "
            "problem.")
    D_new = head

    dcx, dcy = K_new[0, 2] - K[0, 2], K_new[1, 2] - K[1, 2]
    shift = float(np.hypot(dcx, dcy))
    sigma = float(np.hypot(sig_x, sig_y))
    # A sigma of zero is a fit that reproduces its points exactly, which happens
    # only on synthetic input; on real clicks it would mean fewer independent
    # observations than parameters. Either way the shift is not zero-over-zero.
    ratio = shift / sigma if sigma > 1e-9 else float("inf")
    max_shift = MAX_SHIFT_FRACTION * size[0]

    centre, up_new = cal_.camera_from_pose(R, t)
    tilt_pre = float(np.rad2deg(np.arccos(np.clip(-R_pre[2, 2] * up, -1.0, 1.0))))
    tilt = float(np.rad2deg(np.arccos(np.clip(-R[2, 2] * up_new, -1.0, 1.0))))
    rms_pre = cal_.reproj_px(K, R_pre, t_pre, obj, und).mean()

    print(f"  solver: {solver}")
    print(f"  principal point ({K[0, 2]:.1f}, {K[1, 2]:.1f}) -> ({K_new[0, 2]:.1f}, "
          f"{K_new[1, 2]:.1f}) px   moved {shift:.1f} px")
    print(f"  its own sigma   ({sig_x:.1f}, {sig_y:.1f}) px  ->  the shift is "
          f"{'more than 1000' if ratio > 1000 else format(ratio, '.1f')} sigma")
    print(f"  focal length    ({K[0, 0]:.1f}, {K[1, 1]:.1f}) px, held")
    print(f"  distortion      {np.array2string(D_new, precision=6)}, unchanged")
    print(f"  reprojection    {rms_pre:.3f} -> {rms:.3f} px  (over {n} points; with "
          f"{2 * n - 8} degrees of freedom this is not a check)")
    print(f"  camera   ({centre_pre[0]:.0f}, {centre_pre[1]:.0f}, {centre_pre[2] * up:.0f}) -> "
          f"({centre[0]:.0f}, {centre[1]:.0f}, {centre[2] * up_new:.0f}) mm, moved "
          f"{np.linalg.norm(centre - centre_pre):.0f} mm")
    print(f"  tilt     {tilt_pre:.2f}° -> {tilt:.2f}° from nadir")

    # What the two profiles say about the tarmac itself. The clicks have to be
    # carried into each profile's own undistorted frame first, because that is
    # the frame each homography reads: comparing them on one set of pixels would
    # compare the wrong thing and flatter whichever profile the pixels came from.
    und_new = cal_.undistort_points(img, K_new, D_new, model)
    g_new, s_new = und_new[:len(g_px)], und_new[len(g_px):]
    res_pre_g = cal_.plane_residual_mm(K, R_pre, t_pre, g_px, g_world, 0.0)
    res_pre_s = cal_.plane_residual_mm(K, R_pre, t_pre, s_px, s_world, height_mm)
    res_g = cal_.plane_residual_mm(K_new, R, t, g_new, g_world, 0.0)
    res_s = cal_.plane_residual_mm(K_new, R, t, s_new, s_world, height_mm)
    print(f"  on the ground   rms {res_pre_g.mean():.1f} -> {res_g.mean():.1f} mm   "
          f"max {res_pre_g.max():.1f} -> {res_g.max():.1f} mm")
    print(f"  on the marker   rms {res_pre_s.mean():.1f} -> {res_s.mean():.1f} mm   "
          f"max {res_pre_s.max():.1f} -> {res_s.max():.1f} mm")
    print("  neither residual above is a check on the nudge: these are the very points it "
          "was fitted to. `measure` against a tape, or `carplane` on targets this solve "
          "never saw, are what can still argue with it.", flush=True)

    probe = cal_.probe_shift(K, R_pre, t_pre, cal_.homography_at_height(K_new, R, t, 0.0),
                             np.vstack([g_world, s_world]), size[0], size[1])
    if probe.size:
        print(f"  the ground under the survey moves by {np.median(probe):.0f} mm typical, "
              f"{probe.max():.0f} mm worst")

    # ---- the gate
    if shift > max_shift and not a.force:
        raise SystemExit(
            f"\n  REFUSED: {shift:.0f} px is {100 * shift / size[0]:.1f}% of the frame width. "
            f"A principal point does not move that far between units of one model — that is a "
            f"master from a different camera model, a mistyped world coordinate, or a click on "
            f"the wrong mark. Check the worst residual above. --force writes it anyway.")
    if ratio < a.min_sigma and not a.force:
        raise SystemExit(
            f"\n  NOT WRITTEN: the shift is {shift:.0f} px and its own sigma is {sigma:.0f} px, "
            f"so this fit cannot tell {ratio:.1f} sigma from zero. Keep the master profile — it "
            f"is inside the noise of what was measured here.\n"
            f"  To sharpen it, in the order that pays: hold the targets higher (parallax is "
            f"worth more than points — 3000 mm instead of 1450 halved sigma in testing), click "
            f"more of them, and click them more carefully. Then re-run `gcp`.\n"
            f"  --min-sigma lowers the bar and --force ignores it; below about 2 sigma the "
            f"nudge was as likely to make the survey worse as better.")

    intr = {
        "model": model,
        "camera_matrix": K_new.tolist(),
        "dist_coeffs": D_new.tolist(),
        "image_width": size[0],
        "image_height": size[1],
        "rms_reproj_px": float(rms),
        "n_views": 1,
        "nudged_from": {
            "master": str(a.master) if a.master else "the calibration's own intrinsics block",
            "master_camera_matrix": np.asarray(K).tolist(),
            "calibration": str(a.calibration),
            "solver": solver,
            "n_ground": len(g_px),
            "n_plane": len(s_px),
            "plane_height_mm": height_mm,
            "shift_px": [float(dcx), float(dcy)],
            "sigma_px": [sig_x, sig_y],
            "shift_sigmas": ratio,
            "pose": {"rotation": R.tolist(), "translation_mm": np.ravel(t).tolist(),
                     "center_mm": centre.tolist(), "tilt_deg": tilt},
            "ground_rms_mm": float(res_g.mean()),
            "plane_rms_mm": float(res_s.mean()),
        },
    }
    if "board" in master:
        intr["board"] = master["board"]
    cal_.save_json(a.out, intr)

    if a.out_calibration:
        write_calibration(a, cal, intr, K_new, D_new, model, R, t, up_new,
                          g_new, g_world, res_g, s_new, s_world, res_s, height_mm, rms)
    else:
        print("  the survey in the calibration file still describes the master's undistorted "
              "frame. --out-calibration carries it across; without it, re-run `gcp` against "
              "this profile.", flush=True)


def write_calibration(a, cal, intr, K, D, model, R, t, up,
                      g_px, g_world, res_g, s_px, s_world, res_s, height_mm, rms) -> None:
    """Carry the survey into the nudged profile's frame — and say what cannot be carried.

    A profile change is a *frame* change here, which the `tape` step's pose
    corrections are not: `P = K` projects the undistorted image, so moving the
    principal point by 100 px slides every undistorted pixel by about 100 px.
    The control points survive it because their sensor pixels are recoverable and
    were re-projected through the new profile above. Nothing else in the survey
    is: an ROI, a traced outline and a cut template are pixels with no world
    coordinate stored beside them, so there is no way back to the sensor and no
    way forward into the new frame. They are named rather than silently left,
    because every one of them would still load, still draw, and be wrong by the
    width of the nudge.
    """
    cal = dict(cal)
    cal["intrinsics"] = intr
    centre, _ = cal_.camera_from_pose(R, t)
    cal["field"] = dict(cal["field"])
    cal["field"]["homography"] = cal_.homography_at_height(K, R, t, 0.0).tolist()
    cal["field"]["rms_error_mm"] = float(res_g.mean())
    cal["field"]["max_error_mm"] = float(res_g.max())
    cal["field"]["gcps"] = [
        {"pixel": {"x": float(p[0]), "y": float(p[1])},
         "world": {"x_mm": float(w[0]), "y_mm": float(w[1])},
         "residual_mm": float(r)}
        for p, w, r in zip(g_px, g_world, res_g, strict=True)]
    cal["field"]["pose"] = {
        "rotation": R.tolist(),
        "translation_mm": np.ravel(t).tolist(),
        "center_mm": centre.tolist(),
        "tilt_deg": float(np.rad2deg(np.arccos(np.clip(-R[2, 2] * up, -1.0, 1.0)))),
        "reproj_rms_px": float(rms),
        "solver": "nudge_intrinsics: cx, cy and the pose against both planes",
    }
    sp = dict(cal["sticker_plane"])
    s_obj = np.column_stack([s_world, np.full(len(s_world), up * height_mm)])
    sp["controls"] = [
        {"pixel": {"x": float(p[0]), "y": float(p[1])},
         "world": {"x_mm": float(w[0]), "y_mm": float(w[1])},
         "residual_mm": float(r)}
        for p, w, r in zip(s_px, s_world, res_s, strict=True)]
    sp["rms_error_mm"] = float(res_s.mean())
    sp["max_error_mm"] = float(res_s.max())
    sp["reproj_rms_px"] = float(cal_.reproj_px(K, R, t, s_obj, s_px).mean())
    sp["in_pose"] = True
    cal["sticker_plane"] = sp

    # A surveyed car plane is a homology of the old ground plane, read off pole
    # tops clicked in the old frame. `tape` can re-fit one because a pose change
    # leaves those pixels meaning what they meant; a frame change does not.
    if cal.pop("car", None) is not None:
        print("  DROPPED the surveyed car plane: its pole tops are pixels in the master's "
              "undistorted frame, and this profile no longer projects that frame. Re-run "
              "`carplane`.")
    if cal.pop("tape", None) is not None:
        print("  DROPPED the tape check: its bar endpoints were clicked in the old frame too. "
              "Re-run `tape` — and do, because it is the one check the nudge cannot fake.")

    cal_.save_json(a.out_calibration, cal)
    print("  the ROIs, the car outline and the cut template are all pixels in the old frame "
          "with no world coordinate to carry them across. Re-run `roi`, `sticker` and "
          "`outline` against this calibration before tracking anything.", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calibration", required=True, type=Path,
                    help="calibration.json from `calibrate.py gcp --sticker-height-mm ...` on "
                         "the target camera, clicked through the master profile.")
    ap.add_argument("--out", required=True, type=Path,
                    help="Where to write the nudged intrinsics profile.")
    ap.add_argument("--master", type=Path, default=None,
                    help="Master profile. Default: the intrinsics block inside --calibration, "
                         "which is the profile its pixels were clicked through.")
    ap.add_argument("--out-calibration", type=Path, default=None,
                    help="Also rewrite the survey into the nudged profile's undistorted frame, "
                         "with the pose this step solved. Without it the calibration still "
                         "describes the master's frame.")
    ap.add_argument("--min-sigma", type=float, default=2.0,
                    help="Write only when the measured shift is at least this many of its own "
                         "sigmas. Below 2 the nudge was as likely to hurt as help.")
    ap.add_argument("--force", action="store_true",
                    help="Write regardless of the gates. The numbers are printed either way.")
    a = ap.parse_args()
    print("nudge_intrinsics", flush=True)
    cmd_nudge(a)


if __name__ == "__main__":
    main()
