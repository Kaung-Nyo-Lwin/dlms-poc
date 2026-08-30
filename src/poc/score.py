"""Score a tracked run against the rules in a rules.json.

    python score.py --track track.csv --rules rules.json \
                    --calibration calibration.json --out scorecard.json

Reads the box the pipeline wrote for every frame, works out where the car came
to rest and what it touched on the way, and takes points off. The rules and the
regions they are measured against both live in one file, seeded from
``dsr_rules.json`` and drawn in with ``calibrate.py rules``.

**A rule this cannot check is reported, never skipped.** Every rule carries a
``needs`` list naming the capabilities it depends on, and anything missing comes
back as ``not-evaluated`` with the reason. A scoring engine that quietly passes
what it cannot see is worse than one that refuses: the score looks complete and
is not. Eleven of the thirty-four DSR rules need wheel positions, which no
pipeline here produces yet; the run summary says so every time.

ROI pixels are read on the **field** plane, where paint and kerbs are, and the
car box arrives already in world millimetres. Nothing here touches an image.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from itertools import pairwise
from pathlib import Path

import numpy as np
import pipeline1_bev as pipeline

#: Box-centre speed below which the car counts as stopped, in mm/s. A parked car
#: still jitters by the detector's own noise, which the pipeline reports per
#: frame as sigma_mm — a few millimetres at B7 — so this sits well above that
#: and well below a walking pace.
STOPPED_MM_S = 120.0
#: Shortest run of stopped frames that counts as a stop rather than a hesitation.
MIN_STOP_S = 0.4
#: How far a point must cross a line before it counts as over it, in mm. Below
#: this the answer is dominated by where exactly the paint was clicked.
CROSS_TOL_MM = 20.0


def load_track(path: Path) -> list[dict]:
    """Detected frames, each with its box in world millimetres."""
    rows = []
    with Path(path).open() as fh:
        reader = csv.DictReader(fh)
        rows_in = list(reader)
        has_wheels = all(f"wheel_{w}_x_mm" in (reader.fieldnames or [])
                         for w in ("fl", "fr", "rl", "rr"))
    for row in rows_in:
        if row.get("found") != "1":
            continue
        box = np.array([[float(row[f"box{i}_x_mm"]), float(row[f"box{i}_y_mm"])]
                        for i in range(1, 5)])
        rec = {
            "frame": int(row["frame"]), "t": float(row["time_s"]), "box": box,
            "centre": box.mean(axis=0), "heading": float(row["heading_deg"]),
            "sticker": np.array([float(row["sticker_x_mm"]), float(row["sticker_y_mm"])]),
        }
        if has_wheels:
            rec["wheels"] = {w: np.array([float(row[f"wheel_{w}_x_mm"]),
                                          float(row[f"wheel_{w}_y_mm"])])
                             for w in ("fl", "fr", "rl", "rr")}
        rows.append(rec)
    if not rows:
        raise SystemExit(f"{path} has no detected frames to score")
    return rows, has_wheels


#: Which wheels each rule's `part` name means.
WHEEL_GROUPS = {"wheels": ("fl", "fr", "rl", "rr"), "rear_wheels": ("rl", "rr"),
                "front_wheels": ("fl", "fr"), "right_wheels": ("fr", "rr"),
                "left_wheels": ("fl", "rl")}


def wheel_pts(frame, part):
    """The contact patches a rule's ``part`` names, or None if unavailable."""
    if "wheels" not in frame or part not in WHEEL_GROUPS:
        return None
    return np.array([frame["wheels"][w] for w in WHEEL_GROUPS[part]])


def on_line(pt, roi, tyre_mm):
    """Whether a tyre is touching an ROI — its centre within half a tread of it."""
    d = min(_point_seg(pt, a, b) for a, b in roi_segments(roi))
    return d <= (tyre_mm or 0.0) / 2.0, d


def roi_segments(roi):
    p = np.asarray(roi["world_mm"], dtype=np.float64).reshape(-1, 2)
    if len(p) < 2:
        return [(p[0], p[0])]
    n = len(p) if roi["closed"] else len(p) - 1
    return [(p[i], p[(i + 1) % len(p)]) for i in range(n)]


def _point_seg(p, a, b) -> float:
    d = b - a
    L = float(d @ d)
    t = 0.0 if L < 1e-12 else float(np.clip((p - a) @ d / L, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * d)))


def runs_where(track, pred):
    """Contiguous ``(i, j)`` spans of frames where a predicate holds."""
    out, i = [], 0
    flags = [bool(pred(f)) for f in track]
    while i < len(flags):
        if not flags[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(flags) and flags[j + 1]:
            j += 1
        out.append((i, j))
        i = j + 1
    return out


def parts_of(box: np.ndarray) -> dict:
    """Named pieces of the car box.

    The polygon is stored front-left, front-right, rear-right, rear-left, which
    both `car` and `outline` write and `car_box` preserves, so the bumpers and
    sides fall out of it directly.
    """
    fl, fr, rr, rl = box
    return {"body": box, "front_bumper": np.array([fl, fr]),
            "rear_bumper": np.array([rl, rr]), "left_side": np.array([fl, rl]),
            "right_side": np.array([fr, rr]), "bumpers": np.array([fl, fr, rl, rr])}


def line_of(pts: np.ndarray):
    """A point on an ROI and its unit direction, for signed-side tests."""
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    d = p[-1] - p[0]
    n = float(np.linalg.norm(d))
    if n < 1e-6:
        return p[0], np.array([1.0, 0.0])
    return p[0], d / n


def signed_side(pts: np.ndarray, roi: np.ndarray) -> np.ndarray:
    """Perpendicular offset of each point from an ROI line, signed by side."""
    p0, u = line_of(roi)
    q = np.asarray(pts, dtype=np.float64).reshape(-1, 2) - p0
    return q[:, 0] * u[1] - q[:, 1] * u[0]


def stop_runs(track: list[dict]) -> list[tuple[int, int]]:
    """Index ranges over which the car was not moving.

    Speed comes from the box centre between consecutive detected frames rather
    than from the sticker, so a car that is turning on the spot still reads as
    stopped only if its whole footprint is.
    """
    if len(track) < 2:
        return []
    slow = [True]
    for a, b in pairwise(track):
        dt = b["t"] - a["t"]
        v = float(np.linalg.norm(b["centre"] - a["centre"])) / dt if dt > 1e-6 else 0.0
        slow.append(v < STOPPED_MM_S)
    runs, i = [], 0
    while i < len(slow):
        if not slow[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(slow) and slow[j + 1]:
            j += 1
        if track[j]["t"] - track[i]["t"] >= MIN_STOP_S:
            runs.append((i, j))
        i = j + 1
    return runs


def gap_to(box: np.ndarray, roi: dict) -> float:
    return pipeline.clearance_mm(box, roi["world_mm"], roi["closed"])[0]


# --------------------------------------------------------------------------
# the checks
# --------------------------------------------------------------------------


def check_stop_band(rule, ctx):
    """At rest, the gap to a line must sit inside a band."""
    roi, p = ctx["roi"](rule), rule["params"]
    if roi is None or not ctx["stops"]:
        return None, "no stop detected" if roi is not None else "roi not drawn"
    i, j = ctx["stops"][-1]
    gaps = [gap_to(f["box"], roi) for f in ctx["track"][i:j + 1]]
    gap = float(np.median(gaps))
    lo, hi = p.get("min_mm"), p.get("max_mm")
    if lo is not None and gap <= lo:
        return True, f"stopped {gap:.0f} mm from {roi['name']} (on it; limit >{lo:.0f})"
    if hi is not None and gap > hi:
        return True, f"stopped {gap:.0f} mm from {roi['name']} (limit {hi:.0f})"
    return False, f"stopped {gap:.0f} mm from {roi['name']}"


def check_exceed(rule, ctx):
    """A named part crossed to the far side of a line."""
    roi, p = ctx["roi"](rule), rule["params"]
    if roi is None:
        return None, "roi not drawn"
    part = p.get("part", "body")
    ref = signed_side(parts_of(ctx["track"][0]["box"])["body"], roi["world_mm"]).mean()
    side = math.copysign(1.0, ref) if abs(ref) > 1e-9 else 1.0
    # "Deduct every time" in the DSR means per occurrence, not per frame. A
    # bumper held over a line for four seconds is one fault an examiner would
    # write down once; counting frames turns it into a hundred and the score
    # stops meaning anything. So episodes are counted: contiguous runs of
    # frames over the line, separated by the car coming back behind it.
    over_by, episodes, was_over, worst = [], 0, False, 0.0
    for f in ctx["track"]:
        pts = parts_of(f["box"]).get(part)
        if pts is None:
            pts = wheel_pts(f, part)
        if pts is None:
            return None, f"part {part!r} is not something this pipeline produces"
        d = float((-side * signed_side(pts, roi["world_mm"])).max())
        now = d > CROSS_TOL_MM
        if now and not was_over:
            episodes += 1
        if now:
            worst = max(worst, d)
            over_by.append(d)
        was_over = now
    if not episodes:
        return False, f"{part} stayed behind {roi['name']}"
    n = episodes if p.get("every_time") else 1
    times = f"{episodes} time(s)" if p.get("every_time") else "at least once"
    return n, (f"{part} crossed {roi['name']} {times}, by up to {worst:.0f} mm "
               f"over {len(over_by)} frame(s)")


def check_roll(rule, ctx):
    """The car moved after it had come to rest."""
    if not ctx["stops"]:
        return None, "no stop detected"
    i, j = ctx["stops"][-1]
    ref = ctx["track"][i]["centre"]
    d = max(float(np.linalg.norm(f["centre"] - ref)) for f in ctx["track"][i:j + 1])
    lim = rule["params"]["max_mm"]
    return (d > lim), f"rolled {d:.0f} mm while stopped (limit {lim:.0f})"


def check_leave_within(rule, ctx):
    """The stop lasted longer than the rule allows."""
    if not ctx["stops"]:
        return None, "no stop detected"
    i, j = ctx["stops"][-1]
    held = ctx["track"][j]["t"] - ctx["track"][i]["t"]
    lim = rule["params"]["seconds"]
    return (held > lim), f"held the station {held:.1f} s (limit {lim:.0f} s)"


def check_parallel(rule, ctx):
    """At rest, too far from the kerb or not lined up with it."""
    roi, p = ctx["roi"](rule), rule["params"]
    if roi is None or not ctx["stops"]:
        return None, "no stop detected" if roi is not None else "roi not drawn"
    i, j = ctx["stops"][-1]
    f = ctx["track"][(i + j) // 2]
    gap = gap_to(f["box"], roi)
    _, u = line_of(roi["world_mm"])
    kerb = math.degrees(math.atan2(u[1], u[0]))
    off = abs((f["heading"] - kerb + 90.0) % 180.0 - 90.0)
    bad = gap > p["max_mm"] or off > p["max_deg"]
    return bad, f"{gap:.0f} mm from {roi['name']}, {off:.1f} deg off parallel " \
                f"(limits {p['max_mm']:.0f} mm, {p['max_deg']:.0f} deg)"


def check_straight_before_reverse(rule, ctx):
    """Heading wandered on the approach, before the car began to reverse.

    Heuristic, and labelled as one. Reversing is taken as the run of frames
    where the centre moves against the car's own heading; everything before the
    first such run is the approach. What counts as "not straight" is the spread
    of heading over it, which is a judgement the DSR does not put a number on —
    so ``max_deg`` lives in the rules file, not in this code.
    """
    track = ctx["track"]
    if len(track) < 4:
        return None, "too few frames"
    rev = None
    for k, (a, b) in enumerate(pairwise(track)):
        v = b["centre"] - a["centre"]
        if float(np.linalg.norm(v)) < 30.0:
            continue
        h = math.radians(a["heading"])
        if float(v @ np.array([math.cos(h), math.sin(h)])) < 0:
            rev = k
            break
    if rev is None or rev < 2:
        return None, "no reversing phase found on this track"
    head = np.unwrap(np.radians([f["heading"] for f in track[:rev]]))
    spread = math.degrees(float(head.max() - head.min()))
    lim = rule["params"]["max_deg"]
    return (spread > lim), f"heading varied {spread:.1f} deg on the approach " \
                           f"(limit {lim:.0f} deg, heuristic)"


def _tyre(ctx):
    return ctx.get("tyre_width_mm") or 0.0


def check_dwell(rule, ctx):
    """Every wheel in the group sat on the line for long enough — or not long enough."""
    roi, p = ctx["roi"](rule), rule["params"]
    if roi is None:
        return None, "roi not drawn"
    part, tyre = p.get("part", "rear_wheels"), _tyre(ctx)

    def all_on(f):
        pts = wheel_pts(f, part)
        return pts is not None and all(on_line(q, roi, tyre)[0] for q in pts)

    spans = runs_where(ctx["track"], all_on)
    held = [ctx["track"][j]["t"] - ctx["track"][i]["t"] for i, j in spans]
    best = max(held) if held else 0.0
    want, cmp_ = p["seconds"], p.get("cmp", "ge")
    if cmp_ == "ge":
        # A pass condition: the fault is failing to achieve it.
        return (best < want), (f"both {part} held {roi['name']} for {best:.1f} s "
                               f"(needs {want:.0f} s)")
    if not spans:
        return False, f"{part} never sat on {roi['name']}"
    short = [h for h in held if h < want]
    if not short:
        return False, f"{part} held {roi['name']} for {best:.1f} s, over the {want:.0f} s mark"
    n = 1 if p.get("once") else len(short)
    return n, (f"{part} sat on {roi['name']} for {max(short):.1f} s, under the "
               f"{want:.0f} s mark, on {len(short)} occasion(s)")


def check_dwell_one(rule, ctx):
    """Exactly one wheel of the pair touched the line — the other missed it."""
    roi, p = ctx["roi"](rule), rule["params"]
    if roi is None:
        return None, "roi not drawn"
    part, tyre = p.get("part", "rear_wheels"), _tyre(ctx)

    def lone(f):
        pts = wheel_pts(f, part)
        if pts is None:
            return False
        return sum(on_line(q, roi, tyre)[0] for q in pts) == 1

    spans = runs_where(ctx["track"], lone)
    if not spans:
        return False, f"never just one of {part} on {roi['name']}"
    n = 1 if p.get("once") else len(spans)
    return n, (f"only one of {part} was on {roi['name']}, on {len(spans)} occasion(s)")


def check_wheel_on_line(rule, ctx):
    """A wheel reached the line, and the car is square enough to it."""
    roi, p = ctx["roi"](rule), rule["params"]
    if roi is None or not ctx["stops"]:
        return None, "no stop detected" if roi is not None else "roi not drawn"
    i, j = ctx["stops"][-1]
    f = ctx["track"][(i + j) // 2]
    pts = wheel_pts(f, p.get("part", "right_wheels"))
    if pts is None:
        return None, "this track carries no wheel positions"
    tyre = _tyre(ctx)
    touching = [on_line(q, roi, tyre) for q in pts]
    off = _angle_to(f["heading"], roi)
    hit = any(t[0] for t in touching)
    return (hit and off <= p["max_deg"]), (
        f"{'a wheel on' if hit else 'no wheel on'} {roi['name']} "
        f"(nearest {min(t[1] for t in touching):.0f} mm), {off:.1f} deg off it")


def check_wheel_short_of_line(rule, ctx):
    """A wheel never reached the line, or the car finished badly askew."""
    roi, p = ctx["roi"](rule), rule["params"]
    if roi is None or not ctx["stops"]:
        return None, "no stop detected" if roi is not None else "roi not drawn"
    i, j = ctx["stops"][-1]
    f = ctx["track"][(i + j) // 2]
    pts = wheel_pts(f, p.get("part", "right_wheels"))
    if pts is None:
        return None, "this track carries no wheel positions"
    tyre = _tyre(ctx)
    near = min(on_line(q, roi, tyre)[1] for q in pts)
    off = _angle_to(f["heading"], roi)
    short = near > (tyre or 0.0) / 2.0
    return (short or off > p["max_deg"]), (
        f"nearest wheel {near:.0f} mm from {roi['name']}, {off:.1f} deg off it "
        f"(limit {p['max_deg']:.0f} deg)")


def check_on_roi(rule, ctx):
    """A given number of wheels were on a region at once — a kerb, usually."""
    roi, p = ctx["roi"](rule), rule["params"]
    if roi is None:
        return None, "roi not drawn"
    want, tyre = p.get("count", 1), _tyre(ctx)

    def n_on(f):
        pts = wheel_pts(f, "wheels")
        return 0 if pts is None else sum(on_line(q, roi, tyre)[0] for q in pts)

    # "two wheels on the kerb" and "one wheel on the kerb" are different faults
    # in the DSR, so this asks for exactly the count, not at least it — the
    # heavier rule would otherwise fire alongside the lighter one every time.
    spans = runs_where(ctx["track"], lambda f: n_on(f) == want)
    if not spans:
        return False, f"never exactly {want} wheel(s) on {roi['name']}"
    return len(spans), f"{want} wheel(s) on {roi['name']} on {len(spans)} occasion(s)"


def _angle_to(heading_deg, roi) -> float:
    """How far off parallel the car is from an ROI line, 0-90 degrees."""
    _, u = line_of(roi["world_mm"])
    return abs((heading_deg - math.degrees(math.atan2(u[1], u[0])) + 90.0) % 180.0 - 90.0)


CHECKS = {
    "stop_band": check_stop_band, "exceed": check_exceed, "roll": check_roll,
    "leave_within": check_leave_within, "parallel": check_parallel,
    "straight_before_reverse": check_straight_before_reverse,
    "dwell": check_dwell, "dwell_one": check_dwell_one,
    "wheel_on_line": check_wheel_on_line,
    "wheel_short_of_line": check_wheel_short_of_line, "on_roi": check_on_roi,
}

#: What this pipeline can supply. A rule needing anything outside this set is
#: reported unevaluated rather than guessed at. `wheels` is the big one: the
#: pipeline tracks a body box and nothing inside it, so every rule written about
#: a wheel is out of reach until a wheel position exists to test.
BASE_CAPABILITIES = {"box", "heading", "time"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--track", required=True, type=Path, help="Detection CSV from a pipeline.")
    ap.add_argument("--rules", required=True, type=Path)
    ap.add_argument("--calibration", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None, help="Write the scorecard as JSON.")
    ap.add_argument("--station", default=None, help="Only rules for this station, plus GEN.")
    ap.add_argument("--roi-distorted", action="store_true")
    ap.add_argument("--tyre-width-mm", type=float, default=None,
                    help="Tread width, so a rule can ask whether a tyre is ON a line "
                         "rather than whether its centre point is. Without it a wheel "
                         "is a point and only an exact hit counts.")
    a = ap.parse_args()

    spec = json.loads(a.rules.read_text())
    track, has_wheels = load_track(a.track)
    cal = pipeline.load_calibration(a.calibration)
    caps = set(BASE_CAPABILITIES) | ({"wheels"} if has_wheels else set())

    drawn = {r["name"]: r for r in spec.get("rois", []) if r.get("points_px")}
    rois = {}
    if drawn:
        tmp = {"rois": list(drawn.values())}
        scratch = a.rules.with_suffix(".rois.tmp.json")
        scratch.write_text(json.dumps(tmp))
        try:
            for r in pipeline.load_rois(scratch, cal["H_field"], cal["K"], cal["D"],
                                        cal["model"], a.roi_distorted):
                rois[r["name"]] = r
        finally:
            scratch.unlink(missing_ok=True)

    rules = [r for r in spec["rules"] if r.get("enabled", True)]
    if a.station:
        rules = [r for r in rules if r["station"] in (a.station, "GEN")]

    stops = stop_runs(track)
    ctx = {"track": track, "stops": stops, "tyre_width_mm": a.tyre_width_mm,
           "roi": lambda rule: rois.get(rule["params"].get("roi"))}

    print(f"track  {a.track}  —  {len(track)} detected frames, "
          f"{track[-1]['t'] - track[0]['t']:.1f} s")
    print(f"stops  {len(stops)}" + "".join(
        f"  [{track[i]['t']:.1f}-{track[j]['t']:.1f}s]" for i, j in stops))
    print(f"rois   {', '.join(sorted(rois)) or '(none drawn)'}")
    print(f"wheels {'yes' if has_wheels else 'NO — every wheel rule scores as blocked'}"
          + (f", tyre {a.tyre_width_mm:.0f} mm wide" if a.tyre_width_mm else
             " (as points; pass --tyre-width-mm to give them width)" if has_wheels else ""))
    print()

    results, deducted, failed = [], 0, False
    for rule in rules:
        missing = [n for n in rule["needs"] if n not in caps]
        if rule["check"] == "manual" or not rule["image_processing"]:
            verdict, why,state = None, "not an image-processing rule", "MANUAL"
        elif missing:
            verdict, why,state = None, f"needs {', '.join(missing)}", "BLOCKED"
        elif rule["check"] not in CHECKS:
            verdict, why,state = None, f"check {rule['check']!r} is not implemented", "BLOCKED"
        else:
            verdict, why = CHECKS[rule["check"]](rule, ctx)
            state = "SKIP" if verdict is None else ("DEDUCT" if verdict else "ok")

        pts = 0
        if verdict:
            raw = str(rule["points"])
            if "Fail" in raw or rule["params"].get("fail"):
                failed = True
                pts = int(raw.split("/")[-1]) if "/" in raw else 0
            elif raw.isdigit():
                pts = int(raw) * (int(verdict) if rule["params"].get("every_time") else 1)
            deducted += pts
        results.append({"id": rule["id"], "status": state, "points_off": pts,
                        "reason": why, "detail": rule["detail"]})
        mark = {"ok": "  ok  ", "DEDUCT": "DEDUCT", "BLOCKED": "BLOCK ",
                "MANUAL": "manual", "SKIP": " skip "}[state]
        off = f"-{pts:<3d}" if pts else "    "
        print(f"  {mark} {rule['id']:<8} {off} {why}")
        if state == "DEDUCT":
            print(f"         {rule['detail']}")

    n_block = sum(1 for r in results if r["status"] == "BLOCKED")
    print(f"\n  deducted {deducted} point(s)" + ("  — TEST FAILED" if failed else ""))
    if n_block:
        print(f"  {n_block} rule(s) could not be checked — the score above is incomplete:")
        for r in results:
            if r["status"] == "BLOCKED":
                print(f"      {r['id']:<8} {r['reason']}")
    if a.out:
        a.out.write_text(json.dumps({
            "track": str(a.track), "rules": str(a.rules),
            "deducted_points": deducted, "failed": failed,
            "unevaluated": n_block, "results": results}, indent=2, ensure_ascii=False) + "\n")
        print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
