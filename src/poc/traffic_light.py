"""Read the signal aspect from the camera that faces it.

    python traffic_light.py split --image roi/traffic_light.jpg \
                                  --lit red,amber,green --out roi
    python traffic_light.py lamps --image $S/signal.png --out $S/lamps.json
    python traffic_light.py check --lamps roi/traffic_light.lamps.json \
                                  roi/traffic_light_red.png roi/traffic_light_amber.png \
                                  roi/traffic_light_green.png
    python traffic_light.py video --video $S/signal.mp4 --lamps $S/lamps.json \
                                  --out $S/spans.csv

WHY IT IS NEEDED
    DSR B8R1, "Running a red light (Whole car passes)", is 35 points. Deciding
    it needs two facts that live in different places: *when the car crossed the
    line*, which the B8 tracking camera already gives to the frame, and *what
    the signal was showing at that moment*, which nothing else here reads.

WHY IT IS EASY, AND WHERE THE DIFFICULTY ACTUALLY IS
    The signal does not move and the camera does not move, so there is no search
    and no pose: a lamp is a fixed box, surveyed once, and the whole decision is
    a colour statistic inside it. What is *not* easy is everything around that —
    picking a rule that survives the weather, and reconciling two clocks. Both
    are handled below, and the second one is not finished.

THE DECISION: chroma and hue, not brightness
    A lit lamp is *coloured*; an unlit one is dark grey behind a tinted lens. So
    a pixel counts for a lamp when it is both coloured enough and the right hue,
    and a lamp's response is the fraction of its box that qualifies.

    "Coloured enough" is measured as **chroma** — ``max(R,G,B) - min(R,G,B)`` —
    rather than HSV's own S channel. That is not a detail. S is ``(max-min)/max``,
    so it *rises* as a pixel goes dark, and the unlit lenses in
    ``roi/traffic_light.jpg`` sit at S = 59..96 with V = 21..34: an S floor
    alone calls them saturated. Chroma is 6..10 on those same pixels and
    101..176 on the lit ones, more than an order of magnitude apart, which is
    the gap a threshold wants to sit in. Chroma also degrades the right way: a
    dim lamp at dusk stays coloured, while an overcast sky that lifts every V
    does not become coloured.

    Brightness on its own is no good at all, which is the trap worth naming: at
    night the lamp is the brightest thing in frame and at noon it is not.

RANKING, NOT THREE THRESHOLDS
    Each lamp is scored against its own hue band, then the aspects are *ranked*
    and the winner must both clear a floor and beat the runner-up by a margin.
    An amber that responds half as strongly as the red beside it is a bounce off
    a wet road, not an aspect. Anything short of a clear win is ``"unknown"``,
    and `score.py` must render that as "not evaluated" — never as "no fault",
    which would silently pass every car that ran the light while the sun was on
    the lens.

WHAT WAS MEASURED, AND ON WHAT
    ``DEFAULT_SPEC`` below was measured on ``roi/traffic_light.jpg``, which is a
    **stock photograph of three signal heads, not this site's lamps**. `split`
    cuts it into one still per aspect and `check` scores all three; per lamp
    box, inset 22% to keep the housing rim out:

        aspect   hue p5..p95   chroma p50   response   the same box, unlit
        red        173..179        176        0.931     chroma p50 6,  p95 11
        amber        6..22         156        0.652     chroma p50 10, p95 17
        green       81..85         101        0.780     chroma p50 8,  p95 14

    Nine boxes over the three stills, three of them lit, and every lit one wins
    its own aspect by 0.65 or better against a floor of 0.10 — the two unlit
    boxes beside it respond 0.000 every time. That says the *rule* works.

    Amber's 0.652 is the red/amber band edge, and it is worth reading. 92.6% of
    that lamp is coloured, but 29.5% of it sits below hue 11, outside amber's
    band, and is charged to nobody: a box only ever answers for its own colour,
    so hue that falls outside a band is *lost*, never awarded to the aspect next
    door. The edge costs a lit lamp some response and can never invent one, and
    the whole loss still leaves amber leading red 0.652 to 0.000.

    None of this says these bands fit the site: LED matrices photographed at
    forty metres through a wet windscreen are not a studio shot of a lens.
    `check` prints the hue and chroma it measured for exactly this reason —
    point it at stills of each aspect actually lit here and read the site's own
    numbers off it, then put them in the ``spec`` block of ``lamps.json``.

    The known failure is overexposure. A close LED that clips to white has no
    chroma at its core and reads as unlit; the coloured fringe around it may
    still carry the response, but ``chroma_min`` is the knob and a lamp box that
    reads 0.0 while plainly lit is that failure, not a hue error.

RAW PIXELS, NOT UNDISTORTED ONES
    Everything else in this folder works on the undistorted frame, because it
    measures geometry and a homography is only defined there. Nothing here
    measures geometry: no coordinate leaves this file, no lamp box is ever
    compared against an ROI or a car box. Undistorting would cost a full-frame
    remap per frame and make the signal camera need intrinsics it may not have,
    to move a box that is read in its own frame either way. So lamp boxes are
    stored in **raw camera pixels**, and the file says so.

THE PART THAT IS NOT DONE: one clock
    The signal camera and the B8 tracking camera are separate streams, so their
    timestamps must be reconciled before an aspect can be attributed to a
    crossing. **An unreconciled offset is the whole risk in this file** — a car
    at 30 km/h covers 8 metres in a second, so a clock a second out puts the
    crossing on the wrong side of the change, and nothing downstream would show
    it. Record both against one wall clock at capture; failing that, a
    synchronising event visible to both (the signal itself changing, if it is in
    both fields of view) fixes the offset once. :func:`aspect_at` takes a time in
    the *signal* stream's own clock and cannot know about any offset, so
    whatever converts a crossing time into that clock is the thing to get right.
"""

from __future__ import annotations

import argparse
import csv
import json
from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np

#: Hue bands per aspect, in OpenCV's 0-179 half-degree scale, as a list of
#: closed intervals — a list because red straddles 0 and needs two. Measured on
#: ``roi/traffic_light.jpg``; see the module docstring for why these are a
#: starting point and not a site calibration. Amber's lower edge and red's upper
#: edge are deliberately left adjacent rather than overlapping: an orange lamp
#: does put a tail of pixels below hue 10, and the ranking is what settles it.
BANDS = {
    "red": [(170, 179), (0, 10)],
    "amber": [(11, 35)],
    "green": [(40, 95)],
}

#: The whole decision, in five numbers.
#:
#: ``chroma_min``    a pixel this colourful counts; 60 sits between the 17 an
#:                   unlit lens reaches and the 101 the weakest lit one gives.
#: ``min_response``  a winning lamp must fill this much of its box. Well under
#:                   the 0.65 a lit lamp actually gives, because a box clicked
#:                   generously, or a lamp half behind a pole, still has to win.
#: ``min_margin``    and it must beat the runner-up by this much.
#: ``debounce``      consecutive frames before a state change is believed.
#:                   Signals hold for seconds; a one-frame flip is a bulb
#:                   flicker, a passing roof, or a compression artefact.
#: ``every``         look at every Nth frame. Coarsens the transition times by
#:                   the same factor, so leave it at 1 when the timing matters.
DEFAULT_SPEC = {
    "bands": BANDS,
    "chroma_min": 60,
    "min_response": 0.15,
    "min_margin": 0.10,
    "debounce": 3,
    "every": 1,
}

#: What the signal can be showing. ``unknown`` is a real answer, not a failure.
ASPECTS = ("red", "amber", "green")
#: "yellow" is what the DSR table calls amber; one name reaches the code.
ALIASES = {"yellow": "amber", "orange": "amber"}


def resolve_spec(spec: dict | None) -> dict:
    """Defaults, overridden by whatever the site measured."""
    out = dict(DEFAULT_SPEC)
    out.update(spec or {})
    out["bands"] = {k: [tuple(b) for b in v] for k, v in out["bands"].items()}
    return out


def canonical(aspect: str) -> str:
    a = str(aspect).strip().lower()
    return ALIASES.get(a, a)


# --------------------------------------------------------------------------
# one lamp, one frame
# --------------------------------------------------------------------------


def crop(frame, box_px):
    """The pixels inside a lamp box, clipped to the frame."""
    x, y, w, h = (round(v) for v in box_px)
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return frame[y0:y1, x0:x1]


def lamp_response(patch, band, chroma_min):
    """How much of one lamp's box shows that lamp's colour, and what it measured.

    Returns ``(response, hue, chroma)``: the fraction of the box that is both
    coloured and in band, the median hue *of the coloured pixels only* — hue is
    quantisation noise on a near-grey pixel and averaging it in would report a
    number that means nothing — and the median chroma over the whole box, which
    is the one to look at when a lamp that is plainly lit responds 0.
    """
    if patch is None or patch.size == 0:
        return 0.0, -1.0, 0.0
    hue = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)[..., 0].astype(np.int16)
    chroma = patch.max(axis=2).astype(np.int16) - patch.min(axis=2).astype(np.int16)
    coloured = chroma >= chroma_min
    in_band = np.zeros(hue.shape, dtype=bool)
    for lo, hi in band:
        in_band |= (hue >= lo) & (hue <= hi)
    med_hue = float(np.median(hue[coloured])) if coloured.any() else -1.0
    return float((coloured & in_band).mean()), med_hue, float(np.median(chroma))


def lamp_responses(frame, lamp_boxes, spec=None) -> dict[str, tuple[float, float, float]]:
    """Every aspect's response in one frame, keyed by aspect.

    Each box is scored against *its own* aspect's band and nothing else, because
    a box is a lamp and a lamp has one colour. Two boxes may share an aspect —
    a head with both a disc and an arrow — and the stronger one speaks for it.
    """
    s = resolve_spec(spec)
    out: dict[str, tuple[float, float, float]] = {}
    for lamp in lamp_boxes:
        aspect = canonical(lamp["aspect"])
        band = s["bands"].get(aspect)
        if band is None:
            raise SystemExit(f"no hue band for aspect {aspect!r}; known: {', '.join(s['bands'])}")
        got = lamp_response(crop(frame, lamp["box_px"]), band, s["chroma_min"])
        if aspect not in out or got[0] > out[aspect][0]:
            out[aspect] = got
    return out


def read_aspect(frame, lamp_boxes, spec=None) -> tuple[str, float]:
    """Aspect showing in one frame — "red", "amber", "green" or "unknown".

    Returns the winner and its margin over the runner-up. Callers must treat
    "unknown" as unknown, never as permissive.
    """
    s = resolve_spec(spec)
    resp = lamp_responses(frame, lamp_boxes, s)
    ranked = sorted(((v[0], k) for k, v in resp.items()), reverse=True)
    if not ranked:
        return "unknown", 0.0
    best = ranked[0]
    margin = best[0] - (ranked[1][0] if len(ranked) > 1 else 0.0)
    if best[0] < s["min_response"] or margin < s["min_margin"]:
        return "unknown", margin
    return best[1], margin


# --------------------------------------------------------------------------
# a whole clip -> a table of spans
# --------------------------------------------------------------------------


def debounce_spans(reads, debounce, dt=None) -> list[dict]:
    """Per-frame reads -> half-open ``{t_start, t_end, aspect, frames}`` spans.

    ``reads`` is ``(t, aspect, margin)`` in time order. A candidate aspect has
    to hold for ``debounce`` consecutive frames before the state changes, but
    the span it opens is stamped with the time of the run's **first** frame, not
    its last. That distinction is the whole point: the signal changed when it
    changed, and charging the debounce delay to the timestamp would put every
    transition three frames late — 0.12 s at 25 fps, which is a metre of road
    for a car doing 30 km/h, in the one measurement where a metre decides a
    35-point fault.

    Spans are **half-open**, ``t_start <= t < t_end``, because a frame sampled
    at ``t`` is evidence about the interval ``[t, t + dt)`` and not about the
    instant ``t``. So one span ends exactly where the next begins and the shared
    boundary belongs to the *new* aspect — the instant the amber frame was
    captured is amber, not the tail of the red before it. Closed spans would
    hand that instant to whichever span was listed first, which is to say to
    red, which is to say a car crossing on the change would be charged 35 points
    for a light that had already changed. ``dt`` is the frame interval, taken
    from the reads themselves when not given, and only sets the last span's end.
    """
    reads = list(reads)
    if not reads:
        return []
    times = [t for t, _a, _m in reads]
    if dt is None:
        dt = float(np.median(np.diff(times))) if len(times) > 1 else 0.0

    spans: list[dict] = []
    state = "unknown"
    run_aspect, run_t0, run_n = None, 0.0, 0
    for t, aspect, _margin in reads:
        if aspect != run_aspect:
            run_aspect, run_t0, run_n = aspect, t, 1
        else:
            run_n += 1
        if run_n >= debounce and aspect != state:
            spans.append({"t_start": run_t0, "t_end": t, "aspect": aspect, "frames": run_n})
            state = aspect
        elif spans and aspect == state:
            spans[-1]["frames"] += 1
    for span, following in pairwise(spans):
        span["t_end"] = following["t_start"]
    if spans:
        spans[-1]["t_end"] = times[-1] + dt
    return spans


def frame_reads(video, lamp_boxes, spec=None, start_s=None, end_s=None):
    """Yield ``(t, aspect, margin)`` for the frames of a clip."""
    s = resolve_spec(spec)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"could not open {video}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        idx = -1
        if start_s:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_s * fps))
            idx = round(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        while True:
            ok, frame = cap.read()
            if not ok:
                return
            idx += 1
            t = idx / fps
            if end_s is not None and t > end_s:
                return
            if idx % s["every"]:
                continue
            aspect, margin = read_aspect(frame, lamp_boxes, s)
            yield t, aspect, margin
    finally:
        cap.release()


def aspect_spans(video, lamp_boxes, spec=None) -> list[dict]:
    """Debounced ``{t_start, t_end, aspect}`` spans for a whole clip.

    Times are in the *signal* stream's own clock. Nothing here knows the offset
    to the tracking camera's clock; see the module docstring.
    """
    s = resolve_spec(spec)
    reads = list(frame_reads(video, lamp_boxes, s,
                             s.get("start_s"), s.get("end_s")))
    return debounce_spans(reads, s["debounce"])


def aspect_at(spans, t: float) -> str:
    """The aspect showing at a moment, or "unknown" outside every span.

    Half-open, so a moment on a boundary is the new aspect; see
    :func:`debounce_spans`. "unknown" means the signal was not read at that
    moment — before the clip, after it, or across a stretch that never settled —
    and a rule scored against it must be reported unevaluated, never passed.
    """
    for span in spans:
        if span["t_start"] <= t < span["t_end"]:
            return span["aspect"]
    return "unknown"


# --------------------------------------------------------------------------
# surveying the lamp boxes
# --------------------------------------------------------------------------


def load_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"could not read {path}")
    return img


def save_json(path: Path, data: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2) + "\n")


def load_lamps(path: Path) -> tuple[list[dict], dict]:
    """Lamp boxes and the site's own spec, if the file carries one.

    Bands and thresholds belong beside the boxes: they are both properties of
    these lamps under this camera, and splitting them across two files is how a
    site ends up scored with another site's numbers.
    """
    raw = json.loads(Path(path).read_text())
    lamps = raw["lamps"] if isinstance(raw, dict) else raw
    if not lamps:
        raise SystemExit(f"{path} has no lamp boxes")
    for lamp in lamps:
        lamp["aspect"] = canonical(lamp["aspect"])
        if len(lamp["box_px"]) != 4:
            raise SystemExit(f"lamp {lamp['aspect']!r}: box_px must be [x, y, w, h]")
    spec = raw.get("spec", {}) if isinstance(raw, dict) else {}
    return lamps, spec


def write_lamps(path: Path, lamps, spec=None) -> None:
    save_json(path, {
        "_comment": "box_px is [x, y, w, h] in RAW camera pixels, not undistorted",
        "lamps": [{"aspect": lamp["aspect"], "box_px": [int(v) for v in lamp["box_px"]]}
                  for lamp in lamps],
        **({"spec": spec} if spec else {}),
    })


def cmd_lamps(a) -> None:
    """Draw a box round each lamp on a still from the signal camera.

    One session, one named shape per lamp: name them ``red``, ``amber`` and
    ``green`` and the names become the aspects. A shape of any kind works — its
    bounding box is what is stored — so a rough polygon round a round lens is
    fine, and is easier to draw than a tight rectangle.

    Draw *inside* the lens, not round the housing. The response is a fraction of
    the box, so every pixel of black rim in it dilutes a lit lamp toward the
    floor for no gain.
    """
    import picker

    image = load_image(a.image)
    shapes = picker.pick_shapes(image, "Box each lamp, named red / amber / green",
                                hint="click round a lens, then g to close it, and name it",
                                open_browser=not a.no_open)
    if not shapes:
        raise SystemExit("cancelled — nothing drawn")

    lamps = []
    for shape in shapes:
        pts = np.array(shape["points_px"], dtype=np.float64).reshape(-1, 2)
        if len(pts) < 2:
            raise SystemExit(f"shape {shape['name']!r} is a single point, not a lamp box")
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        aspect = canonical(shape["name"])
        if aspect not in BANDS:
            raise SystemExit(f"shape {shape['name']!r}: name a lamp {' / '.join(ASPECTS)}")
        lamps.append({"aspect": aspect, "box_px": [x0, y0, x1 - x0, y1 - y0]})
        print(f"  {aspect:6s} {int(x1 - x0):4d} x {int(y1 - y0):3d} px "
              f"at ({int(x0)}, {int(y0)})")
    write_lamps(a.out, lamps)
    print(f"  wrote {a.out}", flush=True)


# --------------------------------------------------------------------------
# cutting a contact sheet into one still per head
# --------------------------------------------------------------------------


def head_columns(image, bg_below=200, min_ink=3):
    """Column ranges holding a signal head, on a plain light background."""
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ink = grey < bg_below
    runs, start = [], None
    on = ink.sum(axis=0) > min_ink
    for i, lit in enumerate([*on, False]):
        if lit and start is None:
            start = i
        elif not lit and start is not None:
            runs.append((start, i))
            start = None
    rows = np.flatnonzero(ink.sum(axis=1) > min_ink)
    if not runs or not len(rows):
        raise SystemExit("found no signal heads; is the background light and plain?")
    return runs, (int(rows[0]), int(rows[-1] + 1))


def cmd_split(a) -> None:
    """Cut a picture of several signal heads into one still per head.

    The only picture of all three aspects that exists so far is a contact sheet:
    three heads side by side, one lit in each. This is how it becomes three
    stills a detector can be scored against, one per aspect.

    Every crop is given the **same** size — the median head, centred on each
    head's own centroid — so one ``lamps.json`` describes all three and the
    check is comparing like with like. The lamp boxes inside it are the crop cut
    in equal thirds and inset, which is what a vertical three-aspect head is.
    A head that is not three equal lamps stacked wants `lamps` instead.

    ``--lit`` is the operator's statement of which head shows what, left to
    right. It is the ground truth, and it has to come from outside: a fixture
    labelled by the detector under test would agree with it by construction.
    """
    image = load_image(a.image)
    lit = [canonical(t) for t in a.lit.split(",")]
    runs, (y0, y1) = head_columns(image, a.bg_below)
    if len(runs) != len(lit):
        raise SystemExit(f"found {len(runs)} head(s) but --lit names {len(lit)}: "
                         f"{', '.join(lit)}")

    w = int(np.median([x1 - x0 for x0, x1 in runs]))
    h = y1 - y0
    a.out.mkdir(parents=True, exist_ok=True)
    stem = a.image.stem
    for (x0, x1), aspect in zip(runs, lit, strict=True):
        cx = (x0 + x1) // 2
        left = min(max(0, cx - w // 2), image.shape[1] - w)
        top = min(max(0, y0), image.shape[0] - h)
        path = a.out / f"{stem}_{aspect}.png"
        cv2.imwrite(str(path), image[top:top + h, left:left + w])
        print(f"  {aspect:6s} head at x {x0}-{x1}  ->  {path}  ({w}x{h} px)")

    lamps, cell = [], h / len(a.aspects)
    for k, aspect in enumerate(a.aspects):
        bx, by = w * a.inset, cell * a.inset
        lamps.append({"aspect": canonical(aspect),
                      "box_px": [bx, k * cell + by, w - 2 * bx, cell - 2 * by]})
    path = a.out / f"{stem}.lamps.json"
    write_lamps(path, lamps)
    print(f"  lamps  {len(lamps)} boxes, {a.inset:.0%} inset, top to bottom: "
          f"{', '.join(canonical(x) for x in a.aspects)}")
    print(f"  wrote {path}", flush=True)


# --------------------------------------------------------------------------
# scoring stills, and reading a clip
# --------------------------------------------------------------------------


def spec_from_args(a, base=None) -> dict:
    """The file's spec, with anything given on the command line on top."""
    spec = dict(base or {})
    for key in ("chroma_min", "min_response", "min_margin", "debounce", "every"):
        value = getattr(a, key, None)
        if value is not None:
            spec[key] = value
    return spec


def _case(text: str) -> tuple[Path, str]:
    """``path`` or ``path=aspect``; without one, the file's stem carries it."""
    path, _, aspect = text.partition("=")
    return Path(path), canonical(aspect or Path(path).stem.rsplit("_", 1)[-1])


def cmd_check(a) -> None:
    """Read labelled stills and report what the detector made of each.

    This is the whole test: stills whose aspect is known, the responses the
    boxes gave, and whether the winner was right. The hue and chroma columns are
    what a site calibration is read off — point this at stills of each aspect
    lit *here*, and the numbers to put in ``lamps.json``'s ``spec`` block are on
    the screen.
    """
    lamps, file_spec = load_lamps(a.lamps)
    spec = resolve_spec(spec_from_args(a, file_spec))
    print(f"  {len(lamps)} lamp box(es): "
          + ", ".join(f"{lamp['aspect']} {[int(v) for v in lamp['box_px']]}" for lamp in lamps))
    print(f"  chroma >= {spec['chroma_min']}, response >= {spec['min_response']}, "
          f"margin >= {spec['min_margin']}")
    print("  bands: " + "  ".join(
        f"{k} {'/'.join(f'{lo}-{hi}' for lo, hi in v)}" for k, v in spec["bands"].items()))

    ok = 0
    cases = [_case(t) for t in a.images]
    for path, expect in cases:
        image = load_image(path)
        resp = lamp_responses(image, lamps, spec)
        aspect, margin = read_aspect(image, lamps, spec)
        ok += aspect == expect
        print(f"\n  {path}  ({image.shape[1]}x{image.shape[0]} px), expecting {expect}")
        print(f"    {'aspect':>7} {'response':>9} {'hue':>6} {'chroma':>7}")
        for name in sorted(resp, key=lambda k: -resp[k][0]):
            r, hue, chroma = resp[name]
            print(f"    {name:>7} {r:9.3f} {hue if hue >= 0 else float('nan'):6.0f} "
                  f"{chroma:7.0f}")
        mark = "OK" if aspect == expect else f"WRONG, expected {expect}"
        print(f"    -> {aspect}  margin {margin:.3f}   {mark}")
    print(f"\n  {ok}/{len(cases)} correct", flush=True)
    if ok != len(cases):
        raise SystemExit(1)


def cmd_video(a) -> None:
    """Read a clip into the span table `score.py` queries at the crossing time."""
    import pipeline1_bev as rt  # only for parse_time; nothing else is shared

    lamps, file_spec = load_lamps(a.lamps)
    spec = resolve_spec(spec_from_args(a, file_spec))
    t0, t1 = rt.parse_time(a.start), rt.parse_time(a.end)
    reads = list(frame_reads(a.video, lamps, spec, t0, t1))
    if not reads:
        raise SystemExit(f"no frames read from {a.video}")
    spans = debounce_spans(reads, spec["debounce"])

    counts: dict[str, int] = {}
    for _t, aspect, _m in reads:
        counts[aspect] = counts.get(aspect, 0) + 1
    print(f"  {len(reads)} frame(s), {reads[0][0]:.2f}-{reads[-1][0]:.2f} s")
    print("  per-frame: " + "  ".join(f"{k}x{v}" for k, v in
                                      sorted(counts.items(), key=lambda kv: -kv[1])))
    print(f"  {len(spans)} span(s) after a {spec['debounce']}-frame debounce:")
    for span in spans:
        print(f"    {span['t_start']:8.2f} - {span['t_end']:8.2f} s  {span['aspect']:>7}"
              f"  ({span['frames']} frames)")
    gaps = spans[0]["t_start"] > reads[0][0] if spans else True
    if gaps:
        print("  note: the clip opens on frames that never settled — those read as unknown")
    if a.out:
        with Path(a.out).open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["t_start_s", "t_end_s", "aspect", "frames"])
            for span in spans:
                writer.writerow([f"{span['t_start']:.3f}", f"{span['t_end']:.3f}",
                                 span["aspect"], span["frames"]])
        print(f"  wrote {a.out}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name, fn, help_):
        p = sub.add_parser(name, help=help_, description=fn.__doc__,
                           formatter_class=argparse.RawDescriptionHelpFormatter)
        p.set_defaults(fn=fn)
        return p

    def add_spec(p):
        p.add_argument("--chroma-min", type=int, default=None,
                       help=f"Colourfulness floor, max(RGB)-min(RGB). "
                            f"Default {DEFAULT_SPEC['chroma_min']}.")
        p.add_argument("--min-response", type=float, default=None,
                       help="Fraction of its box a winning lamp must fill.")
        p.add_argument("--min-margin", type=float, default=None,
                       help="How far the winner must lead the runner-up.")
        return p

    p = add("split", cmd_split, "cut a picture of several heads into one still each")
    p.add_argument("--image", required=True, type=Path)
    p.add_argument("--lit", required=True,
                   help="Which aspect is lit in each head, left to right, e.g. red,amber,green.")
    p.add_argument("--out", required=True, type=Path, help="Folder for the stills and lamps.json.")
    p.add_argument("--aspects", type=lambda t: t.split(","), default=list(ASPECTS),
                   help="Lamps down a head, top to bottom. Default red,amber,green.")
    p.add_argument("--inset", type=float, default=0.22,
                   help="Shrink each lamp box by this fraction, to miss the housing rim.")
    p.add_argument("--bg-below", type=int, default=200, help="Grey level the background is above.")

    p = add("lamps", cmd_lamps, "box each lamp on a still from the signal camera")
    p.add_argument("--image", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--no-open", action="store_true",
                   help="Print the picker URL, do not open a browser.")

    p = add_spec(add("check", cmd_check, "read labelled stills — the test"))
    p.add_argument("images", nargs="+",
                   help="Stills as PATH or PATH=aspect. Without one, the last _word "
                        "of the filename is taken as the truth.")
    p.add_argument("--lamps", required=True, type=Path)

    p = add_spec(add("video", cmd_video, "read a clip into debounced aspect spans"))
    p.add_argument("--video", required=True, type=Path)
    p.add_argument("--lamps", required=True, type=Path)
    p.add_argument("--out", type=Path, default=None, help="Write the spans as CSV.")
    p.add_argument("--debounce", type=int, default=None,
                   help=f"Frames a change must hold. Default {DEFAULT_SPEC['debounce']}.")
    p.add_argument("--every", type=int, default=None, help="Look at every Nth frame.")
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)

    a = ap.parse_args()
    print(f"{a.cmd}", flush=True)
    a.fn(a)


if __name__ == "__main__":
    main()
