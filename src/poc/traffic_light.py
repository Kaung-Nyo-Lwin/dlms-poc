"""PLACEHOLDER — read the signal aspect from the camera that faces it.

Not implemented. This file states the problem, proposes a way to solve it, and
gives `score.py` an API to call once it exists. Nothing here runs.

WHY IT IS NEEDED
    DSR B8R1, "Running a red light (Whole car passes)", is 35 points. Deciding
    it needs two facts that live in different places: *when the car crossed the
    line*, which the B8 tracking camera already gives to the frame, and *what
    the signal was showing at that moment*, which nothing currently reads.

THE PROPOSAL: a fixed window, a hue decision, and a shared clock
    The signal does not move and the camera does not move, so this is far easier
    than the tracking problem — it is a fixed-region classification, not a
    search.

    1. **Survey the lamp boxes once.** Click a small box around each lamp on a
       still, exactly as `roi` clicks ground features. Store them in pixels in
       the undistorted frame, so the same rectification everything else uses
       applies here too. Three boxes, or one box per aspect the signal has.

    2. **Decide on hue and saturation, not brightness.** A lit lamp is
       saturated and coloured; an unlit one is dark and grey. Convert the patch
       to HSV, take the fraction of pixels that are both saturated above a floor
       and inside a hue band, and call the aspect lit when that fraction crosses
       a threshold. Red wraps the hue circle and needs two bands. Deciding on
       hue survives the exposure changes that a brightness rule does not — and
       at night the lamp is the brightest thing in frame, at noon it is not.

    3. **Rank the aspects rather than thresholding each.** Take the aspect with
       the strongest response and require it to beat the runner-up by a margin.
       An amber that is half the strength of the red beside it is a reflection,
       not an aspect. Emit `unknown` when nothing wins clearly — a run scored
       against an unknown signal must be flagged, not guessed.

    4. **Debounce in time.** Signals hold for seconds; a one-frame flip is
       noise, a bulb flicker, or a passing roof. Require N consecutive frames
       before the state changes. This also gives the transition timestamps the
       rule actually needs.

    5. **Tie the clocks together.** The signal camera and the B8 tracking camera
       are separate streams, so their timestamps must be reconciled before an
       aspect can be attributed to a crossing. Record both against one wall
       clock at capture; failing that, a synchronising event visible to both
       (the signal itself changing, if it is in both fields of view) fixes the
       offset once. **An unreconciled offset is the whole risk here** — a car at
       30 km/h covers 8 metres in a second, so a clock a second out can put the
       crossing on the wrong side of the change, and nothing downstream would
       show it.

    The output is a small table of `(t_start, t_end, aspect)` spans, which
    `score.py` can query at the crossing time the tracking already knows.

WHAT IT WOULD NEED MEASURING
    * a pixel box per lamp, clicked on an undistorted still from the signal camera
    * the hue band and saturation floor for this site's lamps, sampled from
      footage of each aspect actually lit — not assumed from the colour names
    * the clock offset between the signal stream and the station stream
"""

from __future__ import annotations


def read_aspect(frame, lamp_boxes, spec) -> tuple[str, float]:
    """Aspect showing in one frame — "red", "amber", "green" or "unknown".

    Returns the winner and its margin over the runner-up. Callers must treat
    "unknown" as unknown, never as permissive.
    """
    raise NotImplementedError(
        "traffic-light reading is not implemented; see the module docstring for "
        "the proposed approach and the measurements it needs"
    )


def aspect_spans(video, lamp_boxes, spec) -> list[dict]:
    """Debounced ``{t_start, t_end, aspect}`` spans for a whole clip."""
    raise NotImplementedError("see read_aspect")


def aspect_at(spans, t: float) -> str:
    """The aspect showing at a moment, or "unknown" outside every span."""
    raise NotImplementedError("see read_aspect")
