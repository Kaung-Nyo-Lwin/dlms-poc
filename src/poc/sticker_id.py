"""PLACEHOLDER — read the vehicle's number off its roof marker.

Not implemented. This file states the problem, proposes a way to solve it, and
gives `score.py` an API to call once it exists. Nothing here runs.

WHY IT IS NEEDED
    A test is only valid if the car visited every station in order. The
    pipelines track *a* marker through *one* station's camera; they never say
    which car it was, so a run assembled from several stations cannot tell a
    car that skipped B4 from a car that was never seen there. DSR "Driving to
    stations out of order" (GENR1, 35 points) is exactly this question.

WHY NOT MATCH AGAINST EVERY STICKER
    The obvious approach — hold one template per car and match them all at
    every station — costs a full rotation-bank NCC pass per candidate. On the
    measured numbers that is ~305 ms per template per frame while tracking, so
    twenty cars is ~6 s per frame. It also gets *less* reliable as the fleet
    grows: twenty near-identical roof markers give twenty chances for the best
    correlation peak to land on the wrong one, and a mis-identification is
    worse than no identification because it silently reassigns a fault to
    another candidate's scorecard.

THE PROPOSAL: identify once, then track
    Identity is not a per-frame property. A car keeps its number for the whole
    run, so it is worth paying for once and carrying:

    1. **Detect the marker as now.** Unchanged — one template for the *marker
       design*, not per car. This already gives sub-pixel pose.

    2. **Rectify the number patch from that pose.** The marker's pose puts the
       number field at a known offset in the template's own frame, so the patch
       can be warped flat at a fixed mm/px — the same trick `sticker` uses to
       cut a template. Read it upright and to scale rather than obliquely.

    3. **Read the digits, not the sticker.** Print the number as a small
       fixed-length digit field on the marker. Then this is a ten-class problem
       on isolated glyphs at known positions, not a fleet-sized matching
       problem: cost is constant in the number of cars. A per-digit NCC against
       ten rendered glyphs is enough and needs no training; the classic
       seven-segment or a plain sans face works better than a decorative one.

    4. **Vote across frames.** Any single read can be wrong. Accumulate per-digit
       scores over the frames the car is in view and take the winner, with the
       margin between first and second place as the confidence. A whole run
       gives hundreds of looks at the same number.

    5. **Report a margin, never a guess.** Below a confidence floor, return
       `None` and let the scorecard say "vehicle not identified" — the same
       discipline `score.py` applies to rules it cannot check.

    The cost is one warp and a handful of small correlations per frame, and it
    does not grow with the fleet.

    Encoding the number as a machine-readable pattern instead of digits — an
    ArUco/AprilTag field, or a ring of blocks around the marker — is strictly
    easier to read and carries its own error detection. Prefer that if the
    markers can be reprinted; the digit route exists for markers that already
    carry a human-readable number and cannot be changed.

WHAT IT WOULD NEED MEASURING
    * where the number field sits in the marker's frame, in mm (offset and size)
    * the digit height in mm, and how many digits
    * one rendered reference glyph per digit at the working mm/px
"""

from __future__ import annotations


def read_sticker_id(frame, marker_pose, spec) -> tuple[str | None, float]:
    """Vehicle number from one frame, and the confidence of the read.

    Returns ``(None, 0.0)`` until implemented. Callers must treat a None as
    "unknown", never as "no fault".
    """
    raise NotImplementedError(
        "sticker number reading is not implemented; see the module docstring for "
        "the proposed approach and the measurements it needs"
    )


def vote(reads) -> tuple[str | None, float]:
    """Best number over many frames, with the margin over the runner-up."""
    raise NotImplementedError("see read_sticker_id")
