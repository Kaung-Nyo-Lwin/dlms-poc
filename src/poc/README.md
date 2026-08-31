# poc — survey a site, then track a car through it

Flat, dependency-light versions of the whole path, written to be read start to
finish. Every file runs on its own: **numpy + opencv + the standard library,
nothing imports `dlms`.**

| file | what it does |
|---|---|
| `calibrate.py` | the survey tool — every step, one folder of files |
| `picker.py` | the browser picking widget the survey steps use |
| `pipeline1_bev.py` | warp each frame to the car plane, then match |
| `pipeline2_raw.py` | match in the camera frame, then map the centre |
| `render.py` | draw either pipeline's CSV back onto the video |
| `traffic_light.py` | read the signal aspect from the camera that faces it |
| `score.py` | take a track and `dsr_rules.json`, and deduct points |

There are no stations, no vehicle ids and no workspace. Every input and output is
a path you choose, so a whole site is one directory.

## 1. Survey

Run the steps in order; each writes a file the next one reads.

```sh
# from the repo root; `python` here means `.venv/bin/python` unless it is activated
S=poc/B4                      # one folder per station
V=raw/B4/B4_T2_near.mp4       # the clip this station is surveyed and tracked on

# The lens belongs to the camera, not to the station: calibrate it once and
# point every station at the same file.
python src/poc/calibrate.py intrinsics --video intrinsics/checkerboard.mp4 \
    --board 11x7 --square-mm 30 --model fisheye \
    --out intrinsics/test_harvest_fish.json

# Two stills, because the two halves of the survey want different pictures: the
# ground marks have to be visible for `gcp` and `roi`, and the car has to be
# parked in the station for `sticker` and `outline`.
python src/poc/calibrate.py frame --video $V --at 0  --out $S/gcp.png
python src/poc/calibrate.py frame --video $V --at 22 --out $S/car.png

python src/poc/calibrate.py gcp --image $S/gcp.png \
    --intrinsics intrinsics/test_harvest_fish.json \
    --out $S/calibration.json

python src/poc/calibrate.py sticker --image $S/car.png \
    --calibration $S/calibration.json \
    --height-mm 1450 --out $S/sticker.png

python src/poc/calibrate.py roi --image $S/gcp.png \
    --calibration $S/calibration.json --out $S/roi.json

python src/poc/calibrate.py outline --image $S/car.png \
    --calibration $S/calibration.json \
    --sticker-height-mm 1450 --front-height-mm 160 --side-height-mm 0 \
    --length-mm 4630 --width-mm 1730 --side \
    --marker-corners \
    --template $S/sticker.png --template-mm-per-px 4.07 \
    --front-to-wheel-mm 900 --rear-to-wheel-mm 850 --tyre-width-mm 195 \
    --out $S/car.json
```

`sticker` runs before `outline`, because `outline --template` matches the cut
template against the survey frame to measure the detector offset, and because
`sticker` is what prints the `--template-mm-per-px` the next two steps need. The
numbers above are B4's: a 4630 x 1730 mm car, its marker 1450 mm up, bumper
corners at 160 mm and the sill at ground level.

Two more steps exist and this workflow does not run them. **`measure`** clicks
two ground points and reads the distance back, for a tape to argue with — the
`gcp` residuals only say the fit agrees with itself at the marks it was given,
so this is worth doing before trusting anything downstream. **`carplane`**
surveys the marker's plane from poles standing on known marks, instead of
raising `Z` through the camera pose. Both are checks; neither writes anything
the pipelines require.

Each interactive step prints a `http://127.0.0.1:…` URL and opens it. If your
browser does not open (common under WSL), pass `--no-open` and click the URL.
In the page: **wheel** zoom · **drag** pan · **s** snap on/off · **u** undo ·
**r** reset view · **←↑↓→** nudge ¼ px (Shift for 1 px) · **Enter** save ·
**Esc** cancel. A loupe follows the cursor, and every click can snap to the
nearest sub-pixel corner — which is most of what makes a painted cross
repeatable between operators.

There are no pop-up dialogs anywhere. Text goes in one field in the toolbar,
read only at the moment it is needed: in `roi` and `traffic_light.py lamps` it is
the optional name for the next shape, and in `gcp` it is the world `X, Y` you
type *before* clicking the mark. A dialog can be blocked outright in an
embedded browser such as VS Code's
Simple Browser, and even when it renders it may never take keyboard focus — so
nothing here is allowed to block on one. Finishing a shape with **p** / **l** /
**g** (or the toolbar buttons) is always immediate; an unnamed shape gets
`roi1`, `roi2`, …

Notes on the steps that have a trap in them:

- **`intrinsics`** — `--board` counts *inner corners*, not squares: a board of
  12×8 squares is `11x7`. Two filters decide which views count: blur is rejected,
  and a view is skipped unless the board has actually moved since the last one
  kept. A hundred frames of one pose constrain nothing while making the reported
  RMS look better. Pause at each pose rather than sweeping the board — a swept
  board comes back sheared and pin-sharp, so the blur gate misses it.
  The calibration video must have the **same framing** as the station footage;
  the `gcp` step refuses to mix a 1080×1920 portrait calibration with a
  3840×2160 landscape frame, because scaling across aspect ratios is not valid.
  `--model` picks the lens model and must match the lens: `pinhole` (default)
  or `fisheye`. The pinhole model's two radial terms cannot represent a wide
  fisheye — the fit converges, reports a plausible RMS on the views it was
  given, and then bends straight lines near the frame edge. Every step and both
  pipelines read the `model` field this writes and undistort accordingly, so
  the choice made here follows the calibration everywhere. If straight edges
  stay straight in the `frame` still, it is pinhole.
- **`gcp`** — click a mark, type its world `X, Y` in millimetres. The world frame
  is yours; pick an origin and an axis on the tarmac and stay consistent. Four
  points is the minimum and is exactly determined — six or more spread across
  the working area is what makes the residuals mean anything. A residual over
  50 mm is almost always a mistyped coordinate or a click on the wrong mark.
  **Four coplanar points cannot check themselves**: they determine the
  homography exactly, so the residual measures click jitter and nothing else,
  and it says the same thing on a flat pad and on a slope. Spread them over the
  area the car will actually occupy — a plane fitted to a 2 m square and
  extrapolated 5 m out is extrapolation, whatever the residual says.
- **`carplane`** — *optional, and the one step that measures rather than assumes.*
  Every other path to the marker's plane raises `Z` through the camera pose,
  which is exact only if the pose is. This stands poles at marker height over
  known ground marks and solves the plane from where their tops actually appear.
  Click each pole top and type the world `X, Y` of **the mark beneath it** — the
  displacement between the two is the parallax, and it is what the fit reads the
  camera out of. Because the car plane is parallel to the ground at a known
  height, it has three unknowns rather than a homography's eight, so two targets
  already suffice and three make the residual a real check. The recovered camera
  height is printed next to the pose's, an independent cross-check on both
  surveys — a tape on the mast settles which is right when they disagree. Writes
  a `car` block into `calibration.json`, in place; both pipelines then prefer it
  for a marker at that height and fall back to raising `Z` at any other height.
- **`outline`** — *the car's box, traced instead of taped.* Four clicks going
  round the car — front-left, front-right, rear-right, rear-left — on a
  bird's-eye raster, stored as millimetre offsets from the marker. Replaces
  `car`'s five tape numbers and its nose click; the click order is what fixes
  which end is the front.
  **When the far end is hidden** — which it usually is, since an oblique camera
  sees the near end down to the road but the car's own body covers the rest —
  pass `--length-mm` and click only the two front corners. The rear is built
  square to the front edge at that distance, in world millimetres, so it never
  touches an image and no plane can be wrong back there. Which way is backwards
  comes from the marker, not from a click: it sits inside the footprint, so the
  rearward perpendicular is whichever points towards it.
  **`--side` adds a second pass**: two more clicks, down one flank. The front
  edge is only a car's width long, and a direction taken from it swings the far
  corners sideways in proportion to their distance along the car — which is why
  a survey comes back with good front-to-back numbers and bad left-to-right ones
  from the same clicks. A flank line is four times the baseline, and it pins
  something the front edge cannot: **where the side of the car actually is**, so
  the box stops having to assume the marker sits on the centreline. A marker
  stuck 60 mm off centre otherwise throws every lateral measurement by 60 mm
  with nothing in the file to show it. Measured on a synthetic B7 car with the
  marker 300 mm off centre: 300 mm of corner error without `--side`, **0 mm with
  it**; with a 40 mm click slip on top, 379 mm against 20 mm. Each click then
  does what it is good at — direction and lateral position from the long flank,
  how far forward from the square, hard bumper edge — and the two lines can
  finally check each other, in millimetres at a corner rather than in degrees.
  Pass `--width-mm` too
  — it lays a span of exactly that width half either side of **the marker**
  rather than either side of the clicked midpoint, since the car is symmetric
  about the centreline the marker sits on. A corner clicked wide then costs the
  edge's direction only instead of dragging the whole car sideways by half the
  error; a corner slid 300 mm along the edge leaves the footprint exact. It is
  also compared against the clicked width. That comparison is the **only** check that can see a wrong `--height-mm`
  in this mode, because a built box is square and the right length whatever
  plane it was traced on. The step reports the disagreement as the box error it
  implies (`error = gap * r / width`) rather than as millimetres of width, and
  solves back for the height the clicks actually imply — an 88 mm width gap is
  300 mm of box, which is why a raw width threshold is the wrong unit.
  **The front edge and the flank need not sit at the same height.** A bumper
  corner is near 500 mm, a sill runs lower, a shoulder crease higher — and
  clicking one on the other's plane displaces it. `--front-height-mm` and
  `--side-height-mm` give each line its own raster; both default to
  `--height-mm`, so a command that does not use them behaves exactly as before.
  Nothing needs reconciling afterwards, which is the part worth understanding: a
  raster rectified to height `p` reports where a ray crosses `p`, so a feature
  *at* `p` reads its true horizontal position. Marker, front edge and flank
  therefore arrive as the same world millimetres on the same ground whatever
  heights they were read at, and compose directly. On a synthetic car with
  bumpers at 500 mm and a sill at 300 mm, per-plane clicking returns the box to
  **0.00 mm**; forcing both onto one plane costs 22 mm at 500, 347 mm at 300 and
  589 mm at 160.
  The two mistakes are not equally bad, and provably so. A homology maps a line
  to a *parallel* line, so the direction the flank gives is exact however wrong
  `--side-height-mm` is — 0.000000° over a 600 mm error in testing. All that
  moves is where the flank sits across the car, and the step prints how much per
  10 mm for the geometry in front of it. `--front-height-mm` is the one that
  costs about a millimetre per millimetre, and the width check is what sees it.
  Three levels also make a question askable that two could not answer: how far
  the marker sits from the flank, against half of `--width-mm`. What is left
  over is the marker's own offset from the centreline plus, when the planes
  differ, the body's taper between them — two causes in one number, so it is
  quoted and never corrected.
  Each overlay in the approval view is drawn on the plane that makes it mean
  something. The clicked lines go back on their own levels, where they still
  land on the clicks that made them; **the box is drawn on the lowest of the
  three**. Its corners are horizontal positions carrying no height of their own,
  so putting them on an image is a choice — and the lowest level sets the
  outline on the car's base instead of up at bumper height, where it would stand
  proud of it by exactly the parallax this step exists to remove. The gap you
  see between the box and the front line *is* that parallax. Nothing written
  changes: it moves an overlay, not a millimetre. When the overlays end up on
  different planes the check moves onto the frame, since a flank drawn on the
  front plane's bird's-eye lands where it appears from there rather than where
  it was clicked.
  Nothing is written until you approve it: the step shows the adjusted line and
  the resulting box as two toggleable overlays with the raw clicks marked, and
  cancelling that view re-runs the clicks. `--no-preview` skips the check. **`--height-mm` is the number that matters**: a
  raster rectified to height `p` shows where things *appear* on that plane, so
  a corner at height `h` traced on the wrong one comes back displaced by about
  `|h - p| * r / (H_camera - h)` — roughly a millimetre per millimetre of
  mismatch at B7. Bumper corners sit near 500 mm; tracing them on the marker's
  1450 mm plane returns a 3345 x 1422 mm box for a 4000 x 1700 mm car and a
  footprint 964 mm out. Moving the plane costs nothing in visibility: both
  rasters resample the same undistorted pixels, only the millimetres assigned
  to a click change. The error is a constant in the car's frame, so it rotates
  with the car and tracks it perfectly — it never announces itself, and the
  squareness check cannot see it. **Compare the printed box against the car's
  real length and width; that is the only place a wrong height shows up.**
  **Wheels** come from `--front-to-wheel-mm` and `--rear-to-wheel-mm` — the
  clicked front line back to the front axle, and the box's rear line forward to
  the rear axle — both or neither, plus an optional `--track-mm`. They are not
  clicked: an oblique camera shows the near side behind the sill and the far
  side under the body, so the contact patches are constructed off the box's own
  axes instead. Measuring each end separately retires the `car` step's guess
  that the two overhangs are equal — a front-drive hatchback's front overhang
  runs 100-150 mm longer, which put each axle 50-75 mm out. Leave `--track-mm`
  off and the wheels default to the body width, which puts each one 75-125 mm
  too far out. `--check-wheels` opens a **ground-plane** raster to click
  whatever patches are visible and reports how far each is from the constructed
  one; that is the only independent word on whether the tape numbers are right.
  `--tyre-width-mm` lets a rule ask whether a *tyre* is on a line rather than
  whether its centre point is. Without wheels, ten of the DSR's rules score as
  not-evaluated.
  **`--marker-corners`** changes how the marker centre is found: click its four
  corners and cross the diagonals, rather than clicking the centre directly. A
  corner is the one place on a marker the sub-pixel snap has something to catch;
  the middle of a printed disc is not, and everything the step stores is an
  offset from that point. It costs three extra clicks and turns the centre into
  a measurement with a residual — for a parallelogram the diagonals' crossing
  and the mean of the four corners are the same point, and the gap between them
  is printed and gated by `--max-centre-spread-mm`.
  **`--space frame`** clicks the box lines on the camera frame at native
  resolution instead of on a bird's-eye raster, which is the better view when
  the edge is easier to judge across the plane than down at it.
- **`sticker`** — click roughly where the marker is, then box it on the bird's-eye
  raster that appears. The box readout is in millimetres, so you can check it
  against the printed marker before committing. The step prints the
  `--template-mm-per-px` value the pipelines need (also in `sticker.json`).
- **`car`** — the older tape-measured route to the same file: lengths off a
  tape, orientation from two clicks. `outline` supersedes it and this workflow
  does not use it.
- **`roi`** — click points, then **p** point · **l** line · **g** polygon to
  finish each shape and name it.

Everything the tool records is in the **undistorted** frame, using the original
camera matrix as the projection. That is why the pipelines need no
`--roi-distorted` flag for ROIs surveyed here: the homographies, the ROI pixels
and the frames the pipelines undistort all share one coordinate frame.

## 2. Track

```sh
python src/poc/pipeline1_bev.py --video $V --calibration $S/calibration.json \
    --car $S/car.json --template $S/sticker.png --template-mm-per-px 4.07 \
    --rois $S/roi.json --out $S/detect.csv \
    --start 0:22 --end 0:23 \
    --corner-pnp --detector-offset

python src/poc/render.py --video $V --calibration $S/calibration.json \
    --csv $S/detect.csv --rois $S/roi.json \
    --start 0:22 --end 0:23 --units cm --out $S/B4.mp4
```

**`--corner-pnp` and `--detector-offset` are both on here on purpose**, and each
removes a bias that no amount of averaging will: the marker's tilt, and a
template cut off-centre. Both are explained below, and both are off by default
because each has a precondition — `--corner-pnp` wants the marker ~60 px across,
`--detector-offset` wants `outline --template` to have measured an offset to
read back.

`pipeline2_raw.py` takes the same flags, plus `--ref-mm X,Y` to choose where its
camera-frame crop is synthesised.

**Changing the car or the ROIs does not need a re-detect.** Detection is the
expensive half and none of it depends on the car's geometry or on where the
lines are painted — the CSV already carries where the marker was and which way
it was turned, so a new `car.json` is a rigid transform of the stored polygon by
that pose, and a new `roi.json` is a re-measure of the clearance columns.
`tools/rebox.py` does that, in place:

```sh
python tools/rebox.py --csv $S/detect.csv --out $S/detect.csv \
    --car $S/car.json --rois $S/roi.json \
    --calibration $S/calibration.json --detector-offset
```

It is a scratch helper and lives outside this folder deliberately: the pipelines
stay the only thing that turns a video into a track. Note that a track written
with `--corner-pnp` carries a tilted box, and rebox rebuilds it flat — the
`plane_fit` column says which rows those are, and it warns when it rewrites any.

`render.py` draws on the full-resolution frame and resizes once at the end, so
line widths and text are pre-multiplied by the inverse of `--scale`. If the
labels are still too small for the screen you are reviewing on, raise
`--label-scale` (try `1.5` or `2`); `--units cm` or `mm` changes what the
clearance readout says.

Useful on both: `--start` / `--end` to clip, `--every N` to subsample,
`--margin-mm` to size the search area, and on pipeline 1 `--mm-per-px` for the
raster resolution and `--search-px` for the tracking window.

`--start` and `--end` read seconds, `m:ss` or `h:mm:ss` — the same formats
`render.py` takes, because a clip is nearly always something that was watched
on a scrubber first. `--start` **seeks** rather than decoding up to the mark, so
a late clip costs no more than an early one: on a 61 s 4K file, starting at 55 s
saved 13.8 s of pure decode, and the saving grows with the video. Where the seek
lands is read back from the file rather than computed from the request, since
h264 seeks to a keyframe and a guessed frame index would mislabel every row with
nothing to show for it. `--end` stops the loop instead of skipping to the end.

`--detector-offset` corrects for a template cut off-centre. The matcher reports
the *template's* centre, and every box, wheel and clearance hangs off that — so
a template cut a few pixels out of true puts the whole footprint beside the car,
consistently, in a way no residual can show. `calibrate.py outline --template`
measures the gap on the survey frame and stores it as `detector_offset_mm`;
until now nothing read it back. That search runs at **full resolution**: the
coarse pyramid the runtime uses to find a marker cold is the wrong tool for a
one-shot survey match, and on B1's own frame it locked onto the wall of tyres
behind the car (0.681) in preference to the marker sitting dead centre in the
raster (0.816), reported no match, and silently left the offset unmeasured. A
match landing further from the clicked marker than the template is wide is now
refused rather than stored — that is a lock onto something else, not a template
cut off-centre, and storing one would add a metre of offset to every frame. With no value the flag uses that measurement,
or pass `X,Y` in millimetres to give one directly. It is applied **in the
template's frame**, un-rotated on load by the heading the detector reported when
it was measured and re-rotated by the heading found in each frame — a
world-frame constant would be right only at the pose it was surveyed at and
would swing the wrong way as the car turned. Measured on C6, whose template is
15.8 mm off centre: the footprint moves 15.8 mm, rigidly, with the heading
unchanged.

## 3. The traffic light

B8/C7 is the signal station, and its 35-point rule — "Running a red light (whole
car passes)" — needs two facts that live in different places: *when* the car
crossed the line, which the tracking above already gives to the frame, and *what
the signal was showing then*, which nothing else here reads.

The signal does not move and the camera does not move, so there is no search and
no pose. A lamp is a fixed box, surveyed once, and the whole decision is a colour
statistic inside it.

```sh
python src/poc/traffic_light.py lamps --image $S/signal.png --out $S/lamps.json

python src/poc/traffic_light.py video --video $S/signal.mp4 \
    --lamps $S/lamps.json --out $S/spans.csv
```

`lamps` is one picker session: draw a shape round each lens and name it `red`,
`amber` or `green`, and the names become the aspects — the bounding box is what
is stored, so a rough polygon round a round lens is fine. Draw *inside* the lens
rather than round the housing: the response is a fraction of the box, so every
pixel of black rim in it drags a lit lamp toward the floor for nothing.

Lamp boxes are stored in **raw camera pixels**, which is the one place this
folder does not use the undistorted frame. Nothing here measures geometry — no
coordinate leaves the file and no lamp box is ever compared against an ROI or a
car box — so undistorting would cost a full-frame remap per frame, and make the
signal camera need intrinsics it may not have, to move a box that is read in its
own frame either way.

`video` writes `t_start_s, t_end_s, aspect, frames` after a debounce, since
signals hold for seconds and a one-frame flip is a bulb flicker or a passing
roof. The span opens at the **first** frame of the run that settled it, not the
last: charging the debounce to the timestamp would put every transition three
frames late, which is a metre of road at 30 km/h in the one measurement where a
metre decides a 35-point fault. Spans are half-open, so one ends exactly where
the next begins and a moment on the boundary belongs to the *new* aspect —
`aspect_at(spans, t)` is what a rule calls at the crossing time.

A lit lamp is *coloured* and an unlit one is dark grey behind a tinted lens, so a
pixel counts when its **chroma** — `max(R,G,B) - min(R,G,B)` — clears a floor and
its hue is in that lamp's band. Chroma rather than HSV's S channel, and that is
not a detail: S is `(max-min)/max`, so it *rises* as a pixel goes dark, and the
unlit lenses in the reference picture sit at S = 59..96 with V = 21..34 — an S
floor alone calls them saturated. Their chroma is 6..10 against 101..176 for the
lit ones. Then the aspects are **ranked**, and the winner must clear a floor
*and* beat the runner-up; an amber responding half as strongly as the red beside
it is a bounce off a wet road. Anything short of a clear win is `unknown`, which
`score.py` must render as "not evaluated" and never as "no fault".

### Calibrating it for a site

The bands in `DEFAULT_SPEC` were measured on `roi/traffic_light.jpg`, a stock
photograph of three heads — **not this site's lamps**. `split` cuts a picture
like that into one still per aspect, and `check` scores them:

```sh
python src/poc/traffic_light.py split --image roi/traffic_light.jpg \
    --lit red,amber,green --out roi

python src/poc/traffic_light.py check --lamps roi/traffic_light.lamps.json \
    roi/traffic_light_red.png roi/traffic_light_amber.png roi/traffic_light_green.png
```

| aspect | hue p5..p95 | chroma p50 | response | the same box, unlit |
|---|---|---|---|---|
| red | 173..179 | 176 | 0.931 | chroma p50 6 |
| amber | 6..22 | 156 | 0.652 | chroma p50 10 |
| green | 81..85 | 101 | 0.780 | chroma p50 8 |

3/3, each lit lamp winning its own aspect by 0.65 or better against a floor of
0.10, with the two unlit boxes beside it at 0.000 every time. That says the rule
works. It does not say the bands fit a site — LED matrices at forty metres
through a wet windscreen are not a studio shot of a lens — so `check` prints the
hue and chroma it measured, and a still of each aspect lit *here* is read the
same way. Put the result in `lamps.json`'s `spec` block, which is where a site's
own bands belong: they are a property of these lamps under this camera, exactly
as the boxes are, and splitting the two across separate files is how a site ends
up scored with another site's numbers.

Amber's 0.652 is the red/amber band edge and is worth reading: 92.6% of that
lamp is coloured, but 29.5% of it falls below hue 11, outside amber's band, and
is charged to nobody. A box only ever answers for its own colour, so hue outside
a band is *lost* and never awarded to the aspect next door — the edge can cost a
lit lamp response and can never invent one. The known failure is the opposite
one: a close LED that clips to white has no chroma at its core and reads as
unlit, and `--chroma-min` is the knob for it.

**The unfinished part is the clock.** The signal camera and the tracking camera
are separate streams, and an unreconciled offset is the whole risk here: a car at
30 km/h covers 8 metres in a second, so a clock a second out puts the crossing on
the wrong side of the change with nothing downstream to show it. Record both
against one wall clock at capture, or fix the offset once off an event visible to
both — the signal itself changing, if it is in both fields of view.

## 4. Score

`score.py` takes a track and a **rules file** — which is not `roi.json`. One file
holds both halves of a score: the rules, copied from the `dsr_rules.json` beside
the script, and the regions they are measured against, drawn into it. Keeping
them together is what makes a scorecard reproducible; a rule citing `stop_line`
and a `stop_line` clicked six weeks later in another file are related only by
hope.

```sh
S=poc/B7
V=raw/B7/B7_park.mp4

# Seed B7's rules from the DSR catalogue and draw the regions they cite
python src/poc/calibrate.py rules --image $S/gcp.png \
    --calibration $S/calibration.json --station B7 --out $S/rules.json

# Track. --rois is optional here: score.py reads the regions from rules.json
# itself, so this only adds the per-frame clearance columns and the render overlay
python src/poc/pipeline1_bev.py --video $V --calibration $S/calibration.json \
    --car $S/car.json --template $S/sticker.png --template-mm-per-px 9.41 \
    --rois $S/rules.json --out $S/detect.csv \
    --corner-pnp --detector-offset

python src/poc/score.py --track $S/detect.csv --rules $S/rules.json \
    --calibration $S/calibration.json --tyre-width-mm 195 \
    --out $S/scorecard.json
```

`rules` seeds the file on its first run and draws whatever is still undrawn on
every run after, so it can be stopped and resumed; `--redraw` does the ones
already done, and `--add NAME --type line` appends a region the DSR does not
name. B7 pulls 18 rules — its own plus `GEN` — and six regions between them:
`stop_line`, `front_boundary`, `rear_boundary`, `right_line`, `curb` and
`any_boundary`. Regions are clicked on the raw undistorted frame and read on the
**field** plane, because paint and kerbs are on the ground, which is why the
`gcp` still is the right picture to draw them on rather than the one with the car
parked in the way.

**Finish drawing before pointing the pipeline at the file.** `score.py` skips a
region with no points; the pipeline's `--rois` does not, and a half-drawn rules
file fails inside the clearance arithmetic with `ValueError: min() iterable
argument is empty`, naming no region. Either finish the set, or leave the
pipeline on a plain `roi.json` — scoring does not need it either way.

`--tyre-width-mm` is what lets a rule ask whether a *tyre* is on a line rather
than whether its centre point is, and without wheel columns in the track ten of
the DSR's rules cannot be evaluated at all. `--station` filters the rules to one
station plus `GEN`; a file seeded by `rules` is already filtered, so it is only
useful on a hand-merged one.

**A rule this cannot check is reported, never skipped.** Every rule carries a
`needs` list, and anything missing comes back `BLOCK` with the reason rather than
quietly passing. A scoring engine that silently passes what it cannot see is
worse than one that refuses: the score looks complete and is not. On B7's own
201-frame track:

```
   skip  B7R2          no reversing phase found on this track
    ok   B7R3          front_bumper stayed behind front_boundary
    ok   B7R5          rear_wheels stayed behind rear_boundary
  BLOCK  B7R6          needs handbrake
  DEDUCT B7R9     -15  nearest wheel 2039 mm from right_line, 0.6 deg off it (limit 10 deg)
  manual GENR1         not an image-processing rule
  BLOCK  GENR4         needs multi_vehicle

  deducted 15 point(s)
  2 rule(s) could not be checked — the score above is incomplete
```

Five states, and they mean different things. `ok` and `DEDUCT` are verdicts.
`skip` is a rule whose precondition never arose — no reversing phase, no stop —
and is not a pass. `manual` is a rule the DSR marks as not image-processing at
all. `BLOCK` is the one to read: `handbrake` and `multi_vehicle` are capabilities
nothing here produces, so those rules are unevaluated and the run summary says so
every time.

## The track CSV

One CSV row per processed frame:

```
frame,time_s,found,method,score,sigma_mm,
sticker_x_mm,sticker_y_mm,heading_deg,          marker position and car heading
box1_x_mm,box1_y_mm, … box4_x_mm,box4_y_mm,     car footprint, on the ground
sticker_height_mm,                              which plane the next four sit on
stick1_x_mm,stick1_y_mm, … stick4_y_mm,         detection box, on the marker plane
<roi>_mm,<roi>_hit, …                           two columns per ROI
```

The `box…` corners are the car's outline on the tarmac; the `stick…` corners are
the matched template's own outline, up on the roof. They are on **different
planes** and `render.py` projects them through different homographies —
synthesising the marker's plane from the camera pose at `sticker_height_mm`.
Push the detection box through the ground homography instead and it lands about
two metres away, still looking entirely plausible.

In the render: **green** is the car footprint (heavy edge = front), **yellow**
the detection box on the marker, **red cross** the ground point beneath it,
**purple** the heading. ROI shapes are amber when clear and red when the car
box is touching or crossing them.

`<roi>_mm` is the gap between the car box and that ROI, negative when the ROI
lies wholly under the car and otherwise the shortest distance between the two
outlines. `<roi>_hit` is the boolean a rule actually asks — kept separate so a
clearance of `0.0` is never mistaken for a near miss that rounded down.

**A line the box straddles reads negative**, and its size is how far the box
would have to move to come off that line. Zero is the honest answer to "how far
apart are they" the moment two outlines touch, and it stays zero however far the
car drives on — so a graze and a bumper 800 mm over a boundary used to render
alike, both `+0.00 m`. The shortest distance that still means something once
they overlap is the shortest one that *separates* them: push the box
perpendicular to the line until every corner is on one side, whichever side is
cheaper. A car halfway across a line reads half its own length.

Polygons keep their `0.0` on crossing. A closed shape has an inside, so how far
a car has intruded into a region is a different question from how far it has
crossed a boundary, and it is not the one this column answers. Every scoring
threshold is a monotone test on this number (`gap <= min_mm`, `gap > max_mm`)
and every one of them is non-negative, so the lines going negative leaves every
rule's verdict exactly as it was. Pipeline 2 appends `scale`, `expected_scale` and
`scale_ratio`; `render.py` ignores columns it does not know, so it reads either.

## The two things worth understanding

**Two planes, never mixed.** The sticker sits on the roof, so it is read on a
horizontal plane at the marker's height — synthesised from the camera pose at
*this vehicle's* `sticker_height_mm`, which reports the ground position directly
beneath it. The ROIs are painted on the tarmac and are read on the field plane.
The two only ever meet as world millimetres, never as pixels. Read the sticker
on the ground plane instead and a roof marker lands metres away — about 2.0 m at
B7 — while still looking entirely plausible.

**Car geometry is in the sticker template's frame, not the car's.** Origin at
the sticker centre, +X along the template's *width* axis.
`sticker_yaw_offset_deg` is the car's forward axis measured from that +X, and it
is applied only when reporting heading — it does not rotate the stored polygon.
Get it wrong and a pixel-perfect detection draws the box at right angles across
the car, which nothing downstream can catch. The `car` step derives it from a
click so it cannot be got backwards.

**A flat marker is an assumption, and `--corner-pnp` is how you drop it.** The
detector fits three numbers — two of position and one of rotation — which
describes a marker lying flat at exactly `sticker_height_mm`. A braking car
dips its roof: the marker tilts and shifts while the footprint stays put, and
three numbers cannot represent that, so the whole dip is charged to the car's
position. It is a bias, not noise, so averaging frames will not remove it.

`--corner-pnp` refits the same patch with all eight parameters, takes the four
corners that fit implies, and solves them as a marker pose. What it uses from
that pose is **only the attitude**. Position still comes from the planar read,
because the two measure different things well: pinning the marker to its
surveyed height is strong information, and a pose solved from four corners
alone throws it away — tilt and depth trade off along the viewing ray, so the
solved centre drifts by tens of millimetres where the planar read holds a few.
Measured on a synthetic B7 scene, the full pose put the footprint out by 77 mm
where the planar centre held 6 mm. So the offset from marker to footprint is
rotated by the measured attitude and hung off the planar centre.

Mean footprint-corner error, pipeline 1, synthetic B7 geometry, car pitching
about its contact patches with roll at half the pitch:

| pitch | default | `--corner-pnp` |
|---|---|---|
| 0° | 6.5 mm | 4.5 mm |
| 1° | 28.1 mm | 4.5 mm |
| 2° | 60.4 mm | 4.4 mm |
| 3° | 92.9 mm | 4.6 mm |

**It needs a marker at least ~60 px on its short side** — 700 mm at B7's
11 mm/px. Below that the eight-parameter fit is biased rather than merely
noisy: at 36 px a level marker read as 1.5° tilted, which moves the footprint
the wrong way and made `--corner-pnp` *worse* than leaving it off. Both
pipelines print the marker's pixel size and warn when it is under the limit.
Frames where the fit does not converge fall back to the planar read, and the
`plane_fit` column records which was used per frame, alongside `pitch_deg` and
`roll_deg`.

**Check `plane_fit` rather than the banner.** The startup line says the flag is
on and the marker is big enough; it does not say the fit ran. The fallback is
silent by design — a frame the corners cannot be fitted on should degrade to the
old number rather than to no number — which also means a fit that never
converges looks exactly like one that always does. `plane_fit` is the only place
that distinguishes them, and a column of `planar` with the flag on is the thing
to notice. On B4's `B4_T2_near.mp4`, 22-23 s, all 21 frames come back
`corner-pnp` at pitch +0.20° and roll -0.47°, moving the footprint 16 mm from
where the planar read put it.

It is not free: the eight-parameter ECC runs up to 200 iterations on top of the
Euclidean one that seeded it, and on those same 21 B4 frames the run went from
0.65 to 2.5 s per frame. That is the cost of the measurement, not overhead —
budget for roughly double the detection time when it is on.

Pipeline 1's numbers above are measured. Pipeline 2 carries the same code path
but is **not yet validated the same way** — the synthetic harness for it is not
trustworthy, and separately `polish_where_it_is` rejects its re-cut at larger
tilts, because a crop synthesised on the flat plane stops matching a marker
that is not on it. Treat `--corner-pnp` on pipeline 2 as untested.

## Accuracy

Measured on `raw/B7/B7_park.mp4` (3840×2160), 41 frames, against the same frame
`dlms detect measure` reports as sticker `(+357.6, +2577.8) mm, heading -89.97°`:

| | sticker position | heading | frame-to-frame jitter |
|---|---|---|---|
| pipeline 1 | `(+351.8, +2553.9)` — 25 mm out | −89.81° | 0.04 mm mean |
| pipeline 2 | `(+338.0, +2521.0)` — 61 mm out | −89.08° | 0.04 mm mean |

Both detect 100% of frames and agree with each other to **36 mm and 0.76°**. The
residual against `dlms` is resampling phase: these size their own raster from the
ROI extent, so the warp lands on a different sub-pixel grid.

## Cost

Per tracked frame at the defaults (4K source, 9.41 mm/px, `--search-px 140`),
median over 24 frames:

| stage | pipeline 1 | pipeline 2 |
|---|---|---|
| decode (4K h264) | 13.8 ms | 13.8 ms |
| undistort remap (8 MP) | 19.1 ms | 17.8 ms |
| warp to car plane | 7.2 ms | — |
| clahe+unsharp | 8.8 ms | 30.3 ms (full frame) |
| detect | 433 ms | 288 ms |
| site re-cut + ECC | — | 79 ms |
| box + ROI clearances | 0.8 ms | 0.2 ms |
| **total** | **483 ms (2.1 fps)** | **429 ms (2.3 fps)** |

The cold first-frame search — no prior, every angle, whole region — is 0.6 s for
pipeline 1 and 3.8 s for pipeline 2. It happens once.

Detection dominates, and the reason is that masked `matchTemplate` costs the
*product* of the search window's area and the template's. Both scale with the
fourth power of resolution, so `--mm-per-px` and `--search-px` are the levers
that matter; everything else is rounding error.

### Getting to 10 fps (pipeline 1)

| mm/px | `--search-px` | raster | template | detect | total | fps | drift vs 9.41 |
|---|---|---|---|---|---|---|---|
| 9.41 | 140 (default) | 1349×1573 | 85×106 | 468 ms | 518 ms | 1.9 | — |
| 9.41 | 40 | 1349×1573 | 85×106 | 193 ms | 243 ms | 4.1 | 0.1 mm |
| 15.0 | 40 | 846×987 | 53×66 | 67 ms | 110 ms | **9.1** | 2.5 mm |
| 20.0 | 40 | 635×740 | 40×50 | 63 ms | 103 ms | **9.7** | 4.5 mm |

`--search-px 40` is 376 mm of travel between frames — 27 km/h at 20 fps — and
costs nothing measurable in accuracy. Coarsening the raster to 15 mm/px costs
2.5 mm. Those two together reach ~9 fps on 4K input, of which 33 ms is decode
plus undistort that no detector change can remove.

Two things would take it past 10 fps comfortably, neither implemented here:
fold the undistortion into the bird's-eye remap so there is one resample per
frame instead of two (saves ~19 ms and an 8 MP intermediate), and feed 1080p
instead of 4K (decode plus undistort drops from 33 ms to ~11 ms — but note the
stored homographies and intrinsics are for 3840×2160 and would have to be
scaled, which these files do not do).

Pipeline 2 is the harder case at B7: its crop is 369×277 px because the marker
is genuinely that large near the camera, and it needs 14 scale bands to cover
the region. It is viable, but pipeline 1 is both faster and more accurate here.

## Which to use

Pipeline 1 pays a full-frame remap every frame. In exchange the marker is the
same size and shape wherever the car is, and rotation is the only free parameter.

Pipeline 2 resamples nothing, but an oblique camera shears the marker, and a
rotate-and-scale template cannot represent shear at all — which shows up as a
biased *angle*, not as a failure to match. Left uncorrected at B7 that was a
**7.5° heading error**, enough to swing a bumper corner by 200 mm while still
scoring a confident match. So after the band search it re-cuts the crop through
the homography *at the position just found*, where the template is exact, and
refines once more; those rows are marked `hybrid:ecc@site`. That is what brings
it back to 0.76°.

The startup banner prints the worst anisotropy over the search region — 2.94 at
B7, i.e. the marker is nearly three times more squashed along one axis than the
other. Above ~1.25 pipeline 1 is the better tool, and `scale_ratio` in the CSV
tells you per frame how far the raw assumption was stretched. Pipeline 2 refuses
outright if the region needs more than 16 scale bands, on the grounds that at
that point it is doing more search than the warp it was avoiding.
