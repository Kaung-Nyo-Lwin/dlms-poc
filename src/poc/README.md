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

There are no stations, no vehicle ids and no workspace. Every input and output is
a path you choose, so a whole site is one directory.

## 1. Survey

Run the steps in order; each writes a file the next one reads.

```sh
cd src/poc
S=~/site                      # one folder for everything

python calibrate.py frame      --video $S/station.mp4 --at 0 --out $S/frame.png
python calibrate.py intrinsics --video $S/checkerboard.mp4 --board 11x7 --square-mm 30 \
                               --out $S/intrinsics.json
python calibrate.py gcp        --image $S/frame.png --intrinsics $S/intrinsics.json \
                               --out $S/calibration.json
python calibrate.py measure    --image $S/frame.png --calibration $S/calibration.json
python calibrate.py tape       --image $S/frame.png --calibration $S/calibration.json
python calibrate.py carplane   --image $S/frame.png --calibration $S/calibration.json \
                               --height-mm 1450          # optional, see below
python calibrate.py sticker    --image $S/frame.png --calibration $S/calibration.json \
                               --height-mm 1450 --out $S/sticker.png
python calibrate.py outline    --image $S/frame.png --calibration $S/calibration.json \
                               --sticker-height-mm 1450 --height-mm 500 \
                               --length-mm 4700 --width-mm 1800 \
                               --front-to-wheel-mm 900 --rear-to-wheel-mm 850 \
                               --out $S/car.json
python calibrate.py roi        --image $S/frame.png --calibration $S/calibration.json \
                               --out $S/rois.json
```

Each interactive step prints a `http://127.0.0.1:…` URL and opens it. If your
browser does not open (common under WSL), pass `--no-open` and click the URL.
In the page: **wheel** zoom · **drag** pan · **s** snap on/off · **u** undo ·
**r** reset view · **←↑↓→** nudge ¼ px (Shift for 1 px) · **Enter** save ·
**Esc** cancel. A loupe follows the cursor, and every click can snap to the
nearest sub-pixel corner — which is most of what makes a painted cross
repeatable between operators.

There are no pop-up dialogs anywhere. Text goes in one field in the toolbar,
read only at the moment it is needed: in `roi` it is the optional name for the
next shape, in `gcp` it is the world `X, Y` you type *before* clicking the mark,
in `tape` it is the length in millimetres you type before clicking both ends.
A dialog can be blocked outright in an embedded browser such as VS Code's
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
- **`measure`** — click two ground points, read the distance back, compare with a
  tape. The `gcp` residuals only say the fit agrees with itself at the marks it
  was given; this is the number a tape can argue with. Do it before trusting
  anything downstream.
- **`tape`** — *the second opinion on scale, from lengths you never surveyed.*
  Lay a tape between two marks on the tarmac, read the millimetres, click both
  ends. Neither end needs a world coordinate, which is what makes these cheap
  enough to lay several of, out where the four ground control points are not.
  Be blunt about what it can see: with correct GCP coordinates, four points
  against a six-parameter pose is already over-determined and
  `solvePnPRefineLM` has the click noise handled — 2 mm rms on synthetic
  station geometry, where *every* tape-based adjustment made it worse. **A tape
  is not a fix for a noisy pose.** It is the only fix for a wrong *survey*:
  coordinates typed off a stretched tape, a mis-paced offset, a transposed
  digit. Nothing in the image can see that, because the fit is perfectly
  self-consistent with the wrong answer — `gcp` reports millimetres of residual
  while the map is centimetres out.
  So the step **reports and changes nothing** unless you pass `--adjust`, and
  what it does then depends on whether the tapes agree with each other. If they
  do, the fault is scale, and it is corrected in closed form as `t -> a*t` —
  scaling the translation multiplies every ground coordinate by exactly `a` and
  touches nothing else, so the typed coordinates are scaled to match. If they
  disagree, the fault has shape, and the pose and *all* the typed coordinates
  are re-solved together against the tapes. That second part is the whole
  design: hold the typed coordinates as truth and their reprojection residuals
  pin the pose completely, so the tapes have nowhere to push and the fit comes
  back unchanged — a pose-only refinement carrying a tape measured identical to
  plain `solvePnP` in every error mode tried. `--sigma-world-mm` is therefore
  the real knob: it is how far the tapes may overrule the survey.
  Ground-distance rms over an 8x20 m working area, synthetic station geometry:

  | GCP coordinates | 4-pt PnP | rescale | free net, 1 / 2 / 3 tapes |
  |---|---|---|---|
  | exact | **2 mm** | 14 mm | 14 / 10 / 8 mm |
  | 0.5% scale error | 50 mm | **14 mm** | 37 / 30 / 31 mm |
  | 30 mm random | 100 mm | 75 mm | 59 / 46 / **26 mm** |

  Read the first row before reaching for `--adjust`: when the survey is already
  right, adjusting costs a factor of seven, which is why the gate refuses when
  no tape is more than `--gate-z` sigma out. Read the last: three tapes is
  where the free network starts winning consistently. **Lay them long** — a
  tape's noise divides by its length, so one 10 m tape is worth nine 3 m ones,
  and length is free where a fourth tape is not.
  Two refusals guard it, and neither is "the correction was large": a survey
  wrong by 30 mm at the marks is legitimately hundreds of millimetres wrong out
  where nothing was surveyed, so capping the movement would reject exactly the
  corrections worth making. What a blunder looks like is a tape still not
  fitting once the fit has had every chance (`--max-bar-z`), or a mark dragged
  further than a survey could plausibly be wrong (`--max-gcp-move-sigma`). A
  transposed digit in a length, a click on the wrong mark, and a mistyped
  ground control coordinate are each caught and named.
  Adjusting rewrites the pose, so it re-fits the surveyed `car` plane from its
  stored pole tops — leaving that stale is the one outcome that must not
  happen, since both pipelines prefer it and nothing downstream can tell it
  disagrees with the ground beneath it. **`car.json` is stale too** and the step
  says so: its `body_polygon_mm` came from clicks through the old plane. Re-run
  `outline`.
  A tape cannot tell "the pose is wrong" from "the ground is not flat here". On
  a sloped pad one laid across the fall reads long, and `--adjust` will bend the
  pose to absorb a non-planarity no pose can represent. That is the argument for
  leaving the check as the default.
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
  **Wheels** come from `--wheelbase-mm`, `--track-mm` and `--front-overhang-mm`,
  all three or none. They are not clicked: an oblique camera shows the near
  side behind the sill and the far side under the body, so the contact patches
  are constructed off the box's own axes instead. That also retires the `car`
  step's guess that the two overhangs are equal — a front-drive hatchback's
  front overhang runs 100-150 mm longer, which put each axle 50-75 mm out.
  `--check-wheels` opens a **ground-plane** raster to click whatever patches
  are visible and reports how far each is from the constructed one; that is the
  only independent word on whether the three tape numbers are right.
  `--tyre-width-mm` lets a rule ask whether a *tyre* is on a line rather than
  whether its centre point is. Without wheels, ten of the DSR's rules score as
  not-evaluated.
- **`car`** — the older tape-measured route to the same file. Prefer `outline`.
- **`sticker`** — click roughly where the marker is, then box it on the bird's-eye
  raster that appears. The box readout is in millimetres, so you can check it
  against the printed marker before committing. The step prints the
  `--template-mm-per-px` value the pipelines need (also in `sticker.json`).
- **`car`** — lengths come from a tape, orientation from two clicks: the marker
  centre, then the centre of the front bumper. That second click is the whole
  point of the step. Stored geometry is in the *sticker template's* frame, not
  the car's, and a marker printed across the roof rather than along it sits at
  ±90° to the car. Typing that number in is easy to get backwards; clicking the
  nose is not.
- **`roi`** — click points, then **p** point · **l** line · **g** polygon to
  finish each shape and name it.

Everything the tool records is in the **undistorted** frame, using the original
camera matrix as the projection. That is why the pipelines need no
`--roi-distorted` flag for ROIs surveyed here: the homographies, the ROI pixels
and the frames the pipelines undistort all share one coordinate frame.

## 2. Track

```sh
python pipeline1_bev.py --video $S/station.mp4 --calibration $S/calibration.json \
    --car $S/car.json --template $S/sticker.png --template-mm-per-px 4.79 \
    --rois $S/rois.json --out $S/track.csv

python render.py --video $S/station.mp4 --calibration $S/calibration.json \
    --csv $S/track.csv --rois $S/rois.json --out $S/track.mp4
```

`pipeline2_raw.py` takes the same flags, plus `--ref-mm X,Y` to choose where its
camera-frame crop is synthesised.

`render.py` draws on the full-resolution frame and resizes once at the end, so
line widths and text are pre-multiplied by the inverse of `--scale`. If the
labels are still too small for the screen you are reviewing on, raise
`--label-scale` (try `1.5` or `2`); `--units cm` or `mm` changes what the
clearance readout says.

Useful on both: `--start` / `--end` to clip, `--every N` to subsample,
`--margin-mm` to size the search area, and on pipeline 1 `--mm-per-px` for the
raster resolution and `--search-px` for the tracking window.

## Output

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

`<roi>_mm` is the gap between the car box and that ROI: `0.0` when they cross,
negative when the ROI lies wholly under the car. `<roi>_hit` is the boolean a
rule actually asks — kept separate so a clearance of `0.0` is never mistaken for
a near miss that rounded down. Pipeline 2 appends `scale`, `expected_scale` and
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
