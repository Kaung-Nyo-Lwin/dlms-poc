# poc — survey a site, then track a car through it

Flat, dependency-light versions of the whole path, written to be read start to
finish. Every file runs on its own: **numpy + opencv + the standard library,
nothing imports `dlms`.**

| file | what it does |
|---|---|
| `calibrate.py` | the survey tool — all six steps, one folder of files |
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
python calibrate.py sticker    --image $S/frame.png --calibration $S/calibration.json \
                               --height-mm 1450 --out $S/sticker.png
python calibrate.py car        --image $S/frame.png --calibration $S/calibration.json \
                               --sticker-height-mm 1450 --length-mm 4700 --width-mm 1800 \
                               --sticker-to-front-mm 2000 --wheelbase-mm 2750 \
                               --track-mm 1550 --out $S/car.json
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
next shape, in `gcp` it is the world `X, Y` you type *before* clicking the mark.
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
- **`gcp`** — click a mark, type its world `X, Y` in millimetres. The world frame
  is yours; pick an origin and an axis on the tarmac and stay consistent. Four
  points is the minimum and is exactly determined — six or more spread across
  the working area is what makes the residuals mean anything. A residual over
  50 mm is almost always a mistyped coordinate or a click on the wrong mark.
- **`measure`** — click two ground points, read the distance back, compare with a
  tape. The `gcp` residuals only say the fit agrees with itself at the marks it
  was given; this is the number a tape can argue with. Do it before trusting
  anything downstream.
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
