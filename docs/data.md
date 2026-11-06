# Data

The full datasets are not published. This page describes their content and
format so that equivalent data can be produced for another plant.
`examples/data/` holds both dataset types for five heliostats of the Jülich
solar tower (AC26, AA44, AO50, AY27, BC71; images as float16);
`examples/02_example_data.ipynb` shows them.

## Coordinates and image frame

World coordinates are tower-centred ENU in metres: x east, y north, z up
(calibration database `TargetOffsetU - 87`). Heliostats stand north of the
tower; targets and the receiver aperture are x-z planes facing +y.

**All focal spot images use frame C** (camera view, as the calibration images
are taken from the field):

* row 0 is the top (highest z), rows run along -z;
* column 0 is the east edge (highest x), columns run along -x;
* pixel `(r, c)` of an image of size `H x W` covering `side_x x side_z` metres
  around point `p` is centred at
  `x = p_x + side_x/2 - (c + 0.5) side_x / W`, `z = p_z + side_z/2 - (r + 0.5) side_z / H`.

`matplotlib.imshow(img)` (default `origin="upper"`) shows a spot as seen
from the field. The transformer, the projection CNN and
`heliobeamai.RayProjector` all work in frame C; no flips or transposes are
needed anywhere in the chain.

### Inputs of `HelioBeamField`

| input | shape | definition |
|---|---|---|
| `heliostat_ids` | list of N | names like `"AC29"`; numeric database ids (`10123`) are normalized (`heliobeamai.normalize_heliostat_id`) |
| `positions` | [N, 3] | heliostat positions, world frame |
| `aim_points` | [N, 3] | aim points, world frame; the spot plane is the vertical plane through the aim point facing +y |
| `sun` | [3] or [N, 3] | unit vector from the field towards the sun: (cos el sin az, cos el cos az, sin el), azimuth from north, clockwise |
| `powers` | [N] | optional, W; each spot is scaled to integrate to its power |

Derived per heliostat (`heliobeamai.build_conditions`): aim vector
`v = aim_point - position`, distance `d = |v|`, direction `a = v / d`. Model
conditions (7 values): `(sun_x, sun_y, sun_z, a_x, a_y, a_z, d / 350 m)`.

### Rays

`HelioBeamField.rays(...)` returns `[N * H * W, 7]` (`8` with group indices),
heliostat by heliostat in input order, pixels row-major:

| columns | content |
|---|---|
| 0-2 | origin = pixel centre on the spot plane (frame C layout above) |
| 3-5 | unit direction = normalize(`a` + 0.01 * CNN deviation of the pixel); `a` without CNN |
| 6 | energy = power * pixel value / sum of pixel values |
| 7 | group index (if given) |

Origins and directions are differentiable with respect to `aim_points`.
Intersection with a receiver surface is left to the caller.

## Calibration flux datasets (transformer)

Format of `fluxTransData2026.pt` (built by `heliobeamai build-dataset`;
example: `examples/data/calibration_2026.pt`):

| key | shape | content |
|---|---|---|
| `imgs` | `[H, S, 64, 64]` | centred calibration flux images, 6 m x 6 m, frame C, mean 0.1 each |
| `data` | `[H, S, 7]` | sun vector (3), aim direction (3), aim distance in m (1) |
| `ids` | `[H, S]` | calibration record ids |
| `heliostat_ids` | `[H]` | heliostat ids |
| `target_ids`, `target_groups`, `samples_per_target`, `min_total_samples`, `copied_heliostats` | | build metadata |

Samples are stored in target blocks of `samples_per_target`; the first samples
of each block are the held-out test samples of the dataset splits.

### Building a dataset from calibration measurements

`heliobeamai build-dataset` needs three inputs:

| input | content |
|---|---|
| `--calibration-csv` | one row per calibration measurement: `id` (record id), `HeliostatId` (numeric, e.g. `10123` = AA23), `CalibrationTargetId`, `SunPosE/N/U` (unit vector towards the sun, east/north/up), `TargetOffsetE/N/U` (spot centroid on the target, m; height above sea level) |
| `--heliostat-positions` | Excel sheet with `InternalName` (e.g. `AA23`) and `x`, `y`, `z` (m; height above sea level) |
| `--flux-dir` | one flux image per measurement, `<id>.png`: flux density on the calibration target, **centred on the spot centroid, 6 m x 6 m, frame C** (row 0 at the top, column 0 at the east edge, i.e. the camera view of a target facing the field); any resolution |

Heights are converted to the tower-base frame with the offset
`HEIGHT_OFFSET_M = 87` m of the Jülich tower and targets are grouped by
`TARGET_GROUPS_2026` (`heliobeamai.data.build_calibration`); both are
plant-specific. The conditions are computed as sun vector, unit vector
heliostat -> spot centroid and their distance.

Processing per image: resized to 64 x 64, Gaussian smoothing (5 px kernel,
sigma 1 px), scaled to mean 0.1, then reconstructed by the autoencoder `z64`
(`--vae`, a posterior sample; `none` skips it). The transformer therefore
learns from smooth, VAE-reconstructed images. Per heliostat and target group,
`samples_per_target` measurements are chosen by max-min sampling of the sun
position (azimuth, elevation).

How the flux images are made from the camera images is plant-specific and
not part of the package. For Jülich: the camera image of the target is
rectified with the target markers, converted to flux density (a U-Net trained
on the camera response) and centred on the spot centroid. What matters for
HelioBeamAI is the result: flux density (not brightness) on the target plane,
centred, 6 m x 6 m, in frame C, with the matching sun vector and spot
position. A wrong image orientation (e.g. mirrored) is not detected by the
training loss; check it against the spot positions as described in
"Coordinates and image frame".

### Autoencoder (VAE)

The transformer works on 64-value latents of the images. The released VAEs
were trained with `heliobeamai train-vae` (port of the original script, see
`heliobeamai.training.vae`): mean squared reconstruction error, Adam
(lr 1e-3), batch 256, 1000 epochs, on all centred flux images:

```bash
heliobeamai train-vae --images data/CalibrationDataFluxCentered --output runs/vae/z64CH4 --ch 4   # transformer VAE
heliobeamai train-vae --images data/CalibrationDataFluxCentered --output runs/vae/z64 --ch 16     # dataset builder VAE
```

Training configs start from the VAE with `init: {vae: runs/vae/z64CH4/vae.pt}`.

## Projection dataset (projection CNN)

Used by `heliobeamai.training.projector` (`ProjectionData.load`), format
`heliobeamai.projection_data/1`. Ray-traced flux of single heliostats on
parallel planes, so that the CNN learns how rays leaving the aim plane spread.

### Files

Chunk files `<mode>_<chunk>.pt` (`mode` = `receiver` or `calibration`):

| key | shape / type | content |
|---|---|---|
| `fluxes` | `[N, P, 64, 64]` | flux on planes 0..P-1, frame C; all planes of a case share one scale factor (plane 0 has mean 0.1) |
| `plane_centers` | `[N, P, 3]` | plane centres; plane 0 is the aim point, plane k = aim point + k * `plane_spacing_m` * aim direction |
| `heliostat_positions` | `[N, 3]` | heliostat position |
| `sun_vectors` | `[N, 3]` | unit vector towards the sun |
| `heliostat_names` | list of N | e.g. `"AA27"` |
| `record_ids` | list of N | calibration record that supplied the sun vector (and in `calibration` mode the aim point) |
| `frame`, `side_lengths_m`, `plane_spacing_m`, `mode`, `generator` | | metadata |

Planes are x-z planes (normal +y), 6 m x 6 m, 64 x 64 pixels, P = 6, spacing
1 m. Images are stored without smoothing; training smooths them with a
Gaussian (sigma 1 px, `--smooth`).

### Content

* **Heliostats**: all heliostats with deflectometry data and a known position
  (186); the ray tracer uses the measured mirror surfaces.
* **Sun**: per heliostat 20 sun vectors of its real calibration records
  (`SunPosU > 0.05`).
* **Aim point**: `receiver` mode: receiver centre `(0, 1, 54)` plus uniform
  noise in `[-2, 2]` m per axis (training data). `calibration` mode: the
  database aim point of the record (2 per heliostat), so every case can be
  compared with the measured image.
* The CNN conditions are those of the transformer: sun vector, aim direction,
  aim distance / 350 m.

### Generation

An in-house ray tracer (not part of this package) traces each case once,
takes the hits on the aim plane and propagates them along their directions to
the other planes, then bins all planes directly in frame C. Any ray tracer can
be used that delivers the flux of one heliostat on such planes.
