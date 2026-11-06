<p align="center">
  <img src="logos/HelioBeamAI.png" alt="logo" width="50%"/>
</p>

[![DOI](https://img.shields.io/badge/DOI-10.21203/rs.3.rs--6192986/v1-brightgreen)](https://doi.org/10.21203/rs.3.rs-6192986/v1)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)

# HelioBeamAI

Transformer-based receiver-level flux prediction from calibration target images.

HelioBeamAI predicts the focal spot of every heliostat for arbitrary sun
positions and aim points from the images the calibration system records
anyway. A transformer encoder reads a heliostat's calibration measurements
(images compressed by a VAE, plus sun vector, aim direction and distance);
the decoder predicts the focal spot for new conditions. A projection CNN
turns the spot into a ray bundle whose directions follow the beam's
divergence, so the flux can be ray traced onto any receiver surface.

Method and results: [Kuhl et al., 2025](https://doi.org/10.1016/j.solener.2025.113631).

## Installation

```bash
pip install -e .                    # prediction (torch, numpy)
pip install -e ".[examples]"        # + matplotlib, jupyter for the tutorials
pip install -e ".[train,data]"      # + training (pytorch-lightning, PyYAML) and dataset building
pip install -e ".[dev]"             # + pytest
heliobeamai download                # model files for prediction (~30 MB), see ARTIFACTS.md
```

## Tutorials and example data

* [examples/01_predict.ipynb](examples/01_predict.ipynb): focal spots from calibration
  measurements, comparison with held-out measurements, new sun positions and aim points, rays and
  flux on a plane, Gaussian fallback.
* [examples/02_example_data.ipynb](examples/02_example_data.ipynb): every training dataset type.
* [examples/data/](examples/data/): the training datasets of the Jülich solar tower for 5 heliostats
  (the full datasets are not published); `configs/transformer/example.yaml` trains on them.

## Prediction

Simplest: one object for spots and rays (used by aimpoint-control):

```python
from heliobeamai import HelioBeamField, GaussianSource

field = HelioBeamField.from_artifacts(                                # files from `heliobeamai download`
    device="cuda", pointing_accuracy_mrad=1.8,
    fallback=GaussianSource.field_law(sigma_mrad=3.1, sigma0_m=0.58))  # heliostats without calibration context
field.assignment(heliostat_ids)                    # {"transformer": [...], "gaussian": [...], "uncovered": []}
prepared = field.prepare(heliostat_ids, sun, positions, aim_points)   # once per geometry update
rays = field.rays(prepared, aim_points, powers)                       # [N*4096, 7], differentiable in aim_points
```

* All heliostats share one projection chain (frame C, tracking broadening,
  ray generation); sources only produce the focal spot. Each heliostat uses
  the first source that covers it: the transformer for heliostats in the
  encoder cache, then the fallback.
* Without `fallback` only the transformer is used (e.g. transformer-only
  experiments); heliostats without context raise a `KeyError`.
* Own spot models: any object with `name`, `use_cnn`, `covers(ids)` and
  `spots(geometry)` (`heliobeamai.SpotSource`), passed as `fallback` or via
  `HelioBeamField(sources, projector)`.

Coordinates (tower-base ENU, sun vector, image frame, ray layout): [docs/data.md](docs/data.md).

The building blocks:

```python
import torch
from heliobeamai import FluxPredictor, RayProjector, build_conditions

predictor = FluxPredictor.from_checkpoint("artifacts/transformer/T2026.pt",
                                          encoder_cache="artifacts/encoder_cache/T2026.pt")
projector = RayProjector.from_checkpoint("artifacts/projector/cnn_frameC_v1.pt",
                                         pointing_accuracy_mrad=1.8)

sun = torch.tensor([0.3, -0.5, 0.8])                   # towards the sun (E, N, U)
positions = torch.tensor([[-57.3, 24.8, 1.6]])         # heliostat positions (tower base frame)
aim_points = torch.tensor([[0.0, 1.0, 54.0]])
cond, _, _ = build_conditions(sun, positions, aim_points)   # [N, 1, 7]

spots = predictor.predict(["AA23"], cond)[:, 0]        # [N, 64, 64], frame C, 6 m x 6 m
rays = projector(spots, aim_points, cond[:, 0], powers=torch.tensor([30.0]))
# rays: [N * 64 * 64, 7] = origin (3), direction (3), energy
```

* **Image frame C** (all images): row 0 = top (highest z), column 0 = east
  (highest x); `plt.imshow(spot)` shows the spot as seen from the field.
* **Conditions**: sun vector, unit aim direction, aim distance / 350 m.
* New heliostats or contexts: `predictor.encode_context(id, images, conditions)` (images
  `[S, 64, 64]`, conditions `[S, 7]` with the distance in m; see the tutorial); a whole dataset:
  `heliobeamai encode`.
* For optimization loops: `projector.prepare(...)` once per geometry update,
  `projector.rays_from_prepared(...)` per iteration.

### Receivers and total flux

```python
from heliobeamai import OVRReceiver, CylindricalReceiver
from heliobeamai.predict import receiver_flux

receiver = OVRReceiver(center=(0, 0, 55), width=8, height=8, resolution=(80, 80), tilt_deg=0)
result = receiver_flux(field, receiver, ids, positions, aim_points, sun, dni=850, mirror_area=8.2,
                       reflectivity=0.9)                      # P = DNI * A * rho * cos(incidence)
receiver.visualize_flux_distribution(result["flux"])         # W/m^2, as seen from the field
```

`Receiver` subclasses implement `intersect(ray_directions, ray_origins)`;
`project_flux(rays, soft=True)` bins differentiably. From the command line:

```bash
heliobeamai predict --heliostats field.csv --aim 0 0 55 --sun 180 45 --dni 850 \
    --mirror-area 8.2 --reflectivity 0.9 --fallback 3.1 0.58 --output results/noon
```

`field.csv`: `heliostat_id,x,y,z` (optionally `aim_x,aim_y,aim_z,mirror_area,reflectivity`);
`--context dataset.pt` encodes heliostats from their own calibration measurements.

## Training

```bash
heliobeamai train-vae --images data/CalibrationDataFluxCentered --output runs/vae/z64CH4   # autoencoder
heliobeamai build-dataset --output data/flux_2026.pt --seed 0 \
    --calibration-csv ... --heliostat-positions ... --flux-dir ...     # calibration dataset
heliobeamai train-transformer configs/transformer/T2026.yaml           # transformer
heliobeamai encode --checkpoint runs/transformer/T2026/model.pt \
    --dataset data/flux_2026.pt --output runs/transformer/T2026/encoder_cache.pt
heliobeamai train-projector --data 'data/projection/receiver_*.pt' --output runs/projector/cnn.pt
heliobeamai eval-projector --data 'data/projection/receiver_*.pt' --models runs/projector/cnn.pt
heliobeamai train-transformer configs/transformer/example.yaml          # on examples/data, CPU
```

* `configs/transformer/`: the 2026 model and the example; `--set key=value`
  overrides entries. Dataset splits: `heliobeamai.data.get_split`.
* Dataset formats, how to build a dataset from calibration measurements, the
  autoencoder and the projection data: [docs/data.md](docs/data.md).

## Repository

```
src/heliobeamai/
  models/        transformer, VAE, projection CNN, FluxModel (transformer + VAE)
  inference/     FluxPredictor (encoder cache), build_conditions, RayProjector
  data/          dataset builder, splits, batching, Lightning data module
  training/      transformer (Lightning), VAE, projection CNN, losses, AdamP
  field.py       HelioBeamField (sources + one projection chain)
  sources.py     spot sources: TransformerSource, GaussianSource
  receiver.py    receivers: PlanarReceiver, OVRReceiver, CylindricalReceiver
  predict.py     receiver flux of a field (`heliobeamai predict`)
  checkpoints.py loading/saving (also the original dissertation checkpoints)
  artifacts.py   released model files: download, checksums
  cli.py         `heliobeamai` command
configs/         training configs
examples/        tutorials and example data (5 heliostats)
docs/            data formats, coordinates
tests/           unit tests
```

## Tests

```bash
pytest            # tests that need the model files are skipped until `heliobeamai download`
```

## Citation

See [CITATION.cff](CITATION.cff).

## Acknowledgments

This work is supported by the [Helmholtz AI](https://www.helmholtz.ai/) platform grant.

-----------
<div align="center">
  <a href="https://www.dlr.de/EN/Home/home_node.html"><img src="./logos/logo_dlr.svg" height="50px" hspace="3%" vspace="25px"></a>
  <a href="https://www.fz-juelich.de/portal/EN/Home/home_node.html"><img src="./logos/logo_fzj.svg" height="50px" hspace="3%" vspace="25px"></a>
  <a href="https://www.helmholtz.ai/"><img src="./logos/logo_hai.svg" height="25px" hspace="3%" vspace="25px"></a>
</div>
