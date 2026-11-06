# Model files

Model files are not in git. They are published as assets of the GitHub release
`models-v1` and downloaded with

```bash
heliobeamai download                 # prediction (~30 MB)
heliobeamai download training        # + VAEs for training and dataset building
heliobeamai download --verify all    # check local files against the SHA-256 checksums
```

or from Python with `heliobeamai.artifacts.download()`. Files are stored in
`$HELIOBEAMAI_ARTIFACTS`, else `<repository>/artifacts` (source checkout), else
`~/.cache/heliobeamai/artifacts`, under the paths below. A mirror can be used
with `--url` or `$HELIOBEAMAI_ARTIFACTS_URL` (asset name = path with `/`
replaced by `__`). Checksums: `src/heliobeamai/_manifest.py`.

## Prediction (`prediction`)

| file | size | content |
|---|---|---|
| `transformer/T2026.pt` | 1.4 MB | transformer + jointly trained VAE, 2026 calibration data (configs/transformer/T2026.yaml) |
| `encoder_cache/T2026.pt` | 23.1 MB | encoder states of 1988 heliostats (Juelich) for transformer/T2026.pt |
| `projector/cnn_frameC_v1.pt` | 5.0 MB | projection CNN, frame C, 7 transformer conditions |

The encoder cache holds the context of 1988 heliostats of the Jülich solar
tower, encoded from 45 calibration measurements each.

## Training (`training`)

| file | size | content |
|---|---|---|
| `vae/z64CH4.pt` | 0.7 MB | VAE the transformers start from (1x8x8 latent) |
| `vae/z64.pt` | 10.9 MB | VAE used by the dataset builder to smooth flux images |
