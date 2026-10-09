# HDIT-Stereo

HDIT-Stereo is a diffusion model for stereo matching. Given a rectified stereo pair, it
denoises two maps on a **cyclopean grid** (a virtual view centered between the two cameras):

* **disparity**, and
* a **boundary distance field (UDF)**, the distance to each object's boundary, which
  sharpens depth edges.

The denoiser can be conditioned on the stereo pair alone, or additionally on cost volumes:
a **Lean Correlation Conditioner (LCC)**, a supervised symmetric cost volume, and/or a
**DINOv2** feature cost volume.

This repository covers the full pipeline: synthetic data generation, training, sampling, and
evaluation.

## Contents

| Path | Purpose |
|------|---------|
| `train_udf.py` | Training (via `accelerate`) |
| `infer.py` | Multi-seed sampling and per-sample metrics |
| `report_metrics.py` | Aggregate, boundary-sharpness and contour-shape metrics |
| `shape_recovery_eval.py` | Per-object shape recovery (IoU modulo similarity) |
| `k_diffusion/` | Model, sampler, config and dataset code |
| `shapematch/` | Vendored `shape-match-2d` package (used by `shape_recovery_eval.py`) |
| `configs/` | One JSON config per model variant |
| `data_gen/` | Synthetic stereo scene generator (`shape_stereo_slant.py`) |
| `scripts/` | Shell entry points: `generate_data.sh`, `train.sh`, `infer.sh`, `eval_metrics.sh` |

## Quick start

```bash
pip install -r requirements.txt

bash scripts/generate_data.sh train && bash scripts/generate_data.sh test
bash scripts/train.sh lcc_dino
bash scripts/infer.sh <checkpoint.pth>
bash scripts/eval_metrics.sh results/<checkpoint name>
```

Run all commands from the repository root. A GPU is required for training and inference.

## 1. Generate data

```bash
bash scripts/generate_data.sh train   # 25,000 images -> data/stereo_128_train
bash scripts/generate_data.sh test    #    100 images -> data/stereo_128_test
```

Each split contains:

* `left/`, `right/`: the stereo inputs
* `disp_left/`: ground-truth disparity
* `fields_left/`: ground-truth UDF
* `amodal_mask_obj{1,2}/`, `disp_obj{1,2}/`: per-object masks and disparity, used by the shape metric

**Textures.** Procedural textures are used by default. To reproduce paper-style data,
download the [Describable Textures Dataset](https://www.robots.ox.ac.uk/~vgg/data/dtd/) and
place the `banded` and `woven` classes in `data_gen/texture_imgs/banded` and
`data_gen/texture_imgs/woven`; the script detects them automatically.

All generator options: `python data_gen/shape_stereo_slant.py --help`.

## 2. Train

```bash
bash scripts/train.sh <variant> [extra train_udf.py args]
```

| Variant | Outputs | Cost-volume conditioning |
|---------|---------|--------------------------|
| `baseline` | disparity | none |
| `dino` | UDF + disparity | DINOv2 only |
| `lcc` | UDF + disparity | LCC only |
| `lcc_dino` | UDF + disparity | LCC + DINOv2 (**full model**) |

Environment variables:

| Variable | Default | Meaning |
|----------|---------|---------|
| `NUM_GPUS` | 4 | Number of GPUs |
| `BATCH_SIZE` | 16 | Batch size per GPU |
| `RUN_NAME` | `hdit_stereo_<variant>_128_left` | Run / checkpoint prefix |
| `PORT` | 29511 | Distributed port |
| `WANDB_PROJECT`, `WANDB_ENTITY` | unset | Set `WANDB_PROJECT` to enable Weights & Biases |

Checkpoints are saved as `<RUN_NAME>_<step>.pth`. DINOv2 weights are downloaded through
`torch.hub` on first use, so internet access is needed once.

## 3. Inference

```bash
bash scripts/infer.sh <checkpoint.pth> [data/stereo_128_test] [out_dir]
```

Samples 10 seeds per image (override with `SEEDS="0 1 2"`). Outputs go to `out_dir`
(default `results/<checkpoint name>`):

* `*_disp.npy`: predicted disparity
* `*_udf_obj1.npy`, `*_udf_obj2.npy`: predicted UDF (UDF models only)
* plots, `metrics.csv` (per seed) and `metrics_ensemble.csv`

The model configuration is read from the checkpoint.

## 4. Evaluate

```bash
bash scripts/eval_metrics.sh results/<checkpoint name> [data/stereo_128_test]
```

Writes three files next to the predictions:

| File | Contents |
|------|----------|
| `agg.csv` | Per-seed mean / median / best-K aggregation and the map-median ensemble. Report the map-median `disp_epe_med` as the headline number; best-K is an oracle upper bound. |
| `boundary_metrics.csv` | Edge sharpness (`sharp_ratio`) and boundary vs. interior EPE |
| `shape_recovery.csv` | Per-object shape IoU after similarity alignment |

`report_metrics.py` also has a `shape` subcommand (Chamfer / Hausdorff contour distance);
see `python report_metrics.py --help`.