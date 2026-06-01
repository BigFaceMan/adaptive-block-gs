# Repository Guidelines

## Project Structure & Module Organization

This is a Python/CUDA 3D Gaussian Splatting research codebase. Main entry points live at the repository root: `train.py`, `render.py`, `metrics.py`, `partition.py`, `merge_blocks.py`, `convert.py`, and `full_eval.py`. Core scene/model logic is in `scene/`; differentiable rendering wrappers are in `gaussian_renderer/`; CLI argument groups are in `arguments/`; shared helpers are in `utils/`; LPIPS code is vendored in `lpipsPyTorch/`. Experiment YAMLs are under `config/`, shell workflows under `scripts/`, docs under `docs/`, and media under `assets/`. External CUDA/viewer code is managed through `.gitmodules` in `submodules/` and `SIBR_viewers/`.

## Build, Test, and Development Commands

- `conda env create --file environment.yml`: create the `BlockGS` environment with PyTorch, CUDA, rasterizer submodules, and experiment dependencies.
- `conda activate BlockGS`: activate the project environment.
- `python train.py -s <dataset> -m output/<run> --eval`: train on COLMAP or Blender data with a train/test split.
- `python render.py -m output/<run>`: render train/test views from a trained model.
- `python metrics.py -m output/<run>`: compute SSIM, PSNR, and LPIPS for rendered outputs.
- `python partition.py --config config/<file>.yaml`: run configured large-scene partitioning workflows.
- `bash scripts/<workflow>.sh`: run curated pipelines; check paths and GPU variables first.

## Coding Style & Naming Conventions

Follow the existing Python style: 4-space indentation, snake_case functions and variables, CamelCase classes, and concise module-level helpers. Keep training orchestration in `train.py`, Gaussian state/densification in `scene/gaussian_model.py`, renderer changes in `gaussian_renderer/`, and general utilities in `utils/`. Prefer YAML configs for repeatable experiment parameters instead of hard-coded local paths.

## Testing Guidelines

No formal test suite is currently configured. Validate changes with focused smoke runs: train for a small iteration count, render one model, and run `metrics.py` when outputs are available. For partitioning changes, test the smallest relevant YAML and confirm expected files appear under `output/`. Do not commit generated `output/`, checkpoints, PLY files, datasets, or logs.

## Commit & Pull Request Guidelines

Recent history uses short imperative messages such as `Add recursive block training workflow` plus occasional `feat:` and `fix:` prefixes. Keep commits focused and mention the affected area when useful, for example `fix: handle missing depth maps`. Pull requests should describe the bug or experiment, list commands run, note datasets/configs used, and include metrics or screenshots when rendering changes.

## Security & Configuration Tips

Treat dataset paths, SwanLab settings, and GPU assignments as local configuration. Avoid committing credentials, absolute private paths in reusable configs, or modified generated files inside submodules.
