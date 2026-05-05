# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Repository purpose

Personal learning journal for computer vision applied to industrial inspection.
Each week's work lives in its own top-level directory (`weekNN_topic/`)
containing experiment code, configs, notebooks, and a short retrospective.
This is a journal, not a shipped product — code does not need to be
backwards-compatible across weeks, and each week is largely self-contained.

Most source files are currently empty scaffolds. When asked to add to a week,
fill in the existing files rather than creating new top-level ones.

## Domain context

The target domain is **grayscale industrial imaging** with characteristics
similar to medical imaging: low contrast, single-channel, often high resolution.
When suggesting techniques, prefer those proven on medical or grayscale data
(e.g. nnU-Net, U-Net variants, SAM2) over generic RGB-photo techniques.

The downstream task is **segmentation-based defect detection**, NOT bounding
box detection. Defects of interest are often elongated or thin (think:
crack-like linear features), where bounding boxes contain mostly background
and lose information. Treat segmentation as the default; only suggest detection
when there's a specific reason.

## License constraint (HARD CONSTRAINT)

Only suggest libraries and pretrained models with **Apache 2.0, MIT, or BSD**
licenses. Do NOT suggest:

- Ultralytics YOLO (any version) — AGPL-3.0
- Any GPL or AGPL licensed library
- Pretrained weights with non-commercial license terms in the final pipeline

When recommending a model or library, briefly state its license. If the user
asks for "the best model for X" and a strong candidate is GPL/AGPL, explicitly
flag the license issue and suggest the best license-compatible alternative.

## Per-week layout convention

Each `weekNN_*/` directory follows this structure:

- `src/` — Python package (`data.py`, `model.py`, `train.py`, `utils.py`, `__init__.py`)
- `configs/` — YAML configs; keep hyperparameters here, not hardcoded in `src/`
- `notebooks/` — exploratory work (EDA, visualization). Production training code
  belongs in `src/`, not notebooks.
- `outputs/` — runtime artifacts (checkpoints, logs). **Gitignored.**
- `requirements.txt` — week-specific pinned dependencies. Each week has its own.
- `README.md` — week retrospective.

When adding a new week, mirror this structure exactly.

## Stack

- **Framework**: PyTorch (BSD), used raw — NOT PyTorch Lightning
- **Models**: timm (Apache 2.0) for backbones; segmentation_models_pytorch (MIT) for seg heads
- **Augmentation**: Albumentations (MIT)
- **Logging**: TensorBoard (Apache 2.0)
- **Format**: ONNX for deployment-style experiments

Prefer these over hand-rolled equivalents (e.g. use `timm.create_model`
rather than writing a backbone from scratch; use Albumentations rather than
`torchvision.transforms`).

## Hardware

- Primary dev machine: RTX 5090 (Blackwell, sm_120) on Ubuntu, in a corporate
  network environment. Requires PyTorch nightly with CUDA 12.8+.
- Portable: RTX 4090 laptop, used on personal network.

## Environment

- Conda environment for this learning repo: **`rtai-learning`** (Python 3.11).
- Activated with: `conda activate rtai-learning`.
- PyTorch nightly is pinned to `2.12.0.dev+cu128`.
- A separate environment `pytorch_1` exists for legacy/other work; do NOT
  install rtai-learning dependencies into it.

## Known dependency notes

- `pandas` is pinned to `<3.0` because `pandas 3.0.x` causes a NumPy 2.4 ABI
  segfault during install. See `week01_first_training/requirements.txt`.

When suggesting installs or commands, assume the 5090 + nightly PyTorch
context with the `rtai-learning` conda env unless the user says otherwise.
If a fix requires a stable PyTorch release, flag it explicitly.

## Code conventions

- Type hints where they help readability; not mandatory everywhere
- Docstrings: English preferred for code, Korean OK for inline comments
- Always include a sanity check before full training (1-batch overfit test)
- Configs in YAML, separated from code
- `if __name__ == "__main__"` block in data/model files for quick standalone smoke tests

## Things that must not be committed

`.gitignore` is strict because training artifacts are large. Never `git add`:

- `**/outputs/`, `**/data/`, `**/checkpoints/`
- Model weights: `*.pt`, `*.pth`, `*.ckpt`, `*.onnx`, `*.safetensors`
- `runs/`, `wandb/`, `lightning_logs/`, `*.log`
- `CLAUDE.local.md` (personal scratch — untracked on purpose)
- `*_internal.md`, `*_private.md`, `notes_company/`, `secrets/` — used to keep
  work-context material out of this repo. If the user mentions company or
  internal info, suggest using one of these filenames so it stays untracked.

If a file pattern looks like training output, verify with `git check-ignore`
before staging.

## Communication style

- Korean responses are preferred for explanations
- Be direct and critical when you spot issues; do not soften technical
  problems with vague language
- For architecture or model decisions, present trade-offs explicitly
- When recommending a library, always include its license
- Cite official docs when discussing version-specific behavior

## Current state

Currently in **Week 01: First Training Pipeline**.
Goal: end-to-end PyTorch training loop on grayscale data using MedMNIST
(PneumoniaMNIST → ChestMNIST). Deliverable: `python -m src.train` runs the
full pipeline with TensorBoard logging.

Most files in `week01_first_training/` are empty scaffolds waiting to be
filled in. Do not create new files in that directory; fill the existing ones.