# VFM-Guided Multi-Dataset Multi-Task Label Completion for Road-Scene Understanding

Reference implementation for completing heterogeneous road-scene annotations
across BDD100K, Cityscapes, KITTI, and nuImages.

> **ACCV 2026.** This repository is the reference implementation for the accepted
> paper **“VFM-Guided Multi-Dataset Multi-Task Label Completion for Road-Scene
> Understanding”** by Kai-Jun Liang, Chien-Yao Wang, and Yi-Ting Huang.

## Pipeline

The canonical pipeline standardizes all four datasets to a shared 1024x512
frame and Merge52/70-node taxonomy, builds VFM dense cues, trains the Stage-I
four-task MLP, combines Stage-I outputs with signed SAM2 region cues, trains the
Stage-II six-task MLP, and exports GT-preserving integrated labels for semantic,
lane, drivable area, depth, instance, and panoptic tasks.

The learned models do not consume RGB pixels. Stage-I uses the 257-dimensional
dense feature contract. Stage-II uses the same 257 dense features plus 18
region/shape features for a 275-dimensional input. Original instance and
panoptic GT are supervision only and are never Stage-II input features.

## Repository layout

```text
vfm_pipeline/                 reusable datasets, models, losses, provenance
scripts/preprocessing/        deterministic dataset/cue preparation
scripts/training/             Stage-I and Stage-II training
scripts/export/               Stage-I and Stage-II label export
scripts/evaluation/           dense, hidden-test, and region evaluation
configs/taxonomy.pt           canonical Merge52/70-node taxonomy cache
manifests/multitask_samples.json
                              canonical four-dataset sample manifest
artifacts/acceptance/         reproducibility/acceptance evidence
checkpoints/stage1_target_masked_v1/  generated Stage-I best.pt and last.pt
checkpoints/stage2_target_masked_v1/  generated Stage-II best.pt and last.pt
outputs/pseudo/<dataset>/     semantic/lane/drivable/DA2/DA3 VFM cues
outputs/sam2/<dataset>/       signed SAM2 region masks
outputs/stage1/               Stage-I exported cues
outputs/stage2/               Stage-II/final exported labels
results/                      evaluation results
datasets/                     source data and deterministic processed GT
outputs/preprocessing/        standardized RGB / source-index preprocessing
```

Formal artifact names are stable. Version, source checkpoint, hashes, and
lineage belong in provenance metadata rather than date- or experiment-suffixed
filenames.

## Environment

Python 3.12 is the reference runtime. Install a CUDA/driver-compatible PyTorch
build first, then install the project-level packages:

```bash
python3.12 -m venv .venv_runtime
source .venv_runtime/bin/activate
# Install the appropriate PyTorch build for your GPU/CUDA from pytorch.org.
pip install -r requirements.txt
```

The repository intentionally does **not** vendor road-scene datasets, upstream
VFM repositories, or foundation-model weights. Keep those resources outside
the source repository. `config.py` supports the following environment
variables:

```text
VFM_PROJECT_ROOT                 repository root (normally auto-detected)
VFM_WORKSPACE_ROOT               common workspace root
VFM_DATASET_ROOT                 prepared road-dataset root
VFM_CODE_ROOT                    upstream VFM source-repository root
VFM_FOUNDATION_CHECKPOINT_ROOT   upstream/foundation checkpoint root
VFM_TRAINING_CHECKPOINT_ROOT     Stage-I/Stage-II checkpoint root
```

Set these to paths in the environment where Python runs (for example, Linux
paths when running under WSL). The source code does not require the
maintainer's drive letter. See [portability and prepared-data contract](docs/PORTABILITY.md)
before rebuilding a manifest at a new location.

With `VFM_WORKSPACE_ROOT=/path/to/workspace`, the default external layout is:

```text
/path/to/workspace/
├── datasets/processed/road/
│   ├── bdd100k/
│   ├── cityscapes/
│   ├── kitti/
│   └── nuimages/
├── vfm/code/
│   ├── sam2/
│   ├── sam3/
│   ├── Depth-Anything-V2/
│   └── Depth-Anything-3/
└── vfm/checkpoints/
    ├── sam2/sam2.pt
    ├── sam3/sam3.pt
    ├── depth_anything_v2_vitl.pth
    └── da3_large/model.safetensors
```

`datasets/` and `VFMs/` may be local symlinks/junctions for convenience, but
they are ignored by Git and are not required to live inside the repository.

## External datasets

This project does not redistribute any dataset. Download each dataset from its
official source and follow its license/terms of use:

- **BDD100K:** <https://github.com/bdd100k/bdd100k> and
  <https://www.bdd100k.com/>
- **Cityscapes:** <https://www.cityscapes-dataset.com/> (account/login is
  required for downloads)
- **KITTI semantic segmentation:**
  <https://www.cvlibs.net/datasets/kitti/eval_semseg.php?benchmark=semantics>
- **KITTI depth completion:**
  <https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion>
- **nuImages:** <https://github.com/nutonomy/nuscenes-devkit> (download access
  is governed by the nuScenes/nuImages terms of use)

The formal manifest treats BDD100K, Cityscapes, KITTI, and nuImages as four
peer datasets. Dataset-specific preparation is used only where required by the
source format, such as KITTI's source-indexed semantic/depth identities.

## External VFM code and checkpoints

Use the upstream repositories for model code and model-specific installation
requirements:

- **SAM 2:** <https://github.com/facebookresearch/sam2>
- **SAM 3:** <https://github.com/facebookresearch/sam3>
- **Depth Anything V2:** <https://github.com/DepthAnything/Depth-Anything-V2>
- **Depth Anything 3:** <https://github.com/ByteDance-Seed/Depth-Anything-3>

The repository does not redistribute third-party model weights. See
`THIRD_PARTY.md` and the corresponding upstream repositories for model-specific
licenses and usage terms.

## Canonical inputs

```text
manifests/multitask_samples.json
configs/taxonomy.pt
```

`configs/taxonomy.pt` is included in the source repository. The sample manifest
is built locally after the datasets and VFM cues are prepared:

```bash
python scripts/preprocessing/build_manifest.py
```

See [portability and prepared-data contract](docs/PORTABILITY.md) for details
about local paths and prepared-data layout.

## Usage

The public pipeline is organized into preprocessing, Stage-I refinement,
SAM2-guided region construction, Stage-II six-task refinement, integration,
and evaluation.

Build the canonical sample manifest after preparing the datasets and VFM cues:

```bash
python scripts/preprocessing/build_manifest.py
```

The main training entrypoints are:

```bash
python scripts/training/train_stage1.py --help
python scripts/training/train_stage2.py --help
```

For end-to-end execution, use:

```bash
./scripts/run_full_pipeline.sh --help
```

Detailed experiment settings, reproducibility checks, evaluation protocols,
and artifact contracts are kept in the configuration files and project
documentation rather than duplicated in this README.

Historical development artifacts are not part of the canonical runtime flow.

## Licensing and publication rights

### Repository source code

Project-specific source code in this repository is released under the
**MIT License**. See `LICENSE`.

The MIT License applies only to repository material for which the repository
authors hold the necessary rights. It does **not** relicense the paper,
supplementary material, external datasets, upstream VFM repositories,
foundation-model weights, or other third-party material.

### ACCV 2026 paper and supplementary material

The ACCV 2026 OpenReview record for the paper is released under the
**Creative Commons Attribution 4.0 International (CC BY 4.0)** license. The
authors retain copyright while granting the permissions defined by CC BY 4.0,
including reuse and redistribution with appropriate attribution. The paper is
therefore **not** covered by the repository MIT License.

Supplementary material is published with the paper through the ACCV/OpenReview
and CVF publication workflow and is likewise **not** relicensed by this
repository's MIT License. When redistributing a published supplementary
artifact, follow the license metadata attached to the corresponding published
record.

ACCV 2026 will additionally publish the paper in Springer's Lecture Notes in
Computer Science (LNCS) as post-conference proceedings. Springer source files
and a separate Springer publishing agreement / License-to-Publish will be
collected after the conference. This repository does not replace or modify that
publication agreement.

### Third-party assets

External datasets, upstream VFM repositories, model checkpoints, and other
third-party components remain subject to their original licenses and terms.
See `THIRD_PARTY.md` for the dependency and license inventory used by this
project.

## What is intentionally not tracked by Git

The repository `.gitignore` excludes local environments, datasets, external VFM
source links, foundation/training checkpoints, generated outputs, logs, result
files, the 1.1 GB generated manifest, and historical/internal migration
material. A small whitelist of formal acceptance evidence under
`artifacts/acceptance/` remains trackable because the strict training/preflight
contracts bind to those files.

Dataset licenses, upstream VFM source licenses, and model-weight terms remain
those of their respective owners; they are not covered by this repository's
MIT License.

## Public entrypoints

```text
scripts/preprocessing/prepare_kitti.py
scripts/preprocessing/validate_kitti.py
scripts/preprocessing/build_manifest.py
scripts/preprocessing/generate_sam3.py
scripts/preprocessing/generate_depth_v2.py
scripts/preprocessing/generate_depth_v3.py
scripts/preprocessing/generate_sam2.py
scripts/training/train_stage1.py
scripts/export/export_stage1.py
scripts/training/train_stage2.py
scripts/export/export_stage2.py
scripts/evaluation/evaluate_dense.py
scripts/evaluation/evaluate_hidden_test.py
scripts/evaluation/evaluate_regions.py
```

The dataset-specific KITTI preparation entrypoints resolve its two official
source families into stable sample identities. They do not make KITTI the
primary dataset: the canonical manifest and both learned stages operate on all
four peer datasets.
