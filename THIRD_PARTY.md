# Third-Party Software, Models, and Datasets

This repository contains the project-specific pipeline code for
VFM-guided multi-dataset multi-task label completion. It does **not**
redistribute the external datasets, upstream VFM repositories, or foundation
model weights used by the experiments.

Users are responsible for obtaining those resources from their original
providers and complying with the applicable licenses and terms.

## Foundation models

| Component | Upstream source | License / terms relevant to this project |
| --- | --- | --- |
| SAM 2 | <https://github.com/facebookresearch/sam2> | Apache License 2.0. Consult the upstream repository for notices and third-party components. |
| SAM 3 | <https://github.com/facebookresearch/sam3> | SAM License. Consult the upstream `LICENSE` before use or redistribution. |
| Depth Anything V2 | <https://github.com/DepthAnything/Depth-Anything-V2> | The upstream project states that the Small model is Apache-2.0, while Base/Large/Giant model weights are CC-BY-NC-4.0. This project uses the **Large** model. |
| Depth Anything 3 | <https://github.com/ByteDance-Seed/Depth-Anything-3> | Apache License 2.0 for the upstream code. The [DA3-LARGE checkpoint](https://huggingface.co/depth-anything/DA3-LARGE) used by this project is released under **CC BY-NC 4.0**. |

The repository's own MIT License applies only to code and other material for
which the repository authors hold the necessary rights. It does not relicense
any external model, checkpoint, dataset, upstream repository, or third-party
dependency.

## Datasets

The project consumes BDD100K, Cityscapes, KITTI, and nuImages. None of those
datasets is redistributed here. Obtain them from the official sources linked in
`README.md` and comply with their individual licenses, access requirements, and
terms of use.

## Python packages

`requirements.txt` lists project-level Python packages. PyTorch is installed
separately so users can select a build compatible with their CUDA/driver stack.
SAM 2, SAM 3, Depth Anything V2, and Depth Anything 3 also have their own
upstream dependency requirements; install those in accordance with the
corresponding upstream repositories.

This document is an inventory and does not replace the original license texts
or terms supplied by the respective copyright holders.
