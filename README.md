# uncertainty-jnr

Uncertainty-aware jersey number recognition using digit-compositional classifiers and Dirichlet-based uncertainty modeling. Adapted from [Grad, CVPR 2025 Workshop](https://openaccess.thecvf.com/content/CVPR2025W/CVSPORTS/papers/Grad_Single-Stage_Uncertainty-Aware_Jersey_Number_Recognition_in_Soccer_CVPRW_2025_paper.pdf).

## Environment

- Python 3.12.3
- CUDA 12.4
- Ubuntu / Linux 6.17

### Core packages

| Package | Version |
|---------|---------|
| torch | 2.4.1+cu124 |
| torchvision | 0.19.1+cu124 |
| timm | 1.0.25 |
| numpy | 1.26.4 |
| scipy | 1.17.1 |
| pydantic | 2.12.5 |
| albumentations | 1.4.24 |
| scikit-learn | 1.8.0 |
| opencv-contrib-python | 4.11.0.86 |
| numba | 0.60.0 |
| matplotlib | 3.10.8 |
| pandas | 2.3.3 |
| pillow | 12.1.1 |
| rich | 13.9.4 |
| torchdiffeq | 0.2.4 |
| flow-matching | 1.0.10 |

## Setup

```bash
git clone https://github.com/ehzawad/uncertainty-jnr.git
cd uncertainty-jnr
poetry install
poetry shell
```

## Training

```bash
# Single GPU
poetry run python train.py --config configs/final.yaml

# Multi-GPU (DDP)
poetry run torchrun --nproc_per_node=2 train.py --config configs/final.yaml
```

### Configs

| Config | Description |
|--------|-------------|
| `configs/final.yaml` | ViT-Small, folder dataset, tracklet training with spatial decoder |
| `configs/small16_final_dataset.yaml` | ViT-Small, folder dataset, single-frame |
| `configs/small16_final_dataset_v2.yaml` | ViT-Small, folder dataset, single-frame (longer schedule) |

## Inference

```bash
poetry run python predict_images.py \
    --checkpoint runs/final/best_checkpoint.pt \
    --config configs/final.yaml \
    --image-dir /path/to/images \
    --output results/predictions
```

## Evaluation

```bash
poetry run python evaluate_unseen.py \
    --checkpoint runs/final/best_checkpoint.pt \
    --config configs/final.yaml \
    --unseen-dir unseen/ \
    --gt unseen/gt.json
```

## License

[CC-BY-SA-4.0](LICENSE)
