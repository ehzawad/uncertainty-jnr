# uncertainty-jnr

Uncertainty-aware jersey number recognition via spatial localization and frozen scene text reading. A Spatial Transformer Network (STN) learns to localize jersey numbers in player crops, then a frozen [PARseq](https://github.com/baudm/parseq) reads the digits -- all with Dirichlet-based uncertainty estimation.

Adapted from [Grad, "Single-Stage Uncertainty-Aware Jersey Number Recognition in Soccer", CVPR 2025 Workshop](https://openaccess.thecvf.com/content/CVPR2025W/CVSPORTS/papers/Grad_Single-Stage_Uncertainty-Aware_Jersey_Number_Recognition_in_Soccer_CVPRW_2025_paper.pdf).

## Architecture

```
Image (224x224)
  |
  v
Frozen ViT-Small ──> 14x14 patch features
  |                         |
  v                         v
LocalizationHead        AbsentClassifierHead
(spatial soft-argmax)   (is number visible?)
  |
  v
Differentiable crop (grid_sample) ──> tight number region (32x128)
  |
  v
Frozen PARseq ──> per-position character logits
  |
  v
NumberCompositionHead (log-space digit composition) ──> 100-class logits (0-99)
  |
  v
Dirichlet uncertainty (softplus alpha + 1)
```

**Trainable parameters** (~200K): LocalizationHead, AbsentClassifierHead, NumberCompositionHead length bias, and a log-temperature scalar. The ViT backbone and PARseq are frozen.

### Key ideas

- **Spatial soft-argmax localization**: Attention over the 14x14 patch grid predicts a differentiable crop center via soft-argmax. Crop size is predicted from the attended feature. Gradients flow through `grid_sample` to train the localization head end-to-end.
- **Log-space digit composition**: PARseq character logits are composed into jersey numbers in log-space (`log P(tens) + log P(ones)`) to avoid single-digit bias from probability multiplication.
- **Dirichlet uncertainty**: Softplus-activated evidence parameters model epistemic uncertainty. Type II Maximum Likelihood loss trains the Dirichlet, with KL regularization against a uniform prior for absent/unknown samples.
- **Absent detection**: A separate binary classifier on ViT patch features detects when no jersey number is visible (class 100), since PARseq always tries to read something.

## Setup

```bash
git clone https://github.com/ehzawad/uncertainty-jnr.git
cd uncertainty-jnr
poetry install
poetry shell
```

Requires Python 3.12+ and CUDA 12.4. PyTorch is pulled from the `torch+cu124` index (see `pyproject.toml`).

## Training

```bash
# Single GPU
python scripts/train.py --config configs/stn.yaml

# Multi-GPU (DDP)
torchrun --nproc_per_node=2 scripts/train.py --config configs/stn.yaml
```

Training uses AdamW with cosine annealing, AMP, gradient clipping, early stopping, and Stochastic Weight Averaging (SWA) in the final 25% of epochs. Checkpoints are saved to `runs/<experiment_name>/`.

### Config: `configs/stn.yaml`

| Section | Key settings |
|---------|-------------|
| **Data** | Folder dataset, 224x224, batch 512, single-frame mode |
| **Model** | `vit_small_patch16_224.augreg_in21k`, STN architecture |
| **Loss** | Dirichlet with decoder auxiliary CE (weight 0.5), KL warmup 500 steps |
| **Training** | 30 epochs, LR 1e-4, weight decay 5e-3, early stopping patience 5 |

## Evaluation

### Cross-domain evaluation (basketball, ice hockey)

```bash
python scripts/evaluate_unseen.py \
    --config configs/stn.yaml \
    --checkpoint runs/stn/best_checkpoint.pt \
    --unseen-dir unseen/
```

Evaluates on segment-level data with multi-crop aggregation: filters the top-25% most uncertain crops, then performs uncertainty-weighted Dirichlet alpha summation.

### Zip-based tracklet evaluation

```bash
python scripts/eval_jr_nr_test.py
```

### PARseq baseline (zero jersey training)

```bash
python scripts/eval_parseq_direct.py
```

## Inference details

Two aggregation strategies for multi-crop segments:

- **Uncertainty-weighted alpha summation** (`aggregate_predictions`): Filters out the most uncertain 25% of crops, then sums Dirichlet alphas weighted by inverse uncertainty.
- **Digit-level voting** (`digit_level_voting`): Decomposes each frame's prediction into individual digits, votes on tens and ones separately. Handles cases like frame1="23", frame2="28" -> tens=2, ones=vote(3,8).

## Project structure

```
uncertainty-jnr/
  config.py                        # Pydantic configuration schema
  configs/stn.yaml                 # Training config for STN model
  scripts/
    train.py                       # Training loop (single/multi-GPU DDP)
    evaluate_unseen.py             # Cross-domain segment evaluation
    eval_jr_nr_test.py             # Zip-based tracklet evaluation
    eval_parseq_direct.py          # Raw PARseq baseline
  src/uncertainty_jnr/
    model.py                       # STNJerseyModel, LocalizationHead, AbsentClassifierHead, NumberCompositionHead
    loss.py                        # Type2DirichletLoss, SoftmaxWithUncertaintyLoss
    data.py                        # JerseyNumberDataset, FolderJerseyDataset, TrackletDataset, DynamicBatchSampler
    datasets.py                    # Dataset registry (SoccerNet matches, folder configs)
    augmentation.py                # DominantColorShift, JerseyCrop, RandomScaling, train/val transforms
    preprocessing.py               # adaptive_resize, letterbox_resize
    inference.py                   # aggregate_predictions, digit_level_voting
    utils.py                       # Checkpointing, seeding, logging
  pyproject.toml                   # Poetry dependencies
```

## License

[CC BY-NC-SA 4.0](LICENSE)
