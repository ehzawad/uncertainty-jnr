import sys; sys.path.insert(0, ".")
#!/usr/bin/env python3
"""Evaluate trained model on unseen segment-level data with ground truth.

For each segment folder (containing multiple crops of the same player),
runs inference on all crops, aggregates predictions via Dirichlet alpha
summation, and compares the final prediction against ground truth.
"""

import json
import torch
import numpy as np
from pathlib import Path
import logging
import argparse
import os
import cv2
from collections import defaultdict

from uncertainty_jnr.model import STNJerseyModel
from uncertainty_jnr.inference import aggregate_predictions
from uncertainty_jnr.preprocessing import letterbox_resize
from uncertainty_jnr.augmentation import get_val_transforms
from uncertainty_jnr.utils import load_checkpoint, setup_logging
from config import Config


def _extract_frame_id(path):
    """Extract numerical frame ID from filename."""
    parts = path.stem.split("_")
    if len(parts) >= 2:
        try:
            return int(parts[1])
        except ValueError:
            pass
    for part in parts:
        try:
            return int(part)
        except ValueError:
            continue
    return 0


def load_ground_truth(json_path: Path) -> dict[str, int]:
    """Load ground truth. null entries map to class 100 (absent / '-')."""
    with open(json_path) as f:
        raw = json.load(f)
    # null → 100 (absent class), matching the '-' folder convention in training data
    return {k: (int(v) if v is not None else 100) for k, v in raw.items()}


def load_and_preprocess(image_path: Path, target_size, transform):
    """Load a single image and return preprocessed tensor.

    Matches the normalization used by SimpleImageDataset._load_image:
    pixel values mapped to [-1, 1] via (x / 127.5) - 1.0
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if transform is not None:
        transformed = transform(image=img)
        img = transformed["image"]
    else:
        img = cv2.resize(img, (target_size[1], target_size[0]), interpolation=cv2.INTER_CUBIC)

    # Convert to tensor: HWC uint8 -> CHW float, normalize to [-1, 1]
    img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 127.5 - 1.0
    return img


def run_inference_on_segment(model, device, images, batch_size):
    """Run inference on a list of image tensors, return alphas and uncertainties.

    Uses all_logits (101 classes: 0-99 + absent) for alpha computation,
    so the model can potentially predict 'absent' (class 100).
    """
    all_alphas = []
    all_uncertainties = []
    for i in range(0, len(images), batch_size):
        batch = torch.stack(images[i:i + batch_size]).to(device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            output = model(batch)
        # Use all 101 logits (including absent class) for alpha computation
        alpha = torch.exp(output.all_logits) + 1.0  # (B, 101)
        all_alphas.append(alpha.cpu())
        unc = output.uncertainty.cpu()
        if unc.ndim == 0:
            unc = unc.unsqueeze(0)
        all_uncertainties.append(unc)
        del output, batch
        torch.cuda.empty_cache()
    return torch.cat(all_alphas, dim=0), torch.cat(all_uncertainties, dim=0)



# aggregate_predictions imported from uncertainty_jnr.inference


def evaluate_sport(
    model, device, segments_dir: Path, gt: dict[str, int],
    target_size, transform, batch_size: int = 64,
):
    """Evaluate model on one sport's segments."""
    segment_ids = sorted(gt.keys(), key=lambda x: int(x))
    logging.info(f"Evaluating {len(segment_ids)} segments from {segments_dir.name}")

    correct = 0
    total = 0
    top3_correct = 0
    errors = []

    for seg_id in segment_ids:
        seg_dir = segments_dir / seg_id
        if not seg_dir.is_dir():
            logging.warning(f"Segment dir missing: {seg_dir}")
            continue

        # Collect all images in this segment, sorted by numerical frame ID
        all_imgs = [p for p in seg_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        image_paths = sorted(all_imgs, key=lambda p: _extract_frame_id(p))
        if not image_paths:
            logging.warning(f"No images in segment {seg_id}")
            continue

        # Load and preprocess all crops
        images = []
        for ip in image_paths:
            tensor = load_and_preprocess(ip, target_size, transform)
            if tensor is not None:
                images.append(tensor)

        if not images:
            continue

        # Run inference
        alphas, uncertainties = run_inference_on_segment(model, device, images, batch_size)

        # Uncertainty-weighted aggregation
        agg_probs = aggregate_predictions(alphas, uncertainties)
        gt_number = gt[seg_id]

        # All 101 classes: 0-99 jersey numbers + 100 absent
        pred_number = agg_probs.argmax().item()

        # Top-3 across all classes
        top3 = agg_probs.topk(3).indices.tolist()

        total += 1
        if pred_number == gt_number:
            correct += 1
        else:
            errors.append({
                "segment": seg_id,
                "gt": gt_number,
                "pred": pred_number,
                "top3": top3,
                "n_crops": len(images),
                "confidence": agg_probs[pred_number].item(),
                "mean_uncertainty": uncertainties.mean().item(),
            })

        if gt_number in top3:
            top3_correct += 1

    acc = correct / total * 100 if total > 0 else 0
    top3_acc = top3_correct / total * 100 if total > 0 else 0
    return {
        "total": total,
        "correct": correct,
        "accuracy": acc,
        "top3_correct": top3_correct,
        "top3_accuracy": top3_acc,
        "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on unseen segment data")
    parser.add_argument("--config", type=Path, default=Path("configs/small16_final_dataset.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/small16_final_dataset/best_checkpoint.pt"))
    parser.add_argument("--unseen-dir", type=Path, default=Path("unseen"))
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    project_root = Path(os.getenv("OCR_DIR", ".")).resolve()

    # Setup logging
    output_dir = project_root / "results" / "unseen_eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir, filename="evaluate_unseen.log")

    config = Config.from_yaml(args.config)

    # Resolve paths
    checkpoint_path = args.checkpoint if args.checkpoint.is_absolute() else project_root / args.checkpoint
    unseen_dir = args.unseen_dir if args.unseen_dir.is_absolute() else project_root / args.unseen_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")

    # Load model
    use_stn = getattr(config.model, "use_stn", False)
    model = STNJerseyModel(vit_model_name=config.model.model_name)
    load_checkpoint(model, checkpoint_path, device, strict=False)
    model = model.to(device)
    model.eval()
    logging.info("Model loaded")

    transform = get_val_transforms(
        target_size=config.data.target_size,
        interpolation_method=config.data.interpolation_method,
    )

    # Define evaluation tasks
    sports = [
        {
            "name": "Basketball",
            "segments_dir": unseen_dir / "jersey_nr_segments_staidum_basketball",
            "gt_file": unseen_dir / "jersey_nr_corrections_staidium_basketball.json",
        },
        {
            "name": "Ice Hockey",
            "segments_dir": unseen_dir / "jersey_nr_segments_staidium_ice_hockey",
            "gt_file": unseen_dir / "jersey_nr_corrections_staidium_ice_hockey.json",
        },
    ]

    for sport in sports:
        logging.info(f"\n{'='*60}")
        logging.info(f"Evaluating: {sport['name']}")
        logging.info(f"{'='*60}")

        gt = load_ground_truth(sport["gt_file"])
        n_absent = sum(1 for v in gt.values() if v == 100)
        logging.info(f"Ground truth: {len(gt)} segments ({n_absent} absent/null)")

        results = evaluate_sport(
            model, device, sport["segments_dir"], gt,
            config.data.target_size, transform, args.batch_size,
        )

        logging.info(f"\n--- {sport['name']} Results ---")
        logging.info(f"Segments evaluated: {results['total']}")
        logging.info(f"Top-1 Accuracy: {results['accuracy']:.2f}% ({results['correct']}/{results['total']})")
        logging.info(f"Top-3 Accuracy: {results['top3_accuracy']:.2f}% ({results['top3_correct']}/{results['total']})")

        # Print errors summary
        if results["errors"]:
            logging.info(f"\nMisclassified segments ({len(results['errors'])}):")
            for e in results["errors"][:30]:
                logging.info(
                    f"  Segment {e['segment']}: GT={e['gt']}, Pred={e['pred']}, "
                    f"Top3={e['top3']}, Crops={e['n_crops']}, Conf={e['confidence']:.3f}"
                )
            if len(results["errors"]) > 30:
                logging.info(f"  ... and {len(results['errors']) - 30} more errors")

        # Save detailed results
        result_file = output_dir / f"{sport['name'].lower().replace(' ', '_')}_results.json"
        with open(result_file, "w") as f:
            json.dump(results, f, indent=2)
        logging.info(f"Detailed results saved to {result_file}")

    print("\nDone. Check results/unseen_eval/ for detailed output.")


if __name__ == "__main__":
    main()
