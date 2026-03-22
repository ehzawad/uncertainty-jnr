#!/usr/bin/env python3
"""Evaluate STN model on jr_nr_test data (zip-based tracklets)."""

import json
import torch
import numpy as np
from pathlib import Path
import logging
import zipfile
import io
import cv2
from collections import defaultdict

from uncertainty_jnr.stn_model import STNJerseyModel
from uncertainty_jnr.utils import load_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_ground_truth(json_path):
    with open(json_path) as f:
        raw = json.load(f)
    return {k: (int(v) if v is not None else 100) for k, v in raw.items()}


def load_images_from_zip(zip_path, target_size=(224, 224)):
    """Load and preprocess all images from a zip file."""
    images = []
    with zipfile.ZipFile(zip_path) as z:
        for name in sorted(z.namelist()):
            if not name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            data = z.read(name)
            arr = np.frombuffer(data, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (target_size[1], target_size[0]), interpolation=cv2.INTER_CUBIC)
            tensor = torch.from_numpy(img.transpose(2, 0, 1)).float() / 127.5 - 1.0
            images.append(tensor)
    return images


def run_inference(model, device, images, batch_size=16):
    """Run inference on images, return alphas and uncertainties."""
    all_alphas = []
    all_uncertainties = []
    for i in range(0, len(images), batch_size):
        batch = torch.stack(images[i:i + batch_size]).to(device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            output = model(batch)
        alpha = torch.exp(output.all_logits) + 1.0
        all_alphas.append(alpha.cpu())
        unc = output.uncertainty.cpu()
        if unc.ndim == 0:
            unc = unc.unsqueeze(0)
        all_uncertainties.append(unc)
    return torch.cat(all_alphas), torch.cat(all_uncertainties)


def aggregate_predictions(alphas, uncertainties):
    """Uncertainty-weighted alpha aggregation with filtering."""
    n = alphas.size(0)
    if n >= 4:
        keep_n = max(2, int(n * 0.75))
        _, keep_idx = uncertainties.topk(keep_n, largest=False)
        alphas = alphas[keep_idx]
        uncertainties = uncertainties[keep_idx]

    weights = 1.0 / (uncertainties + 1e-6)
    weights = weights / weights.sum()
    weighted_alphas = alphas * weights.unsqueeze(1)
    summed_alpha = weighted_alphas.sum(dim=0)
    return summed_alpha / summed_alpha.sum()


def evaluate_dataset(model, device, segments_dir, gt, batch_size=128):
    """Evaluate on one dataset (zip-based segments). Batched for speed."""
    segment_ids = sorted(gt.keys(), key=lambda x: int(x))
    logging.info(f"Evaluating {len(segment_ids)} segments from {segments_dir.name}")

    # Pre-load all images grouped by segment
    seg_images = {}
    for seg_id in segment_ids:
        zips = list(segments_dir.glob(f"{seg_id}_*.zip"))
        if not zips:
            continue
        images = []
        for zp in zips:
            imgs = load_images_from_zip(zp)
            # Sample max 8 frames per zip to speed up
            if len(imgs) > 8:
                indices = np.linspace(0, len(imgs) - 1, 8, dtype=int)
                imgs = [imgs[i] for i in indices]
            images.extend(imgs)
        if images:
            seg_images[seg_id] = images

    # Batch inference: collect all images, run in large batches
    all_tensors = []
    seg_ranges = {}  # seg_id -> (start, end) in all_tensors
    for seg_id, images in seg_images.items():
        start = len(all_tensors)
        all_tensors.extend(images)
        seg_ranges[seg_id] = (start, len(all_tensors))

    logging.info(f"Total images: {len(all_tensors)}, segments with images: {len(seg_images)}")

    # Run batched inference
    all_alphas = []
    all_uncertainties = []
    for i in range(0, len(all_tensors), batch_size):
        batch = torch.stack(all_tensors[i:i + batch_size]).to(device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            output = model(batch)
        alpha = torch.exp(output.all_logits) + 1.0
        all_alphas.append(alpha.cpu())
        unc = output.uncertainty.cpu()
        if unc.ndim == 0:
            unc = unc.unsqueeze(0)
        all_uncertainties.append(unc)
        del output, batch
        torch.cuda.empty_cache()

    all_alphas = torch.cat(all_alphas)
    all_uncertainties = torch.cat(all_uncertainties)

    # Aggregate per segment
    correct = 0
    total = 0
    top3_correct = 0
    errors = []

    for seg_id in seg_images:
        start, end = seg_ranges[seg_id]
        alphas = all_alphas[start:end]
        uncertainties = all_uncertainties[start:end]

        agg_probs = aggregate_predictions(alphas, uncertainties)
        gt_number = gt[seg_id]

        pred_number = agg_probs.argmax().item()
        top3 = agg_probs.topk(3).indices.tolist()

        total += 1
        if pred_number == gt_number:
            correct += 1
        else:
            errors.append({
                "segment": seg_id, "gt": gt_number, "pred": pred_number,
                "top3": top3, "n_crops": end - start,
            })
        if gt_number in top3:
            top3_correct += 1

    acc = correct / total * 100 if total > 0 else 0
    top3_acc = top3_correct / total * 100 if total > 0 else 0
    return {"total": total, "correct": correct, "accuracy": acc,
            "top3_correct": top3_correct, "top3_accuracy": top3_acc,
            "errors": errors}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dir = Path("jr_nr_test")

    # Load STN model
    model = STNJerseyModel()
    load_checkpoint(model, Path("runs/stn/best_checkpoint.pt"), device, strict=False)
    model = model.to(device).eval()
    logging.info("STN model loaded")

    # Find all dataset pairs
    datasets = []
    for gt_file in sorted(test_dir.glob("jersey_nr_corrections_*.json")):
        name = gt_file.stem.replace("jersey_nr_corrections_", "")
        seg_dir = test_dir / f"jersey_nr_segments_{name}"
        if seg_dir.is_dir():
            datasets.append({"name": name, "gt_file": gt_file, "segments_dir": seg_dir})

    for ds in datasets:
        gt = load_ground_truth(ds["gt_file"])
        n_absent = sum(1 for v in gt.values() if v == 100)
        logging.info(f"\n{'='*60}")
        logging.info(f"Dataset: {ds['name']} ({len(gt)} segments, {n_absent} absent)")
        logging.info(f"{'='*60}")

        results = evaluate_dataset(model, device, ds["segments_dir"], gt)
        logging.info(f"Top-1: {results['accuracy']:.2f}% ({results['correct']}/{results['total']})")
        logging.info(f"Top-3: {results['top3_accuracy']:.2f}% ({results['top3_correct']}/{results['total']})")

        if results["errors"]:
            logging.info(f"Errors ({len(results['errors'])}):")
            for e in results["errors"][:10]:
                logging.info(f"  Seg {e['segment']}: GT={e['gt']}, Pred={e['pred']}, Top3={e['top3']}, Crops={e['n_crops']}")
            if len(results["errors"]) > 10:
                logging.info(f"  ... and {len(results['errors']) - 10} more")


if __name__ == "__main__":
    main()
