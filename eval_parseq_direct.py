#!/usr/bin/env python3
"""Evaluate raw PARseq (zero jersey training) on unseen sports data.

This gives the baseline: how well does a pure scene text reader do
on jersey number crops without any jersey-specific training?
"""

import sys
sys.path.insert(0, "../object-detection-test/parseq")

import json
import torch
import numpy as np
from pathlib import Path
import logging
import cv2
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_parseq_model(weights_path, device):
    """Load PARseq-small via torch hub (standard image-based model)."""
    model = torch.hub.load("baudm/parseq", "parseq", pretrained=True, trust_repo=True)
    model = model.to(device).eval()
    logging.info("Loaded PARseq-small from torch hub (pretrained)")
    return model


def parseq_predict(model, image_tensor, device):
    """Run PARseq inference, return predicted string."""
    with torch.no_grad():
        logits = model(image_tensor.unsqueeze(0).to(device))
    pred = logits.softmax(-1)
    label, _ = model.tokenizer.decode(pred)
    return label[0]


def text_to_number(text):
    """Convert PARseq text prediction to jersey number (0-100)."""
    # Extract only digits
    digits = "".join(c for c in text if c.isdigit())
    if not digits:
        return 100  # absent
    num = int(digits)
    if num > 99:
        num = int(digits[:2])  # take first 2 digits
    return min(num, 100)


def load_ground_truth(json_path):
    with open(json_path) as f:
        raw = json.load(f)
    return {k: (int(v) if v is not None else 100) for k, v in raw.items()}


def evaluate_sport(model, device, segments_dir, gt, transform_size=(32, 128)):
    """Evaluate PARseq on one sport."""
    segment_ids = sorted(gt.keys(), key=lambda x: int(x))
    logging.info(f"Evaluating {len(segment_ids)} segments from {segments_dir.name}")

    correct = 0
    total = 0
    top3_correct = 0
    errors = []

    for seg_id in segment_ids:
        seg_dir = segments_dir / seg_id
        if not seg_dir.is_dir():
            continue

        all_imgs = [p for p in seg_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        if not all_imgs:
            continue

        # Run PARseq on each crop, collect digit predictions
        predictions = defaultdict(int)
        transform = model.val_transform if hasattr(model, 'val_transform') else None
        for img_path in all_imgs:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if transform is not None:
                from PIL import Image
                pil_img = Image.fromarray(img)
                tensor = transform(pil_img)
            else:
                img = cv2.resize(img, (transform_size[1], transform_size[0]))
                tensor = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
                mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
                std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
                tensor = (tensor - mean) / std

            text = parseq_predict(model, tensor, device)
            num = text_to_number(text)
            predictions[num] += 1

        if not predictions:
            continue

        # Majority vote
        pred_number = max(predictions, key=predictions.get)
        gt_number = gt[seg_id]

        # Top-3: get top 3 most voted predictions
        sorted_preds = sorted(predictions.items(), key=lambda x: -x[1])
        top3 = [p[0] for p in sorted_preds[:3]]

        total += 1
        if pred_number == gt_number:
            correct += 1
        else:
            errors.append({"segment": seg_id, "gt": gt_number, "pred": pred_number, "top3": top3})

        if gt_number in top3:
            top3_correct += 1

    acc = correct / total * 100 if total > 0 else 0
    top3_acc = top3_correct / total * 100 if total > 0 else 0
    return {"total": total, "correct": correct, "accuracy": acc, "top3_correct": top3_correct, "top3_accuracy": top3_acc, "errors": errors}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_path = Path("../object-detection-test/parseq/pretrained/parseq_small.pt")
    unseen_dir = Path("unseen")

    model = load_parseq_model(weights_path, device)

    sports = [
        {"name": "Basketball", "segments_dir": unseen_dir / "jersey_nr_segments_staidum_basketball",
         "gt_file": unseen_dir / "jersey_nr_corrections_staidium_basketball.json"},
        {"name": "Ice Hockey", "segments_dir": unseen_dir / "jersey_nr_segments_staidium_ice_hockey",
         "gt_file": unseen_dir / "jersey_nr_corrections_staidium_ice_hockey.json"},
    ]

    for sport in sports:
        gt = load_ground_truth(sport["gt_file"])
        results = evaluate_sport(model, device, sport["segments_dir"], gt)
        logging.info(f"\n--- {sport['name']} ---")
        logging.info(f"Top-1: {results['accuracy']:.2f}% ({results['correct']}/{results['total']})")
        logging.info(f"Top-3: {results['top3_accuracy']:.2f}% ({results['top3_correct']}/{results['total']})")


if __name__ == "__main__":
    main()
