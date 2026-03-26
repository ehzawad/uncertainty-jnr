"""Shared inference utilities for jersey number evaluation."""

import torch
from collections import Counter


def aggregate_predictions(alphas, uncertainties):
    """Aggregate per-crop Dirichlet alphas via uncertainty-weighted summation.

    Two-stage: (1) filter top-25% most uncertain crops, (2) uncertainty-weighted sum.
    Standard argmax over all 101 classes — absent competes directly with numbers.

    Args:
        alphas: (N, 101) Dirichlet alpha parameters per crop
        uncertainties: (N,) uncertainty scores per crop

    Returns:
        agg_probs: (101,) aggregated probability vector
    """
    n = alphas.size(0)

    # Stage 1: filter out top-25% most uncertain crops
    if n >= 4:
        keep_n = max(2, int(n * 0.75))
        _, keep_idx = uncertainties.topk(keep_n, largest=False)
        alphas = alphas[keep_idx]
        uncertainties = uncertainties[keep_idx]

    # Stage 2: uncertainty-weighted alpha summation
    weights = 1.0 / (uncertainties + 1e-6)
    weights = weights / weights.sum()

    weighted_alphas = alphas * weights.unsqueeze(1)
    summed_alpha = weighted_alphas.sum(dim=0)  # (101,)

    return summed_alpha / summed_alpha.sum()


def digit_level_voting(alphas, uncertainties):
    """Digit-level voting aggregation for compositional robustness.

    Instead of voting on the full 100-class number, decomposes each frame's
    prediction into individual digits and votes on tens and ones separately.
    This handles cases like frame1="23", frame2="28" → tens=2, ones=vote(3,8).

    Args:
        alphas: (N, 101) Dirichlet alpha parameters per crop
        uncertainties: (N,) uncertainty scores per crop

    Returns:
        agg_probs: (101,) probability vector with winning number
    """
    n = alphas.size(0)

    per_frame_probs = alphas / alphas.sum(dim=1, keepdim=True)
    per_frame_pred = per_frame_probs[:, :100].argmax(dim=1)  # (N,) numbers 0-99
    per_frame_conf = per_frame_probs[:, :100].max(dim=1).values
    absent_frac = per_frame_probs[:, 100].mean().item()

    # If majority of frames point to absent
    if absent_frac > 0.5:
        result = torch.zeros(101)
        result[100] = 1.0
        return result

    # Decompose each frame's prediction into digits
    tens_votes = Counter()
    ones_votes = Counter()
    single_vs_double = Counter()  # True=single, False=double

    for i in range(n):
        pred = per_frame_pred[i].item()
        conf = per_frame_conf[i].item()
        w = conf / (uncertainties[i].item() + 1e-6)

        if pred < 10:
            # Single digit
            tens_votes[pred] += w  # the digit itself
            single_vs_double[True] += w
        else:
            # Two digit
            tens_d = pred // 10
            ones_d = pred % 10
            tens_votes[tens_d] += w
            ones_votes[ones_d] += w
            single_vs_double[False] += w

    # Decide single vs double
    is_single = single_vs_double.get(True, 0) > single_vs_double.get(False, 0)

    if is_single:
        # Single digit: pick most voted digit
        if tens_votes:
            winner = max(tens_votes, key=tens_votes.get)
        else:
            winner = 0
    else:
        # Two digit: compose from winning tens + winning ones
        if tens_votes and ones_votes:
            best_tens = max(tens_votes, key=tens_votes.get)
            best_ones = max(ones_votes, key=ones_votes.get)
            winner = best_tens * 10 + best_ones
        elif tens_votes:
            winner = max(tens_votes, key=tens_votes.get) * 10
        else:
            winner = 0

    result = torch.zeros(101)
    result[min(winner, 99)] = 1.0
    return result
