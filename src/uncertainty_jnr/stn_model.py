"""Spatial Transformer + Frozen PARseq for cross-domain jersey number recognition.

Architecture:
    Image (224x224) → Frozen ViT → patch features
                    → Localization MLP → crop box (cy, cx, h, w)
                    → Differentiable crop (grid_sample) → tight number region (32x128)
                    → Frozen PARseq → digit string
                    → Number composition + Dirichlet uncertainty

Only the localization MLP and composition head train.
Everything else is frozen pretrained weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class STNModelOutput:
    """Output from the STN model."""
    all_logits: torch.Tensor      # (B, 101) — 0-99 + absent
    number_logits: torch.Tensor   # (B, 100) — 0-99
    probs: torch.Tensor           # (B, 101)
    uncertainty: torch.Tensor     # (B,)
    predicted_number: torch.Tensor  # (B,)
    crop_params: torch.Tensor     # (B, 4) — cy, cx, h, w for visualization
    decoder_pos_logits: Optional[torch.Tensor] = None  # for aux loss compatibility


class LocalizationHead(nn.Module):
    """Predicts crop box from ViT patch features.

    Takes 196 patch tokens (384-dim each) and predicts where the
    jersey number is: (center_y, center_x, height, width) in [0, 1].

    Uses attention pooling to weight patches by importance before predicting.
    """

    def __init__(self, embed_dim: int = 384, hidden_dim: int = 128):
        super().__init__()
        # Attention pooling: learn which patches are informative for localization
        self.attn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        # Predict crop params from pooled features
        self.head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 4),  # cy, cx, h, w
        )
        # Initialize to center crop (bias toward middle of image, moderate size)
        nn.init.zeros_(self.head[-1].weight)
        self.head[-1].bias.data = torch.tensor([0.0, 0.0, 0.0, 1.0])

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch_tokens: (B, 196, 384)
        Returns:
            crop_params: (B, 4) — sigmoid-activated (cy, cx, h, w) in [0, 1]
        """
        # Attention-weighted pooling
        attn_weights = self.attn(patch_tokens)  # (B, 196, 1)
        attn_weights = F.softmax(attn_weights, dim=1)  # normalize over patches
        pooled = (patch_tokens * attn_weights).sum(dim=1)  # (B, 384)

        # Predict crop params
        raw = self.head(pooled)  # (B, 4)
        params = torch.sigmoid(raw)  # all in [0, 1]
        return params


class NumberCompositionHead(nn.Module):
    """Converts PARseq character logits to 101-class jersey number logits.

    PARseq outputs (B, max_len, charset_size) character logits.
    We take the first 2 decoded positions, extract digit probabilities (0-9),
    and compose into 101-class number logits (0-99 + absent).
    """

    def __init__(self, charset_size: int = 95, max_positions: int = 2):
        super().__init__()
        self.charset_size = charset_size
        self.max_positions = max_positions
        # Digit indices in PARseq charset: '0'-'9' are indices 0-9 after BOS/EOS/PAD
        # Standard PARseq charset: 0123456789abcdefg... digits are first 10 chars
        # But charset includes [BOS], [EOS], [PAD] tokens managed by the tokenizer

    def forward(self, parseq_logits: torch.Tensor, eos_idx: int = 0) -> torch.Tensor:
        """
        Args:
            parseq_logits: (B, max_label_len, charset_size) raw logits from PARseq
            eos_idx: index of EOS token in charset
        Returns:
            number_logits: (B, 101) composed jersey number logits
        """
        B = parseq_logits.size(0)
        device = parseq_logits.device

        # Get probabilities for each position
        probs = parseq_logits.softmax(dim=-1)  # (B, max_len, charset)

        # Digit probs: indices 1-10 in PARseq charset (after EOS at 0)
        # PARseq charset: [EOS] + charset_train, so '0' is index 1, '9' is index 10
        digit_probs_pos1 = probs[:, 0, 1:11]  # (B, 10) — first char position
        digit_probs_pos2 = probs[:, 1, 1:11]  # (B, 10) — second char position

        # EOS probability at each position (indicates sequence ended)
        eos_prob_pos1 = probs[:, 0, 0:1]  # (B, 1) — EOS at first position = absent
        eos_prob_pos2 = probs[:, 1, 0:1]  # (B, 1) — EOS at second = single digit

        # Compose 101-class logits
        # Single digit (0-9): pos1 is the digit AND pos2 is EOS
        single_digit = digit_probs_pos1 * eos_prob_pos2  # (B, 10)

        # Two digit (10-99): pos1 is tens, pos2 is ones
        tens_idx = torch.div(torch.arange(10, 100, device=device), 10, rounding_mode="floor")
        ones_idx = torch.remainder(torch.arange(10, 100, device=device), 10)
        two_digit = digit_probs_pos1[:, tens_idx] * digit_probs_pos2[:, ones_idx]  # (B, 90)

        # Absent: EOS at first position
        absent = eos_prob_pos1  # (B, 1)

        # Stack: [single_digit(10), two_digit(90), absent(1)] = 101
        number_probs = torch.cat([single_digit, two_digit, absent], dim=1)  # (B, 101)

        # Convert back to logits (log-space) for Dirichlet head
        number_logits = torch.log(number_probs + 1e-8)
        return number_logits


class STNJerseyModel(nn.Module):
    """Spatial Transformer + Frozen PARseq for jersey number recognition.

    Only the localization head trains. ViT backbone and PARseq are frozen.
    """

    def __init__(
        self,
        vit_model_name: str = "timm/vit_small_patch16_224.augreg_in21k",
        parseq_input_size: tuple = (32, 128),
    ):
        super().__init__()

        # Frozen ViT backbone for patch features
        self.backbone = timm.create_model(vit_model_name, pretrained=True)
        for param in self.backbone.parameters():
            param.requires_grad = False
        embed_dim = self.backbone.embed_dim
        logger.info(f"Frozen ViT backbone: {vit_model_name} (embed_dim={embed_dim})")

        # Trainable localization head
        self.loc_head = LocalizationHead(embed_dim)

        # Frozen PARseq (full encoder + decoder, pretrained on scene text)
        self.parseq = torch.hub.load(
            "baudm/parseq", "parseq", pretrained=True, trust_repo=True
        )
        for param in self.parseq.parameters():
            param.requires_grad = False
        self.parseq.eval()
        self.parseq_input_size = parseq_input_size
        logger.info("Frozen PARseq loaded (pretrained scene text reader)")

        # Number composition: PARseq chars → 101-class logits
        self.composition = NumberCompositionHead()

        # Learnable temperature for Dirichlet sharpness
        self.log_temperature = nn.Parameter(torch.tensor(1.0))

    def _extract_patches(self, x: torch.Tensor) -> torch.Tensor:
        """Extract patch tokens from frozen ViT (no CLS)."""
        with torch.no_grad():
            features = self.backbone.forward_features(x)
        # features: (B, 197, D) — CLS + 196 patches
        return features[:, 1:]  # (B, 196, D)

    def _crop_region(
        self, images: torch.Tensor, crop_params: torch.Tensor
    ) -> torch.Tensor:
        """Differentiable crop using grid_sample.

        Args:
            images: (B, 3, H, W) original images
            crop_params: (B, 4) — (cy, cx, h, w) in [0, 1]

        Returns:
            cropped: (B, 3, 32, 128) — PARseq-sized crops
        """
        B = images.size(0)
        cy, cx, h, w = crop_params[:, 0], crop_params[:, 1], crop_params[:, 2], crop_params[:, 3]

        # Scale h and w: min 0.15, max 0.8 of image
        h = 0.15 + h * 0.65  # [0.15, 0.8]
        w = 0.2 + w * 0.6   # [0.2, 0.8]

        # Convert to grid_sample coordinates [-1, 1]
        # cy, cx are center in [0, 1], convert to [-1, 1]
        cy_grid = cy * 2 - 1
        cx_grid = cx * 2 - 1

        # Build affine grid for each sample
        out_h, out_w = self.parseq_input_size
        theta = torch.zeros(B, 2, 3, device=images.device)
        theta[:, 0, 0] = w      # x scale
        theta[:, 1, 1] = h      # y scale
        theta[:, 0, 2] = cx_grid  # x translation
        theta[:, 1, 2] = cy_grid  # y translation

        grid = F.affine_grid(theta, (B, 3, out_h, out_w), align_corners=False)
        cropped = F.grid_sample(images, grid, align_corners=False, mode="bilinear", padding_mode="border")

        return cropped

    def _parseq_normalize(self, images: torch.Tensor) -> torch.Tensor:
        """Normalize cropped images for PARseq (expects [0,1] → [-1,1])."""
        # Our images are in [-1, 1] (uncertainty_jnr normalization)
        # PARseq expects: (img / 255 - 0.5) / 0.5 = img / 127.5 - 1 = same as ours
        # So no extra normalization needed if input is already [-1, 1]
        return images

    def forward(
        self,
        x: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        size: Optional[torch.Tensor] = None,
    ) -> STNModelOutput:
        """
        Args:
            x: (B, 3, 224, 224) or (B, T, 3, 224, 224) images
        """
        # Handle tracklet input: use middle frame
        if x.ndim == 5:
            B, T = x.shape[:2]
            mid = T // 2
            x = x[:, mid]  # (B, 3, H, W) — pick middle frame

        # Step 1: Extract patch features from frozen ViT
        patch_tokens = self._extract_patches(x)  # (B, 196, 384)

        # Step 2: Predict crop location
        crop_params = self.loc_head(patch_tokens)  # (B, 4)

        # Step 3: Differentiable crop
        cropped = self._crop_region(x, crop_params)  # (B, 3, 32, 128)
        cropped_normalized = self._parseq_normalize(cropped)

        # Step 4: Frozen PARseq reads the cropped region
        with torch.no_grad():
            parseq_logits = self.parseq(cropped_normalized)  # (B, max_len, charset)

        # But we need gradients to flow to loc_head through the crop!
        # Re-run with gradient through the crop (PARseq weights frozen but graph connected)
        parseq_logits = self.parseq(cropped_normalized)  # (B, max_len, charset)

        # Step 5: Compose into 101-class number logits
        number_logits_raw = self.composition(parseq_logits)  # (B, 101)

        # Apply learnable temperature
        temp = self.log_temperature.exp()
        all_logits = number_logits_raw * temp

        # Dirichlet uncertainty
        number_logits = all_logits[:, :100]
        alpha = torch.exp(number_logits) + 1.0
        S = alpha.sum(dim=1, keepdim=True)
        probs = alpha / S
        uncertainty = (100.0 / S).squeeze(1)
        predicted_number = probs.argmax(dim=1)

        # Full probs including absent
        all_alpha = torch.exp(all_logits) + 1.0
        all_S = all_alpha.sum(dim=1, keepdim=True)
        all_probs = all_alpha / all_S

        return STNModelOutput(
            all_logits=all_logits,
            number_logits=number_logits,
            probs=all_probs,
            uncertainty=uncertainty,
            predicted_number=predicted_number,
            crop_params=crop_params,
        )
