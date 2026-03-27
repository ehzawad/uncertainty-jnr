"""Spatial Transformer + Frozen PARseq for cross-domain jersey number recognition.

Architecture (v2 — fixes absent detection, single-digit bias, static crop):
    Image (224x224) → Frozen ViT → patch features
                    → Spatial soft-argmax localization → crop box
                    → Differentiable crop (grid_sample) → tight number region (32x128)
                    → Frozen PARseq → digit string (log-space composition)
                    → Absent classifier (from patches) + Number logits → Dirichlet uncertainty
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import logging
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
    crop_params: torch.Tensor     # (B, 4) — cy, cx, h, w
    decoder_pos_logits: Optional[torch.Tensor] = None


class LocalizationHead(nn.Module):
    """Predicts crop box via spatial soft-argmax on ViT patches.

    Uses attention scores over the 14x14 patch grid to compute a
    differentiable center (cy, cx) via soft-argmax. Size (h, w) is
    predicted from the attended feature. This ensures the crop location
    is spatially meaningful and input-dependent.
    """

    def __init__(self, embed_dim: int = 384, hidden_dim: int = 128):
        super().__init__()
        # Attention scoring: which patches are text-like?
        self.attn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Dropout(0.3),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        # Learnable temperature for attention sharpness (clamped to prevent collapse)
        self.temperature = nn.Parameter(torch.tensor(1.0))

        # Size prediction from attended feature
        self.size_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 2),  # h, w only
        )
        # Initialize to moderate crop size
        nn.init.zeros_(self.size_head[-1].weight)
        self.size_head[-1].bias.data = torch.tensor([0.0, 1.0])

        # Spatial coordinates for 14x14 patch grid
        ys = torch.linspace(0, 1, 14).unsqueeze(1).expand(14, 14).reshape(-1)
        xs = torch.linspace(0, 1, 14).unsqueeze(0).expand(14, 14).reshape(-1)
        self.register_buffer("patch_ys", ys)
        self.register_buffer("patch_xs", xs)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch_tokens: (B, 196, 384)
        Returns:
            crop_params: (B, 4) — (cy, cx, h, w) where cy/cx from soft-argmax [0,1],
                         h/w from sigmoid [0,1]
        """
        # Attention scores
        attn_logits = self.attn(patch_tokens).squeeze(-1)  # (B, 196)
        temp = self.temperature.clamp(min=0.3, max=3.0)
        attn_weights = F.softmax(attn_logits / temp, dim=1)  # (B, 196)

        # Spatial soft-argmax for center
        cy = (attn_weights * self.patch_ys.unsqueeze(0)).sum(dim=1)  # (B,)
        cx = (attn_weights * self.patch_xs.unsqueeze(0)).sum(dim=1)  # (B,)

        # Attended feature for size prediction
        pooled = (patch_tokens * attn_weights.unsqueeze(-1)).sum(dim=1)  # (B, 384)
        size_raw = self.size_head(pooled)  # (B, 2)
        h = torch.sigmoid(size_raw[:, 0])
        w = torch.sigmoid(size_raw[:, 1])

        return torch.stack([cy, cx, h, w], dim=1)  # (B, 4)


class AbsentClassifierHead(nn.Module):
    """Binary classifier for absent/visible jersey number.

    Operates directly on ViT patch features. PARseq can't detect absent
    (it always tries to read something). This head learns from the
    has_prediction labels in training data.
    """

    def __init__(self, embed_dim: int = 384, hidden_dim: int = 128):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
        )
        # Bias toward "number present" (majority class)
        nn.init.zeros_(self.head[-1].weight)
        self.head[-1].bias.data = torch.tensor([-2.0])

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch_tokens: (B, 196, 384)
        Returns:
            absent_logit: (B, 1)
        """
        # Simple mean pool — all patches contribute equally
        pooled = patch_tokens.mean(dim=1)  # (B, 384)
        return self.head(pooled)  # (B, 1)


class NumberCompositionHead(nn.Module):
    """Converts PARseq character logits to 100-class jersey number logits.

    Works in LOG-SPACE to avoid single-digit bias from probability multiplication.
    Absent detection is handled by a separate AbsentClassifierHead.
    """

    def __init__(self):
        super().__init__()
        # Learnable length bias: corrects for systematic single-vs-double digit bias
        self.length_bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, parseq_logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            parseq_logits: (B, max_label_len, charset_size) raw logits from PARseq
        Returns:
            number_logits: (B, 100) composed jersey number logits (no absent)
        """
        device = parseq_logits.device

        # Log-space to avoid probability multiplication bias
        log_probs = F.log_softmax(parseq_logits, dim=-1)  # (B, max_len, charset)

        # Log digit probs: charset indices 1-10 = digits '0'-'9'
        log_digit_pos1 = log_probs[:, 0, 1:11]  # (B, 10)
        log_digit_pos2 = log_probs[:, 1, 1:11]  # (B, 10)

        # Log EOS prob at position 2 (sequence length = 1 → single digit)
        log_eos_pos2 = log_probs[:, 1, 0:1]  # (B, 1)

        # Single digit (0-9): log P(digit@pos1) + log P(EOS@pos2)
        single_digit_logits = log_digit_pos1 + log_eos_pos2  # (B, 10)

        # Two digit (10-99): log P(tens@pos1) + log P(ones@pos2) + length_bias
        tens_idx = torch.div(torch.arange(10, 100, device=device), 10, rounding_mode="floor")
        ones_idx = torch.remainder(torch.arange(10, 100, device=device), 10)
        two_digit_logits = log_digit_pos1[:, tens_idx] + log_digit_pos2[:, ones_idx] + self.length_bias

        # Stack: [single(10), two_digit(90)] = 100
        number_logits = torch.cat([single_digit_logits, two_digit_logits], dim=1)  # (B, 100)
        return number_logits


class STNJerseyModel(nn.Module):
    """Spatial Transformer + Frozen PARseq for jersey number recognition.

    Trainable: LocalizationHead, AbsentClassifierHead, NumberCompositionHead.length_bias,
               log_temperature. Everything else frozen.
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

        # Trainable heads
        self.loc_head = LocalizationHead(embed_dim)
        self.absent_head = AbsentClassifierHead(embed_dim)

        # Frozen PARseq
        self.parseq = torch.hub.load(
            "baudm/parseq", "parseq", pretrained=True, trust_repo=True
        )
        for param in self.parseq.parameters():
            param.requires_grad = False
        self.parseq.eval()
        self.parseq_input_size = parseq_input_size
        logger.info("Frozen PARseq loaded (pretrained scene text reader)")

        # Number composition (log-space, no absent)
        self.composition = NumberCompositionHead()

        # Learnable temperature for Dirichlet sharpness
        self.log_temperature = nn.Parameter(torch.tensor(1.0))

    def _extract_patches(self, x: torch.Tensor) -> torch.Tensor:
        """Extract patch tokens from frozen ViT (no CLS)."""
        with torch.no_grad():
            features = self.backbone.forward_features(x)
        return features[:, 1:]  # (B, 196, D)

    def _crop_region(self, images: torch.Tensor, crop_params: torch.Tensor) -> torch.Tensor:
        """Differentiable crop using grid_sample."""
        B = images.size(0)
        cy, cx, h, w = crop_params[:, 0], crop_params[:, 1], crop_params[:, 2], crop_params[:, 3]

        # Scale h and w: min 0.15, max 0.8
        h = 0.15 + h * 0.65
        w = 0.2 + w * 0.6

        # Convert to grid_sample coordinates [-1, 1]
        cy_grid = cy * 2 - 1
        cx_grid = cx * 2 - 1

        out_h, out_w = self.parseq_input_size
        theta = torch.zeros(B, 2, 3, device=images.device)
        theta[:, 0, 0] = w
        theta[:, 1, 1] = h
        theta[:, 0, 2] = cx_grid
        theta[:, 1, 2] = cy_grid

        grid = F.affine_grid(theta, (B, 3, out_h, out_w), align_corners=False)
        return F.grid_sample(images, grid, align_corners=False, mode="bilinear", padding_mode="border")

    def forward(
        self,
        x: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        size: Optional[torch.Tensor] = None,
    ) -> STNModelOutput:
        # Handle tracklet: use middle frame
        if x.ndim == 5:
            x = x[:, x.shape[1] // 2]

        # Step 1: Frozen ViT patch features
        patch_tokens = self._extract_patches(x)  # (B, 196, 384)

        # Step 2: Predict crop location (spatial soft-argmax)
        crop_params = self.loc_head(patch_tokens)  # (B, 4)

        # Step 3: Differentiable crop
        cropped = self._crop_region(x, crop_params)  # (B, 3, 32, 128)

        # Step 4: Frozen PARseq reads the crop (gradients flow through grid_sample)
        parseq_logits = self.parseq(cropped)  # (B, max_len, charset)

        # Step 5: Log-space number composition (100 classes, no absent)
        number_logits_raw = self.composition(parseq_logits)  # (B, 100)

        # Extract per-position digit logits for digit-level CE (NED proxy)
        # PARseq charset: index 0=EOS, 1-10=digits '0'-'9'
        # Remap to 11-class: 0-9=digits, 10=EOS
        pos_logits_raw = parseq_logits[:, :2, :]  # (B, 2, charset)
        digit_pos_logits = torch.cat([
            pos_logits_raw[:, :, 1:11],   # digits 0-9
            pos_logits_raw[:, :, 0:1],    # EOS → index 10
        ], dim=-1)  # (B, 2, 11)

        # Step 6: Absent classifier from patch features
        absent_logit = self.absent_head(patch_tokens)  # (B, 1)

        # Combine: [100 number logits, 1 absent logit]
        temp = self.log_temperature.exp().clamp(max=5.0)
        all_logits = torch.cat([number_logits_raw * temp, absent_logit], dim=1)  # (B, 101)

        # Dirichlet uncertainty — softplus instead of exp to prevent evidence explosion
        number_logits = all_logits[:, :100]
        alpha = F.softplus(number_logits) + 1.0
        S = alpha.sum(dim=1, keepdim=True)
        probs = alpha / S
        uncertainty = (100.0 / S).squeeze(1)
        predicted_number = probs.argmax(dim=1)

        all_alpha = F.softplus(all_logits) + 1.0
        all_S = all_alpha.sum(dim=1, keepdim=True)
        all_probs = all_alpha / all_S

        return STNModelOutput(
            all_logits=all_logits,
            number_logits=number_logits,
            probs=all_probs,
            uncertainty=uncertainty,
            predicted_number=predicted_number,
            crop_params=crop_params,
            decoder_pos_logits=digit_pos_logits,
        )
