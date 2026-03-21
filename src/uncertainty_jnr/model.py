import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import logging
from typing import Dict, Any, Optional, Literal
from pathlib import Path
from timm.models.vision_transformer import VisionTransformer
from timm.models.eva import Eva
from abc import ABC, abstractmethod
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ModelOutput:
    """Structured output from the OCR model.

    This class provides a consistent interface for different uncertainty modeling approaches.
    """

    all_logits: (
        torch.Tensor
    )  # Raw logits from the model (B, 101) - includes absent class
    number_logits: torch.Tensor  # Logits for jersey numbers only (B, 100)
    number_probs: torch.Tensor  # Probabilities for jersey numbers (B, 100)
    uncertainty: torch.Tensor  # Uncertainty score (B,) - higher means more uncertain


class JerseyClassifier(ABC, torch.nn.Module):
    """Abstract base class for jersey number classifiers."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits for all 100 numbers plus absent class (101 total)."""
        pass


class IndependentClassifier(JerseyClassifier):
    """Classical approach treating each number as independent class, with absent class."""

    def __init__(self, embed_dim: int):
        super().__init__(embed_dim)
        # Add one more output for the "absent" class (101 total)
        self.classifier = torch.nn.Linear(embed_dim, 101)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class DigitAwareClassifier(JerseyClassifier):
    """Digit-aware classifier that reconstructs numbers from digit predictions, with absent class."""

    def __init__(self, embed_dim: int):
        super().__init__(embed_dim)

        # Three separate digit classifiers
        self.single_digit = torch.nn.Linear(embed_dim, 10)  # 0-9
        self.tens_digit = torch.nn.Linear(embed_dim, 10)  # 0-9 for tens place
        self.ones_digit = torch.nn.Linear(embed_dim, 10)  # 0-9 for ones place

        # Separate linear layer for the "absent" class
        self.absent_classifier = torch.nn.Linear(embed_dim, 1)

        # Register buffer for indices to avoid recreation
        tens_idx = torch.div(torch.arange(10, 100), 10, rounding_mode="floor")
        ones_idx = torch.remainder(torch.arange(10, 100), 10)
        self.register_buffer("tens_idx", tens_idx)
        self.register_buffer("ones_idx", ones_idx)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get digit logits
        single_logits = self.single_digit(x)  # (B, 10)
        tens_logits = self.tens_digit(x)  # (B, 10)
        ones_logits = self.ones_digit(x)  # (B, 10)

        # Get absent logit
        absent_logit = self.absent_classifier(x)  # (B, 1)

        batch_size = x.shape[0]

        # Initialize output logits for numbers (0-99)
        number_logits = torch.empty(batch_size, 100, device=x.device)

        # Fill single digits (0-9)
        number_logits[:, :10] = single_logits

        # Fill two-digit numbers (10-99) using vectorized operations
        # Shape: (B, 90)
        two_digit_logits = tens_logits[:, self.tens_idx] + ones_logits[:, self.ones_idx]
        number_logits[:, 10:] = two_digit_logits

        # Concatenate with absent logit to get all logits (0-99 + absent)
        all_logits = torch.cat([number_logits, absent_logit], dim=1)  # (B, 101)

        return all_logits


class TiedDigitAwareClassifier(JerseyClassifier):
    """Digit-aware classifier with shared weights and position embeddings, with absent class."""

    def __init__(
        self,
        embed_dim: int,
        embedding_type: str = "additive",
        per_digit_bias: bool = True,
    ):
        """Initialize the classifier.

        Args:
            embed_dim: Dimension of input embeddings
            embedding_type: Type of position embedding ('additive' or 'multiplicative')
            per_digit_bias: Whether to use per-digit bias
        """
        super().__init__(embed_dim)

        if embedding_type not in ["additive", "multiplicative"]:
            raise ValueError(f"Unknown embedding type: {embedding_type}")

        self.embedding_type = embedding_type
        self.per_digit_bias = per_digit_bias
        # Single digit classifier shared across positions (without bias)
        self.digit_classifier = torch.nn.Linear(embed_dim, 10, bias=False)

        # Separate linear layer for the "absent" class
        self.absent_classifier = torch.nn.Linear(embed_dim, 1)

        # Learnable position-specific biases
        if per_digit_bias:
            self.position_biases = torch.nn.Parameter(torch.zeros(3, 10))
        else:
            self.position_biases = torch.nn.Parameter(torch.zeros(3, 1))

        # Learnable position embeddings - initialize close to 1 for multiplicative
        init_value = (
            torch.randn(3, embed_dim)
            if embedding_type == "additive"
            else torch.ones(3, embed_dim)
        )
        self.position_embeddings = torch.nn.Parameter(init_value)

        # Register buffer for indices
        tens_idx = torch.div(torch.arange(10, 100), 10, rounding_mode="floor")
        ones_idx = torch.remainder(torch.arange(10, 100), 10)
        self.register_buffer("tens_idx", tens_idx)
        self.register_buffer("ones_idx", ones_idx)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        # Apply position embeddings based on type
        if self.embedding_type == "additive":
            positioned_features = x.unsqueeze(1) + self.position_embeddings.unsqueeze(0)
        else:  # multiplicative
            positioned_features = x.unsqueeze(1) * self.position_embeddings.unsqueeze(0)

        digit_logits = self.digit_classifier(positioned_features)
        digit_logits = digit_logits + self.position_biases.unsqueeze(0)

        single_logits = digit_logits[:, 0]
        tens_logits = digit_logits[:, 1]
        ones_logits = digit_logits[:, 2]

        # Get absent logit
        absent_logit = self.absent_classifier(x)  # (B, 1)

        # Initialize output logits for numbers (0-99)
        number_logits = torch.empty(batch_size, 100, device=x.device)
        number_logits[:, :10] = single_logits
        two_digit_logits = tens_logits[:, self.tens_idx] + ones_logits[:, self.ones_idx]
        number_logits[:, 10:] = two_digit_logits

        # Concatenate with absent logit to get all logits (0-99 + absent)
        all_logits = torch.cat([number_logits, absent_logit], dim=1)  # (B, 101)

        return all_logits


class DecoderLayer(nn.Module):
    """Pre-LN Transformer decoder layer with two-stream attention.

    Matches PARseq's DecoderLayer architecture exactly for weight compatibility.
    Supports query stream (position queries) and content stream (self-attending context).
    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 1536,
                 dropout: float = 0.1, activation: str = "gelu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_c = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = F.gelu if activation == "gelu" else F.relu

    def _forward_stream(self, tgt, tgt_norm, tgt_kv, memory, tgt_mask=None):
        """Forward for a single stream (query or content)."""
        tgt2, _ = self.self_attn(tgt_norm, tgt_kv, tgt_kv, attn_mask=tgt_mask)
        tgt = tgt + self.dropout1(tgt2)

        tgt2, ca_weights = self.cross_attn(self.norm1(tgt), memory, memory)
        tgt = tgt + self.dropout2(tgt2)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(self.norm2(tgt)))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt, ca_weights

    def forward(self, query, content, memory, query_mask=None, content_mask=None):
        query_norm = self.norm_q(query)
        content_norm = self.norm_c(content)
        query, ca_weights = self._forward_stream(query, query_norm, content_norm, memory, query_mask)
        content, _ = self._forward_stream(content, content_norm, content_norm, memory, content_mask)
        return query, content, ca_weights


class TextRegionHead(nn.Module):
    """Lightweight patch-level text detector.

    Predicts per-patch probability of containing text/digits.
    Used to modulate patch tokens before the decoder's cross-attention,
    separating WHERE (this head) from WHAT (frozen decoder).
    """

    def __init__(self, embed_dim: int = 384, hidden_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, patch_tokens: torch.Tensor):
        """
        Args:
            patch_tokens: (B, num_patches, embed_dim)
        Returns:
            masked_tokens: (B, num_patches, embed_dim) — text-weighted patches
            scores: (B, num_patches, 1) — per-patch text probability
        """
        scores = torch.sigmoid(self.mlp(patch_tokens))  # (B, 196, 1)
        return patch_tokens * scores, scores


class DigitDecoder(nn.Module):
    """PARseq-style spatial decoder for jersey digit reading.

    Cross-attends to ViT patch tokens using 2 learned position queries
    (tens-digit, ones-digit). Each query produces 11-class logits (0-9 + absent).
    The per-query logits are composed into 101-class number logits using the same
    formula as TiedDigitAwareClassifier.
    """

    def __init__(self, embed_dim: int = 384, num_heads: int = 12,
                 dim_feedforward: int = 1536, num_queries: int = 2):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_queries = num_queries

        # Position queries: [pos1 (tens/single), pos2 (ones)]
        self.pos_queries = nn.Parameter(torch.randn(1, num_queries, embed_dim) * 0.02)

        # Single decoder layer (matching PARseq)
        decoder_layer = DecoderLayer(embed_dim, num_heads, dim_feedforward)
        self.layers = nn.ModuleList([decoder_layer])
        self.norm = nn.LayerNorm(embed_dim)

        # 11-class head per query position: digits 0-9 + absent marker
        self.digit_head = nn.Linear(embed_dim, 11)

        # Index buffers for composing 2-digit numbers
        tens_idx = torch.div(torch.arange(10, 100), 10, rounding_mode="floor")
        ones_idx = torch.remainder(torch.arange(10, 100), 10)
        self.register_buffer("tens_idx", tens_idx)
        self.register_buffer("ones_idx", ones_idx)

    def forward(self, patch_tokens: torch.Tensor):
        """Forward pass.

        Args:
            patch_tokens: (B, num_patches, embed_dim) from ViT backbone

        Returns:
            all_logits: (B, 101) composed number logits
            pos_logits: (B, num_queries, 11) raw per-position digit logits
        """
        B = patch_tokens.size(0)
        query = self.pos_queries.expand(B, -1, -1)
        content = query.clone()

        for layer in self.layers:
            query, content, ca_weights = layer(query, content, patch_tokens)

        query = self.norm(query)  # (B, 2, embed_dim)

        # Per-position digit logits
        pos_logits = self.digit_head(query)  # (B, 2, 11)

        pos1_logits = pos_logits[:, 0, :10]  # (B, 10) — single-digit / tens
        pos2_logits = pos_logits[:, 1, :10]  # (B, 10) — ones

        # Compose 101-class logits (same formula as TiedDigitAwareClassifier)
        number_logits = torch.empty(B, 100, device=patch_tokens.device)
        number_logits[:, :10] = pos1_logits  # single-digit: 0-9
        number_logits[:, 10:] = pos1_logits[:, self.tens_idx] + pos2_logits[:, self.ones_idx]

        # Absent logit: mean of both position absent markers
        absent_logit = (pos_logits[:, 0, 10:11] + pos_logits[:, 1, 10:11]) / 2.0

        all_logits = torch.cat([number_logits, absent_logit], dim=1)  # (B, 101)
        return all_logits, pos_logits


class AttentionPooling(nn.Module):
    """Temporal attention pooling over frame-level features.

    A learnable query attends to T frame features and produces a single
    pooled representation. Returns both the pooled feature and attention weights
    (used for temporal patch pooling).
    """

    def __init__(self, embed_dim: int, num_heads: int = 4):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, frame_features: torch.Tensor):
        """
        Args:
            frame_features: (B, T, embed_dim)

        Returns:
            pooled: (B, embed_dim)
            weights: (B, 1, T) attention weights
        """
        B = frame_features.size(0)
        query = self.query.expand(B, -1, -1)
        pooled, weights = self.attn(query, frame_features, frame_features)
        return pooled.squeeze(1), weights


def load_parseq_weights(decoder: DigitDecoder, weights_path: Path):
    """Load PARseq-small pretrained weights into DigitDecoder.

    Maps PARseq decoder keys to our decoder, taking first 2 of 26 pos_queries
    and first 11 of 95 head classes.
    """
    raw = torch.load(weights_path, map_location="cpu", weights_only=True)
    # PARseq checkpoints nest weights under 'state_dict'
    state = raw.get("state_dict", raw)

    loaded = 0
    our_state = decoder.state_dict()

    for key in list(our_state.keys()):
        if key in ("tens_idx", "ones_idx"):
            continue  # skip registered buffers

        if key == "pos_queries":
            src_key = "pos_queries"
            if src_key in state:
                our_state[key] = state[src_key][:, :2, :]
                loaded += 1
        elif key == "digit_head.weight":
            if "head.weight" in state:
                our_state[key] = state["head.weight"][:11, :]
                loaded += 1
        elif key == "digit_head.bias":
            if "head.bias" in state:
                our_state[key] = state["head.bias"][:11]
                loaded += 1
        else:
            # Decoder keys: our "layers.0.self_attn..." maps to PARseq "decoder.layers.0.self_attn..."
            parseq_key = f"decoder.{key}" if not key.startswith("decoder.") else key
            if parseq_key in state and our_state[key].shape == state[parseq_key].shape:
                our_state[key] = state[parseq_key]
                loaded += 1

    decoder.load_state_dict(our_state)
    logger.info(f"Loaded {loaded}/{len(our_state) - 2} PARseq weights from {weights_path}")


class TimmOCRModel(torch.nn.Module):
    """OCR model based on pretrained timm Vision Transformer."""

    def __init__(
        self,
        model_name: str,
        classifier_type: str = "independent",
        embedding_type: str = "additive",
        per_digit_bias: bool = True,
        uncertainty_head: Literal["dirichlet", "softmax"] = "dirichlet",
        pretrained: bool = True,
        size_embedding: bool = False,
        use_decoder: bool = False,
        parseq_weights_path: Optional[Path] = None,
        freeze_decoder: bool = False,
        **kwargs: Dict[str, Any],
    ):
        """Initialize the OCR model.

        Args:
            model_name: Name of the timm model to use (must be a Vision Transformer)
            classifier_type: Type of classifier to use ('independent', 'digit_aware', or 'tied_digit_aware')
            embedding_type: Type of position embedding for tied_digit_aware ('additive' or 'multiplicative')
            per_digit_bias: Whether to use per-digit bias for tied_digit_aware
            uncertainty_head: Type of uncertainty modeling ('dirichlet' or 'softmax')
            pretrained: Whether to use pretrained weights
            use_decoder: Whether to use spatial DigitDecoder (dual-path)
            parseq_weights_path: Path to PARseq-small pretrained weights
            **kwargs: Additional arguments to pass to timm.create_model
        """
        super().__init__()

        # Create the backbone model
        self.backbone = timm.create_model(model_name, pretrained=pretrained, **kwargs)

        # Verify that it's a Vision Transformer or Eva
        if not isinstance(self.backbone, (VisionTransformer, Eva)):
            raise ValueError(
                f"Model {model_name} is not a Vision Transformer. "
                f"Got {type(self.backbone)} instead."
            )

        # Get embedding dimension from the model
        embed_dim = self.backbone.embed_dim

        # Time embedding layer with small initialization
        self.time_embed = torch.nn.Sequential(
            torch.nn.Linear(1, embed_dim),
        )
        self.size_embed = (
            torch.nn.Sequential(
                torch.nn.Linear(1, embed_dim),
            )
            if size_embedding
            else None
        )
        # Initialize embedding with small values
        with torch.no_grad():
            self.time_embed[0].weight.data.uniform_(-0.001, 0.001)
            self.time_embed[0].bias.data.zero_()

            if self.size_embed is not None:
                self.size_embed[0].weight.data.uniform_(-0.003, 0.003)
                self.size_embed[0].bias.data.zero_()

        # Create classifier based on type
        if classifier_type == "independent":
            self.classifier = IndependentClassifier(embed_dim)
        elif classifier_type == "digit_aware":
            self.classifier = DigitAwareClassifier(embed_dim)
        elif classifier_type == "tied_digit_aware":
            self.classifier = TiedDigitAwareClassifier(
                embed_dim,
                embedding_type=embedding_type,
                per_digit_bias=per_digit_bias,
            )
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")

        # Store uncertainty head type
        if uncertainty_head not in ["dirichlet", "softmax"]:
            raise ValueError(f"Unknown uncertainty head: {uncertainty_head}")
        self.uncertainty_head = uncertainty_head

        # Spatial decoder (dual-path)
        self.use_decoder = use_decoder
        if use_decoder:
            self.decoder = DigitDecoder(embed_dim)
            # Text region head: learns WHERE digits are on patches
            self.text_region_head = TextRegionHead(embed_dim)
            # Gate parameter: sigmoid(0) = 0.5 initial weight
            self.gate_param = nn.Parameter(torch.tensor(0.0))
            # Attention pooling for temporal (5D) input
            self.attn_pool = AttentionPooling(embed_dim)
            # Load PARseq pretrained weights
            if parseq_weights_path is not None:
                load_parseq_weights(self.decoder, Path(parseq_weights_path))
            # Freeze decoder to preserve text-reading ability (only train pos_queries + gate)
            if freeze_decoder:
                for name, param in self.decoder.named_parameters():
                    if "pos_queries" not in name:
                        param.requires_grad = False
                n_frozen = sum(1 for n, p in self.decoder.named_parameters() if not p.requires_grad)
                n_total = sum(1 for _ in self.decoder.parameters())
                logger.info(f"Frozen {n_frozen}/{n_total} decoder params (keeping pos_queries trainable)")

    def forward(
        self,
        x: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        size: Optional[torch.Tensor] = None,
    ) -> ModelOutput:
        """Forward pass of the model.

        Supports both 4D (B,C,H,W) single-frame and 5D (B,T,C,H,W) tracklet input.

        Parameters
        ----------
        x : torch.Tensor
            Input images: 4D (B,3,H,W) or 5D (B,T,3,H,W)
        t : Optional[torch.Tensor]
            Optional time conditioning of shape (B, 1)
        size : Optional[torch.Tensor]
            Optional size conditioning of shape (B, 2)

        Returns
        -------
        ModelOutput
            Structured output containing logits, probabilities, and uncertainty.
            When decoder is active, also includes decoder_pos_logits.
        """
        is_tracklet = x.ndim == 5

        if is_tracklet and self.use_decoder:
            cls_features, patch_tokens = self._forward_tracklet(x, t, size)
        elif is_tracklet:
            # Temporal pooling on CLS only (no decoder)
            B, T = x.shape[:2]
            flat = x.reshape(B * T, *x.shape[2:])
            all_tokens = self._forward_backbone(flat, t, size)
            cls_flat = all_tokens[:, 0]  # (B*T, D)
            cls_seq = cls_flat.reshape(B, T, -1)
            cls_features, _ = self.attn_pool(cls_seq)
            patch_tokens = None
        else:
            # Single-frame: standard path
            all_tokens = self._forward_backbone(x, t, size)
            cls_features = all_tokens[:, 0]  # (B, D)
            patch_tokens = all_tokens[:, 1:] if self.use_decoder else None

        # Text region masking: both paths see ONLY text-focused features
        decoder_pos_logits = None
        if self.use_decoder and patch_tokens is not None:
            masked_patches, _text_scores = self.text_region_head(patch_tokens)

            # Path 1: Classifier on text-focused features (not sport-specific CLS)
            # Attention-pool the text-masked patches into a single vector
            text_features = masked_patches.mean(dim=1)  # (B, D) — avg pool text patches
            classifier_logits = self.classifier(text_features)

            # Path 2: Frozen decoder reads digits from text-masked patches
            decoder_logits, decoder_pos_logits = self.decoder(masked_patches)

            # Geometric mixture: final = (1-g)*cls + g*dec
            gate = torch.sigmoid(self.gate_param)
            all_logits = (1 - gate) * classifier_logits + gate * decoder_logits
        else:
            # Fallback: no decoder, use CLS token
            classifier_logits = self.classifier(cls_features)
            all_logits = classifier_logits

        # Extract number logits (excluding absent class)
        number_logits = all_logits[:, :100]  # (B, 100)

        if self.uncertainty_head == "dirichlet":
            alpha = torch.exp(number_logits) + 1.0
            S = alpha.sum(dim=1, keepdim=True)
            number_probs = alpha / S
            uncertainty = 100.0 / S.squeeze()
        else:
            all_probs = F.softmax(all_logits, dim=1)
            number_probs = all_probs[:, :100]
            absent_prob = all_probs[:, 100]
            uncertainty = absent_prob

        output = ModelOutput(
            all_logits=all_logits,
            number_logits=number_logits,
            number_probs=number_probs,
            uncertainty=uncertainty,
        )
        # Attach decoder pos_logits for auxiliary loss (not in dataclass to keep backward compat)
        output.decoder_pos_logits = decoder_pos_logits
        return output

    def _forward_backbone(
        self,
        x: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        size: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run ViT backbone, returning ALL tokens (CLS + patches).

        Returns:
            (B, 1+num_patches, embed_dim) — token 0 is CLS, rest are patches
        """
        x = self.backbone.patch_embed(x)
        x = self.backbone._pos_embed(x)

        if t is not None:
            t_emb = self.time_embed(t)
            x = x + t_emb.unsqueeze(1)

        if size is not None and self.size_embed is not None:
            size_emb = self.size_embed(size)
            x[:, 1:] = x[:, 1:] + size_emb.unsqueeze(1)

        x = self.backbone.patch_drop(x)
        x = self.backbone.norm_pre(x)
        x = self.backbone.blocks(x)
        x = self.backbone.norm(x)
        return x  # (B, 197, embed_dim) for ViT-Small 224

    def forward_features(
        self,
        x: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        size: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get the CLS embedding vector (backward compatible)."""
        return self._forward_backbone(x, t, size)[:, 0]

    def _forward_tracklet(
        self,
        x: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        size: Optional[torch.Tensor] = None,
    ):
        """Handle 5D tracklet input: (B, T, 3, H, W).

        Returns:
            pooled_cls: (B, embed_dim)
            pooled_patches: (B, num_patches, embed_dim)
        """
        B, T = x.shape[:2]
        flat = x.reshape(B * T, *x.shape[2:])

        all_tokens = self._forward_backbone(flat, t, size)  # (B*T, 197, D)
        D = all_tokens.size(-1)
        num_patches = all_tokens.size(1) - 1

        cls_flat = all_tokens[:, 0]  # (B*T, D)
        patches_flat = all_tokens[:, 1:]  # (B*T, P, D)

        cls_seq = cls_flat.reshape(B, T, D)
        patches_seq = patches_flat.reshape(B, T, num_patches, D)

        # Attention pooling on CLS tokens → (B, D) + weights (B, 1, T)
        pooled_cls, attn_weights = self.attn_pool(cls_seq)

        # Apply same temporal weights to patches: (B, 1, T) × (B, T, P, D) → (B, P, D)
        pooled_patches = torch.einsum("bot, btpd -> bpd", attn_weights, patches_seq)

        return pooled_cls, pooled_patches

    def compile(self):
        """Compile the model."""
        self.backbone = torch.compile(self.backbone)
