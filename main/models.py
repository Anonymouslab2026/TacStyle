from huggingface_hub.inference._generated.types import text_classification
import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from transformers import AutoTokenizer, AutoModel


# ---------------------------------------------------------------------------
# Frozen Qwen2 text encoder
# ---------------------------------------------------------------------------
class FrozenQwen2TextEncoder(nn.Module):
    """
    Frozen Qwen2 language encoder.

    Purpose:
        Convert natural language instructions into fixed semantic embeddings.

    IMPORTANT:
        Qwen2 is fully frozen. No gradients pass through Qwen2.

    Output:
        language embedding tensor [B, hidden_size]
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-0.5B-Instruct",
        max_length: int = 64,
    ):
        super().__init__()

        self.model_name = model_name
        self.max_length = max_length

        # ------------------------------------------------------------
        # Load pretrained tokenizer
        # ------------------------------------------------------------
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        # ------------------------------------------------------------
        # Load pretrained Qwen model
        # ------------------------------------------------------------
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        # ------------------------------------------------------------
        # Freeze all Qwen parameters
        # ------------------------------------------------------------
        for p in self.model.parameters():
            p.requires_grad = False

        self.model.eval()

        self.hidden_size = self.model.config.hidden_size

    @torch.no_grad()
    def forward(self, texts: List[str]) -> torch.Tensor:
        """
        Convert language strings into frozen embeddings.

        Args:
            texts:
                list[str] length B

        Returns:
            pooled embedding:
                [B, hidden_size]
        """

        if isinstance(texts, str):
            texts = [texts]

        device = next(self.model.parameters()).device

        # ------------------------------------------------------------
        # Tokenize text
        # ------------------------------------------------------------
        tokens = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        tokens = {
            k: v.to(device)
            for k, v in tokens.items()
        }

        # ------------------------------------------------------------
        # Forward pass through frozen Qwen
        # ------------------------------------------------------------
        outputs = self.model(**tokens)

        # Hidden states:
        # [B, L, hidden_size]
        hidden = outputs.last_hidden_state

        # ------------------------------------------------------------
        # Mean-pool valid tokens
        # ------------------------------------------------------------
        mask = tokens["attention_mask"].unsqueeze(-1).float()

        pooled = (
            hidden * mask
        ).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

        return pooled


# ---------------------------------------------------------------------------
# Sinusoidal timestep embedding
# ---------------------------------------------------------------------------
class SinusoidalTimeEmbedding(nn.Module):
    """
    Converts timestep indices into sinusoidal embeddings.

    Input: Integer timesteps 0,...,T-1
    Return: [B, T, D] sinusoidal embeddings
    
    Replace with nn.Embedding(T, D) if needed
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device

        # Half of dimensions use sin, half use cos
        half_dim = self.d_model // 2

        # Frequency scale used in standard transformer positional embeddings
        freqs = torch.exp(
            torch.arange(half_dim, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / (half_dim - 1))
        )

        # Multiply timestep by frequencies
        args = t.float().unsqueeze(-1) * freqs

        # Concatenate sin and cos embeddings
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        # If d_model is odd, pad one dimension
        if self.d_model % 2 == 1:
            emb = F.pad(emb, (0, 1))

        return emb


# ---------------------------------------------------------------------------
# Generic MLP encoder
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    """
    Small feedforward network used to encode state and tactile inputs
    into the shared transformer dimension D.
    """

    def __init__(self, in_dim: int, out_dim: int, hidden_mult: int = 4, dropout: float = 0.1):
        super().__init__()

        hidden_dim = hidden_mult * out_dim

        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Image encoder
# ---------------------------------------------------------------------------
class ImageEncoder(nn.Module):
    """
    Encodes image observations into transformer tokens.

    Input:
      images: [B, T, 3, H, W]

    Output:
      image_token: [B, T, D]

    To freeze resnet during training, use:
    model.vision.net_features.eval()
        for p in model.vision.net_features.parameters():
            p.requires_grad = False
    """

    def __init__(self, d_model: int, backbone: str = "resnet18", pretrained: bool = True):
        super().__init__()
        self.pretrained = pretrained

        # Load ResNet backbone
        if backbone == "resnet18":
            net = torchvision.models.resnet18(
                weights=None if not pretrained else torchvision.models.ResNet18_Weights.DEFAULT
            )
        elif backbone == "resnet34":
            net = torchvision.models.resnet34(
                weights=None if not pretrained else torchvision.models.ResNet34_Weights.DEFAULT
            )
        else:
            raise ValueError("Choose either resnet18 or resnet34 as backbone.")

        # Remove final classifier layer
        self.net_features = nn.Sequential(*list(net.children())[:-1])

        # ResNet feature dimension is 512
        self.proj = nn.Linear(512, d_model)

        # Normalize projected feature
        self.norm = nn.LayerNorm(d_model)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = images.shape

        # Merge batch and time for CNN processing
        x = images.reshape(B * T, C, H, W)

        # If pretrained, freeze CNN feature extraction
        if self.pretrained:
            with torch.no_grad():
                f = self.net_features(x).reshape(B * T, -1)
        else:
            f = self.net_features(x).reshape(B * T, -1)

        # Project CNN features to transformer dimension
        image_token = self.norm(self.proj(f)).reshape(B, T, -1)

        return image_token


# ---------------------------------------------------------------------------
# Style encoder
# ---------------------------------------------------------------------------
class StyleEncoderGRU(nn.Module):
    """
    Infers scalar latent style z from a trajectory.

    Input:
      traj_emb: [B, Td, D]

    Output:
      z: [B, 1]

    This is used during training when z_style=None.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 128, dropout: float = 0.0):
        super().__init__()

        # GRU summarizes the full trajectory
        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            dropout=dropout,
        )

        # Convert GRU hidden state into scalar z
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, traj_emb: torch.Tensor) -> torch.Tensor:
        # traj_emb: [B, Td, Din] -> hT: [1, B, H]
        _, hT = self.gru(traj_emb)

        # Last hidden state
        h = hT[-1]

        # Scalar style value
        z = self.head(h)

        return z


# ---------------------------------------------------------------------------
# Main policy configuration
# ---------------------------------------------------------------------------
@dataclass
class PolicyConfig:
    """
    Configuration for TacStylePolicy.

    For cloth:
      state_dim = 7
      action_dim = 7
      tactile_dim = 15
      use_vision = True

    For simple_2d:
      state_dim = 2
      action_dim = 2
      tactile_dim = 1
      use_vision = False
    """

    d_model: int = 256
    nhead: int = 4
    num_layers: int = 4
    ff_mult: int = 4
    dropout: float = 0.1
    use_adaln: bool = False
    vision_backbone: str = "resnet18"
    vision_pretrained: bool = True
    use_vision: bool = True
    tactile_dim: int = 15
    state_dim: int = 7
    action_dim: int = 7
    traj_stride: int = 10           # scale for downsampling trajectory

    # for baseline:
    conditioning_mode: str = "z"    # options "z" or "language"
    qwen_model_name: str = "Qwen/Qwen2-0.5B-Instruct"


# ---------------------------------------------------------------------------
# Main TacStyle policy
# ---------------------------------------------------------------------------
class TacStylePolicy(nn.Module):
    """
    Main TacStyle policy.

    Training mode:
      images + tactile + state + actions
      → infer z from full trajectory (ours)
      → embed language prompt (baseline)
      → predict actions

    Inference mode:
      images + tactile + state + externally provided z (ours) / language prompt (baseline)
      → predict actions

    Inputs:
      images:  [B, T, 3, H, W]
      tactile: [B, T, tactile_dim]
      state:   [B, T, state_dim]
      actions: [B, T, action_dim]   (used ONLY to infer style during training if style_s not provided)

    Either provide:
      style_s: [B, 1]  (user-provided at deployment)
    OR infer it (training) from the entire trajectory.

    Output:
      pred_actions: [B, T, action_dim]  (BC regression head)
    """

    def __init__(self, cfg: PolicyConfig):
        super().__init__()

        self.cfg = cfg
        D = cfg.d_model

        # ------------------------------------------------------------
        # Vision encoder
        # ------------------------------------------------------------
        if cfg.use_vision:
            self.vision_encoder = ImageEncoder(
                D,
                backbone=cfg.vision_backbone,
                pretrained=cfg.vision_pretrained,
            )

            # Freeze pretrained ResNet if using pretrained weights
            if cfg.vision_pretrained:
                for p in self.vision_encoder.net_features.parameters():
                    p.requires_grad = False
                self.vision_encoder.net_features.eval()
        else:
            # For simple_2d, no image encoder is used
            self.vision_encoder = None

        # ------------------------------------------------------------
        # State and tactile encoders
        # ------------------------------------------------------------
        self.tactile_encoder = MLP(cfg.tactile_dim, D, hidden_mult=2)
        self.state_encoder = MLP(cfg.state_dim, D, hidden_mult=2)

        # ------------------------------------------------------------
        # Action encoder
        # Used only for inferring z during training
        # ------------------------------------------------------------
        self.action_emb = nn.Sequential(
            nn.Linear(cfg.action_dim, D),
            nn.GELU(),
            nn.LayerNorm(D),
        )

        # Token type embeddings:
        # 0 = style token
        # 1 = state token
        # 2 = tactile token
        # 3 = image token
        self.type_emb = nn.Embedding(4, D)

        # Time embedding for temporal ordering
        self.time_emb = SinusoidalTimeEmbedding(D)

        # Fuse state, tactile, image, and action tokens before style encoder
        self.emb_fusion = nn.Sequential(
            nn.Linear(4 * D, D),
            nn.GELU(),
            nn.LayerNorm(D),
        )

        if cfg.conditioning_mode == "z":

            # GRU-based style encoder that outputs scalar z
            self.style_encoder = StyleEncoderGRU(in_dim=D, hidden_dim=128)

            style_dim = 1

        elif cfg.conditioning_mode == "language":
                       
            self.style_encoder = FrozenQwen2TextEncoder(
                model_name=cfg.qwen_model_name,
            )

            style_dim = self.style_encoder.hidden_size
        else:
            raise ValueError(f"Unknown conditioning mode: {cfg.conditioning_mode}")

        # Converts style representation into transformer style token
        self.style_tokenizer = nn.Sequential(     
            nn.Linear(style_dim, D),
            nn.GELU(),
            nn.Linear(D, D),
            nn.LayerNorm(D),
        )

        # Learnable base style token
        self.base_style_token = nn.Parameter(torch.zeros(1, 1, D))
        nn.init.normal_(self.base_style_token, mean=0.0, std=0.02)

        # Transformer policy conditioned on style
        self.policy = StyleConditionedTransformerEncoder(
            d_model=D,
            nhead=cfg.nhead,
            num_layers=cfg.num_layers,
            ff_mult=cfg.ff_mult,
            dropout=cfg.dropout,
            use_adaln=cfg.use_adaln,
        )

        # Predict action from transformer state-token outputs
        self.action_head = nn.Sequential(
            nn.LayerNorm(D),
            nn.Linear(D, D),
            nn.GELU(),
            nn.Linear(D, cfg.action_dim),
        )

        self.drop = nn.Dropout(cfg.dropout)

    def train(self, mode: bool = True):
        """
        Override train() so pretrained vision backbone remains frozen.
        """

        super().train(mode)

        if self.cfg.use_vision and self.cfg.vision_pretrained:
            self.vision_encoder.net_features.eval()

        # Keep frozen Qwen in eval mode.
        if self.cfg.conditioning_mode == "language":
            self.style_encoder.eval()

        return self

    def infer_style(self, state_token, tactile_token, image_token, action_token) -> torch.Tensor:
        """
        Infer latent style z from full trajectory during training when z_style=None.
        """

        stride = self.cfg.traj_stride

        # Downsample long trajectories to reduce computation
        xs = state_token[:, ::stride]
        xt = tactile_token[:, ::stride]
        xi = image_token[:, ::stride]
        xa = action_token[:, ::stride]

        # Combine all modalities and action information
        x = torch.cat([xs, xt, xi, xa], dim=-1)

        # Fuse into D-dimensional trajectory embedding
        x = self.emb_fusion(x)

        # GRU summarizes trajectory into scalar z
        z = self.style_encoder(x)

        return z
    
    def combine_tokens(self, state_token, tactile_token, image_token, style_emb: torch.Tensor) -> torch.Tensor:
        """
        Build transformer input sequence.

        Sequence:
          [STYLE,
           (STATE_0, TACTILE_0, IMAGE_0),
           (STATE_1, TACTILE_1, IMAGE_1),
           ...]
        
        Returns x: [B, 1 + 3T, D]
        """

        B, T, D = state_token.shape
        device = state_token.device

        # Ensure style_emb has shape [B, 1]
        if style_emb.ndim == 1:
            style_emb = style_emb.unsqueeze(-1)

        # Convert style embeddings into style token
        style_token = self.base_style_token.expand(B, 1, -1) + self.style_tokenizer(style_emb).unsqueeze(1)

        # Add type embedding for style token
        style_token = style_token + self.type_emb(torch.tensor([0], device=device))

        # Add type embeddings to modality tokens
        state_token = state_token + self.type_emb(torch.tensor([1], device=device))
        tactile_token = tactile_token + self.type_emb(torch.tensor([2], device=device))
        image_token = image_token + self.type_emb(torch.tensor([3], device=device))

        # Add timestep embeddings
        time_idx = torch.arange(T, device=device)
        time_emb = self.time_emb(time_idx).unsqueeze(0)

        state_token = state_token + time_emb
        tactile_token = tactile_token + time_emb
        image_token = image_token + time_emb

        # Interleave modality tokens per timestep
        input_token = torch.stack([state_token, tactile_token, image_token], dim=2)
        input_token = input_token.view(B, 3 * T, D)

        # Prepend style token
        x = torch.cat([style_token, input_token], dim=1)

        x = self.drop(x)

        return x

    def build_causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """
        Build causal attention mask.

        Each timestep can attend to:
          - style token
          - previous timestep tokens
          - current timestep tokens

        It cannot attend to future timesteps.
        """

        seq_len = 1 + 3 * T
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device)

        # Style token attends only to itself
        mask[0, 0] = 0.0

        for t in range(T):
            q_start = 1 + 3 * t
            q_end = q_start + 3

            # Allow each timestep token to attend to style token
            mask[q_start:q_end, 0] = 0.0

            # Allow attention to current and previous tokens
            allowed_end = 1 + 3 * (t + 1)
            mask[q_start:q_end, 1:allowed_end] = 0.0

        return mask

    def forward(
        self,
        image: torch.Tensor,
        tactile: torch.Tensor,
        state: torch.Tensor,
        z_style: Optional[torch.Tensor] = None,
        language_text: Optional[List[str]] = None,
        actions: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        If z_style is None:
          actions must be provided.
          The model infers z from the trajectory (ours).

        If z_style is provided:
          model directly conditions on that z (ours).

        Returns:
          pred_actions: [B, T, action_dim]
          style_s:      [B, 1] (inferred or provided)
        """

        # Encode images or use zero image tokens for simple_2d
        if self.vision_encoder is not None:
            image_token = self.vision_encoder(image)
        else:
            B, T = image.shape[0], image.shape[1]
            image_token = torch.zeros(B, T, self.cfg.d_model, device=image.device)
            # NOTE: this step replaced most "if self.cfg.use_vision" usages

        # Encode tactile and state
        tactile_token = self.tactile_encoder(tactile)
        state_token = self.state_encoder(state)

        if self.cfg.conditioning_mode == "z":
            # Training path: infer style from full trajectory
            if z_style is None:
                if actions is None:
                    raise ValueError("actions must be provided when z_style is not given.")

                action_token = self.action_emb(actions)
                z_style = self.infer_style(state_token, tactile_token, image_token, action_token)

            style_emb = z_style

        elif self.cfg.conditioning_mode == "language":
            if language_text is None:
                raise ValueError("language_text must be provided when conditioning_mode='language'")
            
            with torch.no_grad():
                style_emb = self.style_encoder(language_text).to(state.device)
                
                # Zero out embeddings for empty strings
                for i, text in enumerate(language_text):
                    if text == " ":
                        style_emb[i].zero_()  # same as style_emb[i] = 0.0



        # Build transformer token sequence
        x = self.combine_tokens(state_token, tactile_token, image_token, style_emb)

        # Build causal mask
        causal_mask = self.build_causal_mask(T=state_token.shape[1], device=state_token.device)

        # Run style-conditioned transformer
        out = self.policy(
            x,
            z_style,
            attn_mask=causal_mask,
        )

        # Extract outputs corresponding to state tokens
        T_out = state_token.shape[1]
        state_positions = 1 + 3 * torch.arange(T_out, device=out.device)
        out_state = out[:, state_positions, :]

        # Predict action at each timestep
        pred_actions = self.action_head(out_state)

        return pred_actions, z_style


# ---------------------------------------------------------------------------
# AdaLN layer
# ---------------------------------------------------------------------------
class AdaLayerNorm(nn.Module):
    """
    Adaptive LayerNorm.

    It uses style z to modify normalization parameters.
    Optional feature controlled by use_adaln.
    """

    def __init__(self, d_model: int, style_dim: int = 1, hidden_mult: int = 4):
        super().__init__()

        self.ln = nn.LayerNorm(d_model, elementwise_affine=False)

        self.to_gb = nn.Sequential(
            nn.Linear(style_dim, hidden_mult * d_model),
            nn.GELU(),
            nn.Linear(hidden_mult * d_model, 2 * d_model),
        )

    def forward(self, x: torch.Tensor, s: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B, N, D], s: [B, style_dim]
        x = self.ln(x)

        if s is None:
            return x

        gb = self.to_gb(s)  # [B, 2D]
        gamma, beta = gb.chunk(2, dim=-1)  # each [B, D]

        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)

        return x * (1.0 + gamma) + beta


# ---------------------------------------------------------------------------
# Transformer encoder layer
# ---------------------------------------------------------------------------
class StyleConditionedEncoderLayer(nn.Module):
    """
    Transformer encoder layer.

    Can use:
      - standard LayerNorm
      - AdaLayerNorm conditioned on z
    """

    def __init__(self, d_model: int, nhead: int, ff_mult: int = 4, dropout: float = 0.1, use_adaln: bool = True):
        super().__init__()

        self.use_adaln = use_adaln

        self.adaln1 = AdaLayerNorm(d_model) if use_adaln else nn.LayerNorm(d_model)
        self.adaln2 = AdaLayerNorm(d_model) if use_adaln else nn.LayerNorm(d_model)

        self.attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=True,
        )

        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * d_model, d_model),
        )

        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        s: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:

        # First pre-norm before attention
        h = self.adaln1(x, s) if self.use_adaln else self.adaln1(x)

        # Self-attention
        attn_out, _ = self.attn(
            h,
            h,
            h,
            attn_mask=attn_mask,
            need_weights=False,
        )

        # Residual connection
        x = x + self.drop(attn_out)

        # Second pre-norm before feedforward
        h = self.adaln2(x, s) if self.use_adaln else self.adaln2(x)

        # Feedforward + residual
        x = x + self.drop(self.ff(h))

        return x


# ---------------------------------------------------------------------------
# Full transformer encoder
# ---------------------------------------------------------------------------
class StyleConditionedTransformerEncoder(nn.Module):
    """
    Stack of style-conditioned transformer layers.
    """

    def __init__(self, d_model: int, nhead: int, num_layers: int, ff_mult: int = 4, dropout: float = 0.1, use_adaln: bool = True):
        super().__init__()

        self.layers = nn.ModuleList([
            StyleConditionedEncoderLayer(
                d_model,
                nhead,
                ff_mult=ff_mult,
                dropout=dropout,
                use_adaln=use_adaln,
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        s: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        for layer in self.layers:
            x = layer(
                x,
                s,
                attn_mask=attn_mask,
            )

        return x
