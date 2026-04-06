"""
Core FastVLA model implementation.
Supports any HuggingFace vision encoder + language model, with optional
Unsloth optimizations, dummy mode for testing, and CPU/GPU auto-selection.
"""

import torch
import torch.nn as nn
from typing import Optional
from transformers import AutoTokenizer, PreTrainedModel
from transformers import ViTModel, AutoModelForCausalLM

from .config import FastVLAConfig
from .kernels import vision_language_fusion_forward, TritonActionHead
from .optimization import (
    enable_gradient_checkpointing,
    get_peft_config,
)
from .utils import get_device


# ── Optional Unsloth import ──────────────────────────────────────────────
UNSLOTH_AVAILABLE = False
try:
    from unsloth import (
        FastLanguageModel,
        FastVisionModel,
        patch_forward,
        patch_model,
        patch_saving_functions,
    )

    UNSLOTH_AVAILABLE = True
except ImportError:
    pass


# ── Dummy modules for testing / CPU-only validation ──────────────────────
class DummyVisionEncoder(nn.Module):
    """Tiny vision encoder that mimics ViT output shape."""

    def __init__(
        self, hidden_size: int = 768, image_size: int = 224, patch_size: int = 16
    ):
        super().__init__()
        self.config = type("Config", (), {"hidden_size": hidden_size})()
        self.patch_size = patch_size
        self.patch_embed = nn.Conv2d(
            3, hidden_size, kernel_size=patch_size, stride=patch_size
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, pixel_values, return_dict=False):
        x = self.patch_embed(pixel_values)  # [B, H, H_out, W_out]
        b, h, h_out, w_out = x.shape
        num_patches = h_out * w_out

        # Create positional embedding as a buffer (not parameter) to avoid registration issues
        if (
            not hasattr(self, "pos_embed_buf")
            or self.pos_embed_buf.shape[1] != num_patches
            or self.pos_embed_buf.device != x.device
        ):
            pos_embed = (
                torch.randn(1, num_patches, h, device=x.device, dtype=x.dtype) * 0.02
            )
            self.register_buffer("pos_embed_buf", pos_embed, persistent=False)

        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, H]
        x = x + self.pos_embed_buf
        x = self.norm(x)
        if return_dict:
            return type("Output", (), {"last_hidden_state": x})()
        return x


class DummyLanguageModel(nn.Module):
    """Tiny transformer-like model using MLPs (avoids SDPA bugs on CPU)."""

    def __init__(
        self,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        vocab_size: int = 1000,
    ):
        super().__init__()
        self.config = type(
            "Config",
            (),
            {
                "hidden_size": hidden_size,
                "num_hidden_layers": num_layers,
                "vocab_size": vocab_size,
            },
        )()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        # Simple MLP blocks instead of attention (avoids SDPA issues)
        layers = []
        for _ in range(num_layers):
            layers.append(
                nn.Sequential(
                    nn.LayerNorm(hidden_size),
                    nn.Linear(hidden_size, hidden_size * 4),
                    nn.GELU(),
                    nn.Linear(hidden_size * 4, hidden_size),
                )
            )
        self.layers = nn.ModuleList(layers)
        self.norm = nn.LayerNorm(hidden_size)

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        output_hidden_states=True,
        use_cache=False,
        **kwargs,
    ):
        if inputs_embeds is not None:
            x = inputs_embeds
        else:
            x = self.embed_tokens(input_ids)

        hidden_states_list = [x]
        for layer in self.layers:
            x = x + layer(x)
            hidden_states_list.append(x)

        x = self.norm(x)
        return type(
            "Output",
            (),
            {
                "last_hidden_state": x,
                "hidden_states": tuple(hidden_states_list),
            },
        )()


# ── Main model ────────────────────────────────────────────────────────────
class FastVLAModel(PreTrainedModel):
    """
    FastVLA: Flexible Vision-Language-Action model for robotics.
    Accepts any HuggingFace vision encoder + language model.
    """

    config_class = FastVLAConfig
    supports_gradient_checkpointing = True

    def __init__(self, config: FastVLAConfig):
        super().__init__(config)
        self.config = config
        self._device = get_device()

        # ── Vision encoder ────────────────────────────────────────────
        if config.dummy:
            self.vision_encoder = DummyVisionEncoder(
                hidden_size=config.vision_hidden_size,
                image_size=config.image_size,
                patch_size=config.patch_size,
            )
        else:
            self.vision_encoder = self._load_vision_encoder(config)

        # ── Language model ────────────────────────────────────────────
        if config.dummy:
            self.llm = DummyLanguageModel(
                hidden_size=config.llm_hidden_size,
                num_layers=config.llm_num_layers,
                vocab_size=config.vocab_size,
            )
            # Add a placeholder tokenizer for dummy mode to avoid collator failures
            self._tokenizer = AutoTokenizer.from_pretrained("gpt2")
            self._tokenizer.pad_token = self._tokenizer.eos_token
        else:
            self.llm = self._load_language_model(config)

        # ── Synchronize hidden sizes (Auto-detect) ────────────────────
        if not config.dummy:
            # Update config with actual loaded hidden sizes
            if hasattr(self.vision_encoder.config, "hidden_size"):
                config.vision_hidden_size = self.vision_encoder.config.hidden_size
            elif hasattr(self.vision_encoder.config, "projection_dim"):
                config.vision_hidden_size = self.vision_encoder.config.projection_dim
                
            if hasattr(self.llm.config, "hidden_size"):
                config.llm_hidden_size = self.llm.config.hidden_size
            elif hasattr(self.llm.config, "word_embed_proj_dim"):
                config.llm_hidden_size = self.llm.config.word_embed_proj_dim

        # ── Action head ───────────────────────────────────────────────
        self.action_head = TritonActionHead(
            config.llm_hidden_size,
            config.action_hidden_dim,
            config.action_dim,
        )

        # ── Vision → LLM projection ───────────────────────────────────
        self.vision_proj = nn.Linear(
            config.vision_hidden_size,
            config.llm_hidden_size,
        )
        nn.init.xavier_uniform_(self.vision_proj.weight)

        # ── Optional Unsloth patches (GPU only) ───────────────────────
        if not config.dummy and UNSLOTH_AVAILABLE and get_device() == "cuda":
            try:
                self.llm = patch_model(self.llm)
                patch_saving_functions()
                patch_forward(self.llm)
                if hasattr(self.llm, "to_bettertransformer"):
                    self.llm = self.llm.to_bettertransformer()
                if hasattr(self.vision_encoder, "to_bettertransformer"):
                    self.vision_encoder = self.vision_encoder.to_bettertransformer()
            except Exception:
                pass  # Skip Unsloth patches if they fail

        # ── Gradient checkpointing ────────────────────────────────────
        if config.gradient_checkpointing:
            enable_gradient_checkpointing(self)

    # ── Loader helpers ─────────────────────────────────────────────────
    def _load_vision_encoder(self, config):
        """Load a HuggingFace vision encoder."""
        # Try Unsloth first (GPU only), fall back to standard HF
        if UNSLOTH_AVAILABLE and get_device() == "cuda":
            try:
                return FastVisionModel.from_pretrained(
                    config.vision_encoder_name,
                    load_in_4bit=config.load_in_4bit,
                    token=config.hf_token,
                    device_map="auto",
                    torch_dtype=torch.bfloat16,
                )
            except Exception:
                pass

        device_map = getattr(config, "device_map", "auto") if get_device() == "cuda" else None
        
        return ViTModel.from_pretrained(
            config.vision_encoder_name,
            torch_dtype=torch.float16 if config.load_in_4bit else torch.float32,
            device_map=device_map,
        )

    def _load_language_model(self, config):
        """Load a HuggingFace language model + tokenizer."""
        # Try Unsloth first (GPU only), fall back to standard HF
        if UNSLOTH_AVAILABLE and get_device() == "cuda":
            try:
                llm, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=config.llm_name,
                    max_seq_length=config.max_sequence_length,
                    dtype=torch.bfloat16,
                    load_in_4bit=config.load_in_4bit,
                    token=config.hf_token,
                    device_map="auto",
                    rope_scaling={"type": "dynamic", "factor": 2.0},
                )
                self._tokenizer = tokenizer
                if config.use_peft:
                    peft_config = get_peft_config(
                        r=config.lora_rank,
                        lora_alpha=config.lora_alpha,
                        lora_dropout=config.lora_dropout,
                    )
                    llm = FastLanguageModel.get_peft_model(llm, peft_config)
                return llm
            except Exception:
                pass

        device_map = getattr(config, "device_map", "auto") if get_device() == "cuda" else None
        
        # Standard HuggingFace fallback
        llm = AutoModelForCausalLM.from_pretrained(
            config.llm_name,
            torch_dtype=torch.float16 if config.load_in_4bit else torch.float32,
            device_map=device_map,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            config.llm_name,
            padding_side="right",
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        if config.use_peft:
            from peft import get_peft_model

            peft_config = get_peft_config(
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
            )
            llm = get_peft_model(llm, peft_config)

        return llm

    # ── Forward pass ───────────────────────────────────────────────────
    @property
    def tokenizer(self):
        return getattr(self, "_tokenizer", None)

    def forward(self, pixel_values, input_ids, attention_mask=None, labels=None):
        """
        Forward pass.

        Args:
            pixel_values: [B, num_cameras, C, H, W]
            input_ids:    [B, seq_len]
            attention_mask: [B, seq_len] or None
            labels:       [B, action_dim] or None

        Returns:
            action_preds: [B, action_dim]
            loss:         scalar or None
        """
        num_cameras = pixel_values.size(1)
        batch_size = pixel_values.size(0)

        # ── Encode each camera view ───────────────────────────────────
        visual_features = []
        vision_device = next(self.vision_encoder.parameters()).device
        proj_device = next(self.vision_proj.parameters()).device

        for cam_idx in range(num_cameras):
            # Move inputs to vision device (usually first GPU)
            cam_images = pixel_values[:, cam_idx].to(vision_device)
            vision_out = self.vision_encoder(pixel_values=cam_images, return_dict=True)

            # Project onto LLM space (move outputs to projector device)
            cam_feats = self.vision_proj(vision_out.last_hidden_state.to(proj_device))
            visual_features.append(cam_feats)

        # Average across cameras (move stack to common device)
        visual_features = torch.stack(visual_features, dim=0).to(proj_device).mean(dim=0)

        # ── Text embeddings ───────────────────────────────────────────
        llm_device = next(self.llm.parameters()).device
        text_embeds = self.llm.get_input_embeddings()(input_ids.to(llm_device))

        # ── Fuse visual + text ────────────────────────────────────────
        visual_features = visual_features.to(llm_device)
        if visual_features.size(1) != text_embeds.size(1):
            visual_features = visual_features.mean(dim=1, keepdim=True)
            visual_features = visual_features.expand(-1, text_embeds.size(1), -1)

        fused_embeds = vision_language_fusion_forward(visual_features, text_embeds)

        # ── LLM forward ───────────────────────────────────────────────
        if get_device() == "cuda":
            with torch.backends.cuda.sdp_kernel(
                enable_flash=True, enable_math=False, enable_mem_efficient=True
            ):
                outputs = self.llm(
                    inputs_embeds=fused_embeds,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                )
        else:
            outputs = self.llm(
                inputs_embeds=fused_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )

        # ── Pool & predict action ─────────────────────────────────────
        last_hidden = outputs.hidden_states[-1]
        pooled = last_hidden.mean(dim=1)

        # Ensure action head receives input on its correct device AND dtype
        head_device = next(self.action_head.parameters()).device
        head_dtype = next(self.action_head.parameters()).dtype
        action_preds = self.action_head(pooled.to(device=head_device, dtype=head_dtype))

        # ── Loss ──────────────────────────────────────────────────────
        loss = None
        if labels is not None:
            labels = labels.to(head_device)
            
            # Validate shapes match
            if action_preds.shape != labels.shape:
                # Handle shape mismatch - this can happen in distributed training
                # if different processes have different batch compositions
                if action_preds.shape[0] != labels.shape[0]:
                    # Batch size mismatch - take minimum
                    min_batch = min(action_preds.shape[0], labels.shape[0])
                    action_preds = action_preds[:min_batch]
                    labels = labels[:min_batch]
                
                # If action dimension mismatch, try to fix or raise informative error
                if action_preds.shape[1] != labels.shape[1]:
                    raise ValueError(
                        f"Action dimension mismatch: model predicts {action_preds.shape[1]} dims "
                        f"but labels have {labels.shape[1]} dims. "
                        f"Ensure your dataset's action dimensions match the model's action_dim config "
                        f"(model action_dim={self.config.action_dim}). "
                        f"Batch shape: {action_preds.shape}, Labels shape: {labels.shape}"
                    )
            
            loss = nn.MSELoss()(action_preds, labels)

        return action_preds, loss

    def generate(self, images, input_ids, **kwargs):
        """Generate actions from images and text."""
        action_preds, _ = self(images, input_ids)
        return action_preds

    # ── High-level loader ──────────────────────────────────────────────
    @classmethod
    def from_pretrained(
        cls,
        model_name: Optional[str] = None,
        vision_encoder_name: Optional[str] = None,
        llm_name: Optional[str] = None,
        config: Optional[FastVLAConfig] = None,
        dummy: bool = False,
        load_in_4bit: bool = False,
        max_seq_length: int = 2048,
        gradient_checkpointing: bool = True,
        use_peft: bool = False,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        token: Optional[str] = None,
        device_map: str = "auto",
        **kwargs,
    ):
        """
        Load a FastVLA model.

        Set `dummy=True` for a tiny random-weight model (fast testing / CPU).
        Otherwise downloads real models from HuggingFace.
        """
        if config is None:
            config = FastVLAConfig(
                vision_encoder_name=vision_encoder_name
                or "google/vit-base-patch16-224",
                llm_name=llm_name or "meta-llama/Llama-2-7b-hf",
                max_sequence_length=max_seq_length,
                load_in_4bit=load_in_4bit,
                use_peft=use_peft,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                gradient_checkpointing=gradient_checkpointing,
                hf_token=token,
                dummy=dummy,
                device_map=device_map,
                **kwargs,
            )

        model = cls(config)

        if gradient_checkpointing and not dummy:
            enable_gradient_checkpointing(model)

        return model
