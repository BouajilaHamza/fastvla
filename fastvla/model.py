"""
Core FastVLA model implementation.
Refactored for production-grade reliability, distributed sharding support,
and robust 4-bit device placement.
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, Union
from transformers import AutoTokenizer, PreTrainedModel, AutoModel, AutoModelForCausalLM

from .config import FastVLAConfig
from .kernels import vision_language_fusion_forward, TritonActionHead
from .optimization import enable_gradient_checkpointing, get_peft_config
from .utils import check_environment, get_gpu_memory_report

# Setup logging
logger = logging.getLogger(__name__)

# ── Optional Unsloth import ──────────────────────────────────────────────
UNSLOTH_AVAILABLE = False
FastLanguageModel = None
FastVisionModel = None

def _dummy_patch_model(model): return model
def _dummy_patch_forward(model): pass
def _dummy_patch_saving_functions(): pass

patch_model = _dummy_patch_model
patch_forward = _dummy_patch_forward
patch_saving_functions = _dummy_patch_saving_functions

try:
    from unsloth import FastLanguageModel as _FLM, FastVisionModel as _FVM
    FastLanguageModel, FastVisionModel = _FLM, _FVM
    
    try:
        from unsloth import patch_forward as _pf, patch_model as _pm, patch_saving_functions as _ps
        patch_forward, patch_model, patch_saving_functions = _pf, _pm, _ps
    except ImportError:
        try:
            from unsloth.models.patcher import patch_forward as _pf, patch_model as _pm
            from unsloth.models.loader import patch_saving_functions as _ps
            patch_forward, patch_model, patch_saving_functions = _pf, _pm, _ps
        except ImportError:
            pass
    UNSLOTH_AVAILABLE = True
except ImportError:
    pass


# ── Internal Helpers ──────────────────────────────────────────────────────

def _get_target_device_map(config: FastVLAConfig) -> Union[str, Dict]:
    """Calculate the safest device map for the current environment."""
    if not torch.cuda.is_available():
        return "cpu"
    
    # If explicit device map is provided, use it
    if config.device_map not in ["auto", "balanced"]:
        return config.device_map
        
    # On multi-GPU (Kaggle T4 x2), "auto" creates AlignDevicesHook which
    # crashes Dynamo. We prefer a static mapping to GPU 0 for 4-bit single-process.
    if config.load_in_4bit:
        return {"": 0}
        
    return config.device_map


# ── Dummy modules for testing / CPU-only validation ──────────────────────

class DummyVisionEncoder(nn.Module):
    def __init__(self, hidden_size: int = 768, **kwargs):
        super().__init__()
        self.config = type("Config", (), {"hidden_size": hidden_size})()
        self.patch_embed = nn.Conv2d(3, hidden_size, kernel_size=16, stride=16)
        self.norm = nn.LayerNorm(hidden_size)
    def forward(self, pixel_values, **kwargs):
        x = self.patch_embed(pixel_values).flatten(2).transpose(1, 2)
        return type("Output", (), {"last_hidden_state": self.norm(x)})()

class DummyLanguageModel(nn.Module):
    def __init__(self, hidden_size: int = 128, vocab_size: int = 1000, **kwargs):
        super().__init__()
        self.config = type("Config", (), {"hidden_size": hidden_size, "vocab_size": vocab_size})()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layer = nn.Linear(hidden_size, hidden_size)
    def get_input_embeddings(self): return self.embed_tokens
    def forward(self, inputs_embeds=None, **kwargs):
        x = self.layer(inputs_embeds)
        return type("Output", (), {"last_hidden_state": x, "hidden_states": (x,)})()


# ── Main model ────────────────────────────────────────────────────────────

class FastVLAModel(PreTrainedModel):
    config_class = FastVLAConfig
    supports_gradient_checkpointing = True

    def __init__(self, config: FastVLAConfig):
        super().__init__(config)
        self.config = config
        
        # 1. Health Check
        if not config.dummy:
            check_environment(require_cuda=config.load_in_4bit)
            logger.info(f"Initializing FastVLA with {get_gpu_memory_report()}")

        # 2. Vision Encoder
        if config.dummy:
            self.vision_encoder = DummyVisionEncoder(hidden_size=config.vision_hidden_size)
        else:
            self.vision_encoder = self._load_component("vision", config)

        # 3. Language Model
        if config.dummy:
            self.llm = DummyLanguageModel(hidden_size=config.llm_hidden_size, vocab_size=config.vocab_size)
            self._tokenizer = AutoTokenizer.from_pretrained("gpt2")
            self._tokenizer.pad_token = self._tokenizer.eos_token
        else:
            self.llm = self._load_component("llm", config)

        # 4. Sync Hidden Sizes
        if not config.dummy:
            self._sync_config_with_loaded_models()

        # 5. Multimodal Projection & Action Head
        # Initialize on the same device as the LLM entry point
        llm_device = next(self.llm.parameters()).device
        self.vision_proj = nn.Linear(config.vision_hidden_size, config.llm_hidden_size).to(llm_device)
        nn.init.xavier_uniform_(self.vision_proj.weight)

        self.action_head = TritonActionHead(
            config.llm_hidden_size, config.action_hidden_dim, config.action_dim
        ).to(llm_device)

        # 6. Unsloth Optimizations
        if not config.dummy and UNSLOTH_AVAILABLE and torch.cuda.is_available():
            self._apply_unsloth_patches()

        # 7. Final Stabilization
        self.is_loaded_in_4bit = config.load_in_4bit
        if config.gradient_checkpointing:
            enable_gradient_checkpointing(self)

    def _sync_config_with_loaded_models(self):
        """Update config attributes to match actual loaded model dimensions."""
        v_conf = self.vision_encoder.config
        self.config.vision_hidden_size = getattr(v_conf, "hidden_size", getattr(v_conf, "projection_dim", 768))
        
        l_conf = self.llm.config
        self.config.llm_hidden_size = getattr(l_conf, "hidden_size", getattr(l_conf, "word_embed_proj_dim", 4096))

    def _load_component(self, component_type: str, config: FastVLAConfig):
        """Unified loader for Model components (Vision/LLM)."""
        device_map = _get_target_device_map(config)
        
        if component_type == "vision":
            return self._load_vision_encoder_internal(config, device_map)
        else:
            return self._load_language_model_internal(config, device_map)

    def _load_vision_encoder_internal(self, config, device_map):
        # Path 1: Standard HF + BNB (Reliable)
        if config.load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
                bnb_cfg = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True
                )
                return AutoModel.from_pretrained(
                    config.vision_encoder_name, quantization_config=bnb_cfg,
                    device_map=device_map, token=config.hf_token, trust_remote_code=True
                )
            except Exception as e:
                logger.warning(f"Standard 4-bit vision load failed: {e}. Trying Unsloth...")

        # Path 2: Unsloth Fallback
        if UNSLOTH_AVAILABLE and torch.cuda.is_available():
            try:
                return FastVisionModel.from_pretrained(
                    config.vision_encoder_name, load_in_4bit=config.load_in_4bit,
                    device_map=device_map, token=config.hf_token
                )
            except Exception as e:
                logger.error(f"Unsloth vision load failed: {e}")

        # Path 3: FP32 Fallback
        return AutoModel.from_pretrained(
            config.vision_encoder_name, device_map=device_map, token=config.hf_token
        )

    def _load_language_model_internal(self, config, device_map):
        # Path 1: Unsloth (Performance)
        if UNSLOTH_AVAILABLE and torch.cuda.is_available():
            try:
                llm, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=config.llm_name, max_seq_length=config.max_sequence_length,
                    load_in_4bit=config.load_in_4bit, device_map=device_map, token=config.hf_token
                )
                self._tokenizer = tokenizer
                if config.use_peft:
                    llm = FastLanguageModel.get_peft_model(llm, get_peft_config(config.lora_rank))
                return llm
            except Exception as e:
                logger.warning(f"Unsloth LLM load failed: {e}. Falling back to HF...")

        # Path 2: Standard HF
        from .optimization import get_quantization_config
        kwargs = {"device_map": device_map, "token": config.hf_token}
        if config.load_in_4bit:
            kwargs["quantization_config"] = get_quantization_config(load_in_4bit=True)
        
        llm = AutoModelForCausalLM.from_pretrained(config.llm_name, **kwargs)
        self._tokenizer = AutoTokenizer.from_pretrained(config.llm_name, token=config.hf_token)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
            
        if config.use_peft:
            from peft import get_peft_model
            llm = get_peft_model(llm, get_peft_config(config.lora_rank))
        return llm

    def _apply_unsloth_patches(self):
        """Safe application of Unsloth patches after model loading."""
        try:
            self.llm = patch_model(self.llm)
            patch_saving_functions()
            patch_forward(self.llm)
        except Exception as e:
            logger.warning(f"Unsloth patching failed (skipping): {e}")

    @property
    def tokenizer(self): return getattr(self, "_tokenizer", None)

    def forward(self, pixel_values, input_ids, attention_mask=None, labels=None):
        """Forward pass handles sharded inputs across devices automatically."""
        # 1. Vision Encoding
        visual_features = []
        for cam_idx in range(pixel_values.size(1)):
            cam_images = pixel_values[:, cam_idx]
            # Ensure images are on the same device as the first vision layer
            vision_device = next(self.vision_encoder.parameters()).device
            vision_out = self.vision_encoder(pixel_values=cam_images.to(vision_device), return_dict=True)
            
            # Project onto LLM space (proj is on LLM entry device)
            proj_device = next(self.vision_proj.parameters()).device
            cam_feats = self.vision_proj(vision_out.last_hidden_state.to(proj_device))
            visual_features.append(cam_feats)

        # Average fusion
        visual_features = torch.stack(visual_features).mean(dim=0)

        # 2. Text Embed-Fusion
        llm_device = next(self.llm.parameters()).device
        text_embeds = self.llm.get_input_embeddings()(input_ids.to(llm_device))
        
        fused_embeds = vision_language_fusion_forward(visual_features.to(llm_device), text_embeds)

        # 3. LLM Core
        outputs = self.llm(inputs_embeds=fused_embeds, attention_mask=attention_mask, output_hidden_states=True)

        # 4. Action Prediction
        head_device = next(self.action_head.parameters()).device
        pooled = outputs.hidden_states[-1].mean(dim=1).to(head_device)
        action_preds = self.action_head(pooled)

        # 5. Loss
        loss = None
        if labels is not None:
            labels = labels.to(device=head_device, dtype=action_preds.dtype)
            loss = nn.MSELoss()(action_preds, labels)

        return action_preds, loss

    @classmethod
    def from_pretrained(cls, **kwargs):
        """Compatible entry point for loading models."""
        config = FastVLAConfig(**kwargs)
        return cls(config)
