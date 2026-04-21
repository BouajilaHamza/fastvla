"""
Core FastVLA model implementation.
Refactored for production-grade reliability and environment stability.
"""

import logging
import torch

# ── 1. Absolute Preemptive Imports (MUST be first) ───────────────────────
# We initialize unsloth and accelerate before ANY transformers import.
try:
    import accelerate
    import accelerate.big_modeling
except ImportError:
    pass

UNSLOTH_AVAILABLE = False
FastLanguageModel = None
FastVisionModel = None
try:
    import unsloth
    from unsloth import (
        FastLanguageModel as _FLM, 
        FastVisionModel as _FVM,
        patch_model,
        patch_forward,
        patch_saving_functions
    )
    FastLanguageModel, FastVisionModel = _FLM, _FVM
    UNSLOTH_AVAILABLE = True
except ImportError:
    pass

# ── 2. Standard Imports ──────────────────────────────────────────────────
import torch.nn as nn
import torch._dynamo
from pathlib import Path
from typing import Optional, Dict, Any, Union
from transformers import AutoTokenizer, PreTrainedModel, AutoModel, AutoModelForCausalLM, AutoConfig

from .config import FastVLAConfig
from .kernels import vision_language_fusion_forward, TritonActionHead
from .optimization import enable_gradient_checkpointing, get_peft_config
from .utils import check_environment, get_gpu_memory_report, get_device
from .exceptions import ModelLoadingError, DistributedTrainingError, QuantizationError
from .registry import VLAModelRegistry

# Setup logging
logger = logging.getLogger(__name__)

# ── Internal Helpers ──────────────────────────────────────────────────────

def _get_target_device_map(config: FastVLAConfig) -> Union[str, Dict]:
    """Calculate the safest device map for the current environment."""
    if not torch.cuda.is_available():
        return "cpu"
    
    # If explicit device map is provided, use it
    if config.device_map not in ["auto", "balanced"]:
        return config.device_map
        
    # On multi-GPU (Kaggle T4 x2), "auto" creates AlignDevicesHook which
    # crashes Dynamo. We prefer a static mapping for 4-bit models.
    if config.load_in_4bit:
        # Default to current device (usually GPU 0)
        curr = torch.cuda.current_device()
        return {"": curr}
        
    return config.device_map


# ── Dummy modules for testing / CPU-only validation ──────────────────────

class DummyVisionEncoder(nn.Module):
    def __init__(self, hidden_size: int = 768, **kwargs):
        super().__init__()
        self.embed_dim = hidden_size
        self.config = type("Config", (), {"hidden_size": hidden_size})()
        self.patch_embed = nn.Conv2d(3, hidden_size, kernel_size=16, stride=16)
        self.norm = nn.LayerNorm(hidden_size)
    def forward(self, pixel_values, **kwargs):
        x = self.patch_embed(pixel_values).flatten(2).transpose(1, 2)
        return self.norm(x)

class DummyLanguageModel(nn.Module):
    def __init__(self, hidden_size: int = 128, vocab_size: int = 50257, **kwargs):
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
            device_map = _get_target_device_map(config)
            
            from .adapters.vision import get_vision_adapter, OpenVLAFusedVisionAdapter
            
            # CRITICAL: Always use OpenVLA adapter if it's OpenVLA, bypassing standard AutoModel
            v_name = str(config.vision_encoder_name)
            if "openvla" in v_name.lower():
                self.vision_encoder = OpenVLAFusedVisionAdapter.from_pretrained(
                    v_name, device_map=device_map, 
                    load_in_4bit=config.load_in_4bit, hf_token=config.hf_token
                )
            else:
                # Use model registry to get vision config if available
                reg_config = VLAModelRegistry.get(v_name)
                v_config = reg_config.vision.to_dict() if reg_config else {"model_name": v_name, "model_type": "vit"}
                v_config["load_in_4bit"] = config.load_in_4bit
                self.vision_encoder = get_vision_adapter(v_config, device_map=device_map, hf_token=config.hf_token)

        # 3. Language Model
        if config.dummy:
            self._tokenizer = AutoTokenizer.from_pretrained("gpt2")
            self._tokenizer.pad_token = self._tokenizer.eos_token
            actual_vocab = len(self._tokenizer)
            self.llm = DummyLanguageModel(hidden_size=config.llm_hidden_size, vocab_size=actual_vocab)
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

        # 6. Apply Unsloth Patches (Safety layer if not already handled by root)
        if not config.dummy and UNSLOTH_AVAILABLE and torch.cuda.is_available():
            try:
                self.llm = patch_model(self.llm)
                patch_forward(self.llm)
            except Exception as e:
                logger.warning(f"Unsloth patching failed (skipping): {e}")

        # 7. Model Stability (Distributed)
        if not config.dummy and torch.cuda.device_count() > 1:
            self._stabilize_distributed_hooks()

        # 8. Action Dimension Check
        if config.action_dim != self.action_head.action_dim:
            logger.warning(
                f"Config action_dim ({config.action_dim}) != Head action_dim ({self.action_head.action_dim}). Syncing..."
            )
            config.action_dim = self.action_head.action_dim

        # 9. Handle PEFT (LoRA) for Dummies / Fallbacks
        if config.use_peft and config.dummy:
            self._apply_peft_freezing(config)

    def save_pretrained(self, save_directory: Union[str, Path], **kwargs):
        """Unified save method that handles LoRA adapters and model configs correctly."""
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        
        # 1. Save Config
        self.config.save_pretrained(save_directory)
        
        # 2. Save Model (Handles LoRA automatically if PEFT is wrapped)
        self.llm.save_pretrained(save_directory, **kwargs)
        
        # 3. Save Heads
        torch.save(self.vision_proj.state_dict(), save_directory / "vision_proj.bin")
        torch.save(self.action_head.state_dict(), save_directory / "action_head.bin")
        
        # 4. Save Tokenizer
        if self.tokenizer:
            self.tokenizer.save_pretrained(save_directory)

    def push_to_hub(self, repo_id: str, token: Optional[str] = None, **kwargs):
        """Push the model and its components to the HF Hub."""
        from huggingface_hub import HfApi
        api = HfApi()
        
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.save_pretrained(tmp_dir, **kwargs)
            api.upload_folder(
                folder_path=tmp_dir,
                repo_id=repo_id,
                token=token,
                **kwargs
            )
        logger.info(f"Model pushed to https://huggingface.co/{repo_id}")

    def _apply_peft_freezing(self, config: FastVLAConfig):
        """Freeze base model parameters if PEFT is enabled."""
        for param in self.parameters():
            param.requires_grad = False
        for param in self.action_head.parameters():
            param.requires_grad = True
        for param in self.vision_proj.parameters():
            param.requires_grad = True

    def _stabilize_distributed_hooks(self):
        """Remove Accelerate hooks that cause Dynamo graph breaks."""
        try:
            from accelerate.hooks import remove_hook_from_module
            target_device = get_device()
            remove_hook_from_module(self.vision_encoder, recurse=True)
            remove_hook_from_module(self.llm, recurse=True)
            
            if not getattr(self, "is_loaded_in_4bit", False) and target_device == "cuda":
                self.vision_encoder.to(target_device)
                self.llm.to(target_device)
                self.vision_proj.to(target_device)
                self.action_head.to(target_device)
                
            torch._dynamo.disable(self.vision_encoder.forward)
            torch._dynamo.disable(self.llm.forward)
        except Exception as e:
            logger.debug(f"Distributed stabilization skipped: {e}")

    def _sync_config_with_loaded_models(self):
        """Update config attributes to match actual loaded model dimensions."""
        self.config.vision_hidden_size = self.vision_encoder.embed_dim
        l_conf = self.llm.config
        self.config.llm_hidden_size = getattr(l_conf, "hidden_size", getattr(l_conf, "word_embed_proj_dim", 4096))

    def _load_component(self, component_type: str, config: FastVLAConfig):
        """Unified loader for Model components (Vision/LLM) with smart conflict resolution."""
        device_map = _get_target_device_map(config)
        if component_type == "llm":
            return self._load_language_model_internal(config, device_map)
        raise ValueError("Use get_vision_adapter for vision components.")

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
                    llm = FastLanguageModel.get_peft_model(
                        llm, r=config.lora_rank, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout,
                        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                    )
                return llm
            except Exception as e:
                logger.warning(f"Unsloth LLM load failed: {e}. Falling back to HF...")

        # Path 2: Standard HF
        from .optimization import get_quantization_config
        kwargs = {"device_map": device_map, "token": config.hf_token, "trust_remote_code": True}
        if config.load_in_4bit:
            kwargs["quantization_config"] = get_quantization_config(load_in_4bit=True)
        
        llm = AutoModelForCausalLM.from_pretrained(config.llm_name, **kwargs)
        self._tokenizer = AutoTokenizer.from_pretrained(config.llm_name, token=config.hf_token, trust_remote_code=True)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        if config.use_peft:
            from peft import get_peft_model
            llm = get_peft_model(llm, get_peft_config(config.lora_rank))
        return llm

    @property
    def tokenizer(self): return getattr(self, "_tokenizer", None)

    def forward(self, pixel_values, input_ids, attention_mask=None, labels=None):
        """Forward pass handles multi-camera inputs and action prediction."""
        visual_features = []
        for cam_idx in range(pixel_values.size(1)):
            cam_images = pixel_values[:, cam_idx]
            # Use the adapter interface directly
            cam_feats = self.vision_encoder(cam_images)
            
            # Project to LLM dimension if needed
            proj_device = next(self.vision_proj.parameters()).device
            cam_feats = self.vision_proj(cam_feats.to(proj_device))
            visual_features.append(cam_feats)

        # Concatenate multi-camera features along sequence dimension [B, num_cams * T_v, D]
        visual_features = torch.cat(visual_features, dim=1)
        
        llm_device = next(self.llm.parameters()).device
        text_embeds = self.llm.get_input_embeddings()(input_ids.to(llm_device))
        
        # Use optimized Cross-Attention Fusion (Text attends to Visual)
        from .kernels import vision_language_cross_attention
        fused_embeds = vision_language_cross_attention(text_embeds, visual_features.to(llm_device))
        outputs = self.llm(inputs_embeds=fused_embeds, attention_mask=attention_mask, output_hidden_states=True)

        head_device = next(self.action_head.parameters()).device
        pooled = outputs.hidden_states[-1].mean(dim=1).to(head_device)
        action_preds = self.action_head(pooled)

        loss = None
        if labels is not None:
            labels = labels.to(device=head_device, dtype=action_preds.dtype)
            if action_preds.shape != labels.shape:
                raise ValueError(
                    f"Action dimension mismatch: model predicts {action_preds.shape} but labels have {labels.shape}."
                )
            loss = nn.MSELoss()(action_preds, labels)

        return action_preds, loss

    @classmethod
    def from_pretrained(cls, model_name_or_path: Optional[str] = None, **kwargs):
        """Load a FastVLA model from registry or HF."""
        if model_name_or_path:
            reg_config = VLAModelRegistry.get(model_name_or_path)
            if reg_config:
                kwargs["vision_encoder_name"] = reg_config.vision.model_name
                kwargs["llm_name"] = reg_config.llm.model_name
                kwargs["image_size"] = reg_config.vision.image_size
            else:
                if "vision_encoder_name" not in kwargs:
                    kwargs["vision_encoder_name"] = model_name_or_path
                if "llm_name" not in kwargs:
                    kwargs["llm_name"] = model_name_or_path
        config = FastVLAConfig(**kwargs)
        return cls(config)
