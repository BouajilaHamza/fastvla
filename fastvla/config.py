import os
from typing import Optional
from transformers.configuration_utils import PretrainedConfig


def _auto_torch_compile() -> bool:
    """Default ``use_torch_compile`` policy: on for Ada (sm_89) + Hopper (sm_90+),
    off elsewhere. Centralised here so tests can patch one symbol.

    Bitsandbytes 4-bit kernels currently regress under ``torch.compile`` on
    Turing (sm_75) and have marginal-to-negative benefit on Ampere (sm_80);
    Ada and Hopper are reliable wins, so the default flips on there.
    """
    import torch

    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return (major, minor) >= (8, 9)


class FastVLAConfig(PretrainedConfig):
    """Configuration class for FastVLA model."""

    model_type = "fastvla"

    def __init__(
        self,
        # Model selection
        vision_encoder_name: str = "google/vit-base-patch16-224",
        llm_name: str = "unsloth/Llama-3.2-1B-Instruct",
        hf_token: Optional[str] = None,
        # Dummy mode
        dummy: bool = False,
        vision_hidden_size: int = 768,
        llm_hidden_size: int = 128,
        llm_num_layers: int = 2,
        num_attention_heads: int = 4,
        vocab_size: int = 1000,
        # Vision encoder details
        image_size: int = 224,
        patch_size: int = 16,
        # Language model details
        max_sequence_length: int = 2048,
        attn_implementation: Optional[str] = None,  # e.g. "flash_attention_2"; None = auto
        # Action head
        action_dim: int = 7,
        action_hidden_dim: int = 256,
        chunk_size: int = 1,
        head_type: str = "mlp_continuous",  # "mlp_continuous", "mlp_discrete", "flow_matching"
        num_bins: int = 256,  # bins for the discrete head
        num_inference_steps: int = 10,  # ODE steps for the flow-matching head
        pooling_strategy: str = "masked_mean",  # "masked_mean", "mean", "last"
        # Visual token budget (Roadmap Thrust A / H1). None = no reduction.
        # When set, the visual token sequence is reduced to this many tokens
        # before the vision->language projection, shrinking the LLM sequence
        # length (the dominant compute term for a VLA).
        visual_token_budget: Optional[int] = None,
        token_reduction_strategy: str = "attention_pool",  # mean_pool|attention_pool|perceiver|token_merge
        token_reduction_heads: int = 8,
        # How visual tokens enter the language model:
        #   "cross_attention" — text queries attend to visual K/V (default; the
        #        token budget shrinks the cross-attention KV + projection cost).
        #   "concat"          — reduced visual tokens are prepended to the LLM
        #        input sequence (OpenVLA-style), so the token budget directly
        #        shrinks the LLM sequence length — the dominant compute term (H1).
        fusion_mode: str = "cross_attention",
        # Action Normalization
        norm_min: Optional[list] = None,
        norm_max: Optional[list] = None,
        # Training
        learning_rate: float = 5e-5,
        loss_type: str = "l1",  # "l1", "mse", "huber"
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        # Quantization
        load_in_4bit: bool = False,
        bnb_4bit_quant_type: str = "nf4",
        bnb_4bit_compute_dtype: str = "float16",
        # PEFT / LoRA
        use_peft: bool = False,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        # Memory
        gradient_checkpointing: bool = True,
        freeze_vision_encoder: bool = True,
        use_torch_compile: Optional[bool] = None,  # None = auto-detect (Ada/Hopper)
        use_relative_delta: bool = False,
        device_map: str = "auto",
        # RL Integration
        use_rl: bool = False,
        rl_hidden_dim: int = 256,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.device_map = device_map
        self.vision_encoder_name = vision_encoder_name
        self.llm_name = llm_name
        self.hf_token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HF_API_KEY")
        self.dummy = dummy
        self.vision_hidden_size = vision_hidden_size
        self.llm_hidden_size = llm_hidden_size
        self.llm_num_layers = llm_num_layers
        self.num_attention_heads = num_attention_heads
        self.vocab_size = vocab_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.max_sequence_length = max_sequence_length
        self.attn_implementation = attn_implementation
        self.action_dim = action_dim
        self.action_hidden_dim = action_hidden_dim
        self.chunk_size = chunk_size
        self.head_type = head_type
        self.num_bins = num_bins
        self.num_inference_steps = num_inference_steps
        self.pooling_strategy = pooling_strategy
        self.visual_token_budget = visual_token_budget
        self.token_reduction_strategy = token_reduction_strategy
        self.token_reduction_heads = token_reduction_heads
        self.fusion_mode = fusion_mode
        self.norm_min = norm_min
        self.norm_max = norm_max
        self.learning_rate = learning_rate
        self.loss_type = loss_type
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.load_in_4bit = load_in_4bit
        self.bnb_4bit_quant_type = bnb_4bit_quant_type
        self.bnb_4bit_compute_dtype = bnb_4bit_compute_dtype
        self.use_peft = use_peft
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.gradient_checkpointing = gradient_checkpointing
        self.freeze_vision_encoder = freeze_vision_encoder
        self.use_torch_compile = (
            _auto_torch_compile() if use_torch_compile is None else use_torch_compile
        )
        self.use_relative_delta = use_relative_delta
        self.use_rl = use_rl
        self.rl_hidden_dim = rl_hidden_dim
