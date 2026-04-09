from typing import Optional
from transformers.configuration_utils import PretrainedConfig


class FastVLAConfig(PretrainedConfig):
    """Configuration class for FastVLA model."""

    model_type = "fastvla"

    def __init__(
        self,
        # Model selection
        vision_encoder_name: str = "google/vit-base-patch16-224",
        llm_name: str = "meta-llama/Llama-2-7b-hf",
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
        # Action head
        action_dim: int = 7,
        action_hidden_dim: int = 256,
        # Training
        learning_rate: float = 5e-5,
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
        device_map: str = "auto",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.device_map = device_map
        self.vision_encoder_name = vision_encoder_name
        self.llm_name = llm_name
        self.hf_token = hf_token
        self.dummy = dummy
        self.vision_hidden_size = vision_hidden_size
        self.llm_hidden_size = llm_hidden_size
        self.llm_num_layers = llm_num_layers
        self.num_attention_heads = num_attention_heads
        self.vocab_size = vocab_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.max_sequence_length = max_sequence_length
        self.action_dim = action_dim
        self.action_hidden_dim = action_hidden_dim
        self.learning_rate = learning_rate
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
