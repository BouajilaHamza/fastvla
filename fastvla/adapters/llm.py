"""LLM Adapters for FastVLA — Unified interface for any language model."""
import torch.nn as nn


class BaseLLMAdapter(nn.Module):
    """Base class for all LLM adapters."""

    def __init__(self):
        super().__init__()

    def get_input_embeddings(self) -> nn.Embedding:
        raise NotImplementedError


class LLaMAQLoRAAdapter(BaseLLMAdapter):
    """
    LLaMA-2 with 4-bit QLoRA.
    Uses bitsandbytes for quantization and peft for LoRA.
    """

    def __init__(self, model, tokenizer, config: dict):
        """
        Args:
            model: Already-loaded and quantized LLaMA model
            tokenizer: Associated tokenizer
            config: LLMConfig.to_dict()
        """
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.get_input_embeddings()

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None,
                output_hidden_states=True, use_cache=False, **kwargs):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            **kwargs,
        )


class GenericLLMAdapter(BaseLLMAdapter):
    """
    Generic HuggingFace causal LM adapter.
    Works with any HF causal LM (Gemma, Qwen, etc.)
    """

    def __init__(self, model, tokenizer, config: dict):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.get_input_embeddings()

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None,
                output_hidden_states=True, use_cache=False, **kwargs):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            **kwargs,
        )


def get_llm_adapter(model, tokenizer, config: dict) -> BaseLLMAdapter:
    """Create an LLM adapter from config."""
    model_type = config.get("model_type", "generic")

    if model_type == "llama":
        return LLaMAQLoRAAdapter(model, tokenizer, config)
    else:
        return GenericLLMAdapter(model, tokenizer, config)
