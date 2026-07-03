"""Tests for visual token reducers (Roadmap Thrust A / H1)."""

import torch
import pytest

from fastvla.adapters.token_reducer import (
    get_token_reducer,
    MeanPoolReducer,
    AttentionPoolReducer,
    PerceiverResampler,
    TokenMergeReducer,
)
from fastvla import FastVLAModel, FastVLAConfig


STRATEGIES = ["mean_pool", "attention_pool", "perceiver", "token_merge"]


class TestTokenReducerShapes:
    @pytest.mark.parametrize("strategy", STRATEGIES)
    @pytest.mark.parametrize("k", [64, 32, 16])
    def test_reduces_to_budget(self, strategy, k):
        B, N, D = 2, 197, 128
        reducer = get_token_reducer(strategy, num_tokens=k, dim=D)
        x = torch.randn(B, N, D)
        out = reducer(x)
        assert out.shape == (B, k, D), f"{strategy}: got {out.shape}"

    @pytest.mark.parametrize("strategy", STRATEGIES)
    def test_passthrough_when_already_small(self, strategy):
        # Budget larger than input: parametric reducers still emit `k` queries,
        # parameter-free ones pass through unchanged.
        B, N, D = 2, 8, 64
        reducer = get_token_reducer(strategy, num_tokens=16, dim=D)
        out = reducer(torch.randn(B, N, D))
        if strategy in ("mean_pool", "token_merge"):
            assert out.shape[1] == N  # nothing to reduce -> unchanged
        else:
            assert out.shape[1] == 16

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError):
            get_token_reducer("does_not_exist", num_tokens=8, dim=64)


class TestTokenReducerGradients:
    @pytest.mark.parametrize("strategy", ["attention_pool", "perceiver"])
    def test_gradients_flow(self, strategy):
        B, N, D = 2, 100, 64
        reducer = get_token_reducer(strategy, num_tokens=32, dim=D)
        x = torch.randn(B, N, D, requires_grad=True)
        out = reducer(x)
        out.sum().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()
        # Learnable queries receive gradient.
        params = [p for p in reducer.parameters() if p.requires_grad]
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in params)

    def test_token_merge_is_parameter_free(self):
        reducer = get_token_reducer("token_merge", num_tokens=16, dim=32)
        assert sum(p.numel() for p in reducer.parameters()) == 0


class TestTokenReducerInModel:
    """The reducer must shrink the visual sequence inside a full forward pass."""

    def _config(self, budget, strategy="mean_pool"):
        return FastVLAConfig(
            dummy=True,
            vision_hidden_size=64,
            llm_hidden_size=64,
            llm_num_layers=2,
            num_attention_heads=4,
            vocab_size=500,
            action_dim=7,
            action_hidden_dim=32,
            gradient_checkpointing=False,
            visual_token_budget=budget,
            token_reduction_strategy=strategy,
        )

    def test_model_builds_reducer(self):
        model = FastVLAModel(self._config(budget=8))
        assert model.token_reducer is not None
        assert model.token_reducer.num_tokens == 8

    def test_disabled_by_default(self):
        model = FastVLAModel(self._config(budget=None))
        assert model.token_reducer is None

    @pytest.mark.parametrize("strategy", STRATEGIES)
    def test_forward_pass_with_reduction(self, strategy):
        model = FastVLAModel(self._config(budget=8, strategy=strategy))
        B, seq = 2, 8
        # 32x32 image, patch 16 -> 4 patches/cam, 2 cams -> 8 visual tokens.
        out, loss = model(
            pixel_values=torch.randn(B, 2, 3, 32, 32),
            input_ids=torch.randint(0, 500, (B, seq)),
            attention_mask=torch.ones(B, seq),
            labels=torch.randn(B, 7),
        )
        assert out.shape == (B, 7)
        assert loss is not None and torch.isfinite(loss)

    def test_concat_fusion_shrinks_llm_sequence(self):
        """In concat mode the token budget must reduce the LLM's input length."""
        from fastvla.model import DummyLanguageModel

        cfg = self._config(budget=4, strategy="mean_pool")
        cfg.fusion_mode = "concat"
        model = FastVLAModel(cfg)

        seen = {}
        orig = DummyLanguageModel.forward

        def spy(self, inputs_embeds=None, **k):
            seen["seq"] = inputs_embeds.shape[1]
            return orig(self, inputs_embeds=inputs_embeds, **k)

        B, seq = 2, 8  # 8 text tokens; 8 visual tokens reduced to 4
        DummyLanguageModel.forward = spy
        try:
            model(
                pixel_values=torch.randn(B, 2, 3, 32, 32),
                input_ids=torch.randint(0, 500, (B, seq)),
                attention_mask=torch.ones(B, seq),
            )
        finally:
            DummyLanguageModel.forward = orig
        # 4 visual (reduced from 8) + 8 text = 12
        assert seen["seq"] == 12
