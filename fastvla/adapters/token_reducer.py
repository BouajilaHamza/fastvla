"""
Visual Token Reducers — Roadmap Thrust A / Phase 1.1 (Hypothesis H1)
====================================================================

The single highest-payoff VLA-specific compute lever. On a 7B-class VLA the LLM
backbone is ~85-90% of compute, and its cost scales with **sequence length** —
which, for a VLA, is dominated by image tokens (OpenVLA feeds ~256 image tokens
vs ~20 text tokens). Cutting the visual token count therefore shrinks the
expensive trunk directly, and it is the VLA-native analogue of sequence packing
in LLMs.

Each reducer maps a visual token sequence ``[B, N, D] -> [B, k, D]`` where
``k = num_tokens`` is the budget. They are drop-in and dimension-preserving so
they slot in before the vision->language projection without touching the rest of
the pipeline.

Strategies (ablation axis for H1):
  - ``mean_pool``      parameter-free adaptive average pooling over the token axis.
  - ``attention_pool`` ``k`` learnable query tokens attend over the N inputs.
  - ``perceiver``      attention pooling + a feed-forward block (Perceiver-style).
  - ``token_merge``    ToMe-style parameter-free bipartite soft matching.

``get_token_reducer`` is the factory used by the model.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseTokenReducer(nn.Module):
    """Common interface: ``[B, N, D] -> [B, num_tokens, D]``."""

    def __init__(self, num_tokens: int, dim: int):
        super().__init__()
        self.num_tokens = num_tokens
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    @staticmethod
    def _passthrough_if_smaller(x: torch.Tensor, k: int) -> Optional[torch.Tensor]:
        """If there is nothing to reduce (``N <= k``) return the input as-is."""
        return x if x.shape[1] <= k else None


class MeanPoolReducer(BaseTokenReducer):
    """Parameter-free adaptive average pooling over the token dimension.

    A strong, zero-overhead baseline: it makes no learned choice about *which*
    tokens matter, only that neighbouring tokens can be averaged. Useful as the
    control in the H1 ablation.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        passthrough = self._passthrough_if_smaller(x, self.num_tokens)
        if passthrough is not None:
            return passthrough
        # [B, N, D] -> [B, D, N] -> pool -> [B, D, k] -> [B, k, D]
        pooled = F.adaptive_avg_pool1d(x.transpose(1, 2), self.num_tokens)
        return pooled.transpose(1, 2).contiguous()


class AttentionPoolReducer(BaseTokenReducer):
    """``num_tokens`` learnable query vectors attend over the input tokens.

    Unlike mean pooling this *learns* which visual content to keep, at the cost
    of a small number of parameters (the queries + one attention block).
    """

    def __init__(self, num_tokens: int, dim: int, num_heads: int = 8):
        super().__init__(num_tokens, dim)
        # Ensure the head count divides the embedding dim.
        while dim % num_heads != 0 and num_heads > 1:
            num_heads -= 1
        self.query = nn.Parameter(torch.randn(1, num_tokens, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        q = self.query.expand(b, -1, -1).to(x.dtype)
        out, _ = self.attn(q, x, x, need_weights=False)
        return self.norm(out)


class PerceiverResampler(BaseTokenReducer):
    """Attention pooling followed by a feed-forward block (Perceiver-IO style).

    The extra FFN gives the resampler capacity to reorganise the pooled content,
    which tends to matter more as the budget gets aggressive (e.g. 32 tokens).
    """

    def __init__(self, num_tokens: int, dim: int, num_heads: int = 8, mlp_ratio: int = 4):
        super().__init__(num_tokens, dim)
        while dim % num_heads != 0 and num_heads > 1:
            num_heads -= 1
        self.latents = nn.Parameter(torch.randn(1, num_tokens, dim) * 0.02)
        self.attn_norm_q = nn.LayerNorm(dim)
        self.attn_norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        q = self.attn_norm_q(self.latents.expand(b, -1, -1).to(x.dtype))
        kv = self.attn_norm_kv(x)
        attended, _ = self.attn(q, kv, kv, need_weights=False)
        latents = q + attended
        return latents + self.ff(self.ff_norm(latents))


class TokenMergeReducer(BaseTokenReducer):
    """ToMe-style parameter-free bipartite soft matching.

    Repeatedly split the tokens into two alternating sets, match each token in
    the source set to its most similar token in the destination set (by cosine
    similarity), and merge the most-similar pairs by averaging. Runs in rounds
    (each halving at most) until the budget is reached, then a final adaptive
    pool guarantees exactly ``num_tokens`` outputs.

    Parameter-free and content-aware — merges *redundant* tokens rather than
    arbitrary neighbours, which is why it usually beats mean pooling at equal k.
    """

    def _bipartite_merge_round(self, x: torch.Tensor, target: int) -> torch.Tensor:
        b, n, d = x.shape
        if n <= target:
            return x
        # Alternating split: dst = even indices, src = odd indices.
        dst = x[:, 0::2, :]
        src = x[:, 1::2, :]
        n_src = src.shape[1]

        # Cosine similarity of each src token to each dst token.
        src_n = F.normalize(src, dim=-1)
        dst_n = F.normalize(dst, dim=-1)
        sim = torch.bmm(src_n, dst_n.transpose(1, 2))  # [B, n_src, n_dst]
        best_val, best_idx = sim.max(dim=-1)           # [B, n_src]

        # Merge as many src tokens as we can afford this round, choosing the
        # most-confident (highest-similarity) matches first.
        r = min(n_src, n - target)
        if r <= 0:
            return x
        merge_order = best_val.argsort(dim=-1, descending=True)  # [B, n_src]
        merge_mask = torch.zeros(b, n_src, dtype=torch.bool, device=x.device)
        merge_mask.scatter_(1, merge_order[:, :r], True)

        out = []
        for i in range(b):
            dst_i = dst[i].clone()                     # [n_dst, D]
            counts = torch.ones(dst_i.shape[0], 1, device=x.device, dtype=x.dtype)
            keep_src = []
            for j in range(n_src):
                if merge_mask[i, j]:
                    t = best_idx[i, j]
                    dst_i[t] = dst_i[t] + src[i, j]
                    counts[t] += 1
                else:
                    keep_src.append(src[i, j])
            dst_i = dst_i / counts
            merged = torch.cat([dst_i] + ([torch.stack(keep_src)] if keep_src else []), dim=0)
            out.append(merged)
        # Rows may differ in length only if target isn't hit; pad by trimming to min.
        min_len = min(o.shape[0] for o in out)
        return torch.stack([o[:min_len] for o in out], dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        passthrough = self._passthrough_if_smaller(x, self.num_tokens)
        if passthrough is not None:
            return passthrough
        # Merge in rounds until we are within one adaptive-pool step of the budget.
        guard = 0
        while x.shape[1] > self.num_tokens and guard < 16:
            x = self._bipartite_merge_round(x, self.num_tokens)
            guard += 1
        if x.shape[1] != self.num_tokens:
            x = F.adaptive_avg_pool1d(x.transpose(1, 2), self.num_tokens).transpose(1, 2)
        return x.contiguous()


_REDUCERS = {
    "mean_pool": MeanPoolReducer,
    "attention_pool": AttentionPoolReducer,
    "perceiver": PerceiverResampler,
    "token_merge": TokenMergeReducer,
}


def get_token_reducer(
    strategy: str,
    num_tokens: int,
    dim: int,
    num_heads: int = 8,
) -> BaseTokenReducer:
    """Factory for visual token reducers.

    Args:
        strategy: one of ``mean_pool``, ``attention_pool``, ``perceiver``, ``token_merge``.
        num_tokens: the visual token budget ``k``.
        dim: the visual embedding dimension ``D``.
        num_heads: attention heads for the parametric reducers.
    """
    if strategy not in _REDUCERS:
        raise ValueError(
            f"Unknown token_reduction_strategy '{strategy}'. "
            f"Choose from {sorted(_REDUCERS)}."
        )
    cls = _REDUCERS[strategy]
    if strategy in ("attention_pool", "perceiver"):
        return cls(num_tokens, dim, num_heads=num_heads)
    return cls(num_tokens, dim)
