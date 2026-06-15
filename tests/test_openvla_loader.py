"""OpenVLA vision-tower load path must keep the fused DINOv2+SigLIP backbone,
not silently fall back to SigLIP-only."""

import os
import pytest
import torch

from fastvla.adapters.vision import (
    OpenVLAFusedVisionAdapter,
    SigLIPVisionAdapter,
)


@pytest.mark.skipif(
    not os.environ.get("HF_TOKEN") and not os.environ.get("HF_API_KEY"),
    reason="needs HF token for openvla/openvla-7b download",
)
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="OpenVLA load path uses bitsandbytes / device_map='auto'",
)
def test_openvla_loader_keeps_fused_backbone():
    """Loading openvla/openvla-7b must not return a plain SigLIPVisionAdapter.

    If the adapter is SigLIPVisionAdapter, the loader silently fell back
    (current bug — `transformers.AutoModel` does not recognise
    `OpenVLAConfig`). The fused DINOv2 + SigLIP backbone must be retained.
    """
    adapter = OpenVLAFusedVisionAdapter.from_pretrained(
        "openvla/openvla-7b",
        device_map="auto",
        load_in_4bit=True,
        hf_token=os.environ.get("HF_TOKEN") or os.environ.get("HF_API_KEY"),
    )
    assert not isinstance(adapter, SigLIPVisionAdapter), (
        "OpenVLAFusedVisionAdapter fell back to SigLIP-only — "
        "AutoModelForVision2Seq path failed."
    )
    # OpenVLA's fused tower exposes 1024 (DINOv2-L), 1152 (SigLIP-SO400M),
    # or 2176 (concat) depending on whether the wrapped class projects.
    assert adapter.embed_dim in (1024, 1152, 2176), (
        f"unexpected embed_dim {adapter.embed_dim}"
    )
