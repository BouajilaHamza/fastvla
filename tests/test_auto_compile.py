"""Auto-enable torch.compile on Ada (sm_89) / Hopper (sm_90+), keep off on
Turing (sm_75) and Ampere (sm_80) where compile yields little or causes
issues with bitsandbytes."""

import pytest
from unittest.mock import patch

from fastvla.config import _auto_torch_compile


@pytest.mark.parametrize(
    "major,minor,expected",
    [
        (7, 5, False),   # T4 Turing
        (8, 0, False),   # A100 Ampere
        (8, 6, False),   # RTX 3090 Ampere
        (8, 9, True),    # L4 / RTX 4090 Ada
        (9, 0, True),    # H100 Hopper
        (10, 0, True),   # future
    ],
)
def test_auto_torch_compile_per_arch(major, minor, expected):
    with patch("torch.cuda.is_available", return_value=True), patch(
        "torch.cuda.get_device_capability", return_value=(major, minor)
    ):
        assert _auto_torch_compile() is expected


def test_auto_torch_compile_no_cuda():
    with patch("torch.cuda.is_available", return_value=False):
        assert _auto_torch_compile() is False
