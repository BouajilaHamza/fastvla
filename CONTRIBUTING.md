# Contributing to FastVLA

Thank you for your interest in FastVLA! We welcome contributions that help make high-performance VLA finetuning more accessible on commodity hardware.

## Development Principles
1.  **Efficiency First**: All kernels and loaders must be optimized for NVIDIA L4 hardware (16GB VRAM).
2.  **Test-Driven Development (TDD)**: Every new feature or bug fix must include a corresponding test in the `tests/` directory.
3.  **Distributed Safety**: Ensure all changes are compatible with Hugging Face `Accelerator` and multi-GPU environments.

## How to Contribute
1.  **Fork the Repository**: Create a personal fork of the project.
2.  **Create a Branch**: Use descriptive branch names (e.g., `feat/new-fusion-kernel` or `fix/dtype-mismatch`).
3.  **Implement & Test**: Ensure all tests pass using `uv run pytest`.
4.  **Submit a PR**: Provide a clear description of your changes and why they are necessary.

## Coding Standards
-   **Typing**: Use type hints for all function signatures.
-   **Kernels**: Triton kernels should include CPU fallbacks for robustness.
-   **Logging**: Use the standard `logging` module; avoid print statements in library code.

## Production Scripts
The `finetune_on_modal.py` script is considered a production entry point. If you modify core model logic, ensure this script is updated and verified on Modal L4.
