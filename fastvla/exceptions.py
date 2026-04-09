"""
Custom exceptions for FastVLA.
Provides structured error handling for environment and model loading issues.
"""

class FastVLAError(Exception):
    """Base exception for all FastVLA errors."""
    pass

class EnvironmentCompatibilityError(FastVLAError):
    """Raised when the hardware or software environment is incompatible."""
    pass

class ModelLoadingError(FastVLAError):
    """Raised when a model or weights fail to load."""
    pass

class DistributedTrainingError(FastVLAError):
    """Raised when there are issues with multi-GPU/Distributed training."""
    pass

class QuantizationError(FastVLAError):
    """Raised when 4-bit or 8-bit quantization fails or is misused."""
    pass
