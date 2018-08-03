"""Utilities."""

import torch  # type: ignore


def clip_float(low: float, high: float, x: float) -> float:
    """Clips a float to a range.

    Args:
      low: The lower bound of the range (inclusive).
      high: The upper bound of the range (inclusive).

    Returns:
      x clipped to the specified range.

    """
    return max(min(x, high), low)


def using_cuda() -> bool:
    return torch.cuda.is_available()


def to_cuda(tensor: torch.Tensor) -> torch.Tensor:
    """Converts torch tensor to cuda tensor if available.

    Args:
      tensor: Tensor to be converted to cuda tensor (if possible).

    Returns:
      Cuda-fied tensor or just same tensor as before.

    """
    return tensor.cuda() if using_cuda() else tensor
