"""Utilities."""
from typing import Iterable


def clip_float(low: float, high: float, x: float) -> float:
    """Clips a float to a range.

    Args:
      low: The lower bound of the range (inclusive).
      high: The upper bound of the range (inclusive).

    Returns:
      x clipped to the specified range.

    """
    return max(min(x, high), low)


def clip_iterable(low: float, high: float,
                  xs: Iterable[float]) -> Iterable[float]:
    """Clips an iterable to a range

    Args:
      low: The lower bound of the range (inclusive).
      high: The upper bound of the range (inclusive).
      xs: The iterable to clip.

    Returns:
      xs clipped to the specified range.

    """
    return [clip_float(low, high, x) for x in xs]
