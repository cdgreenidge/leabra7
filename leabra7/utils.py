"""Utilities."""


def clip_float(low: float, high: float, x: float) -> float:
    """Clips a float to a range.

    Args:
      low: The lower bound of the range (inclusive).
      high: The upper bound of the range (inclusive).

    Returns:
      x clipped to the specified range.

    """
    return max(min(x, high), low)
