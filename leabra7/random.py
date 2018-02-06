"""Random number distributions.

At the moment, this module is not thread-safe.

"""
import abc
import math
import random


class Distribution(metaclass=abc.ABCMeta):
    """Abstract base class for all random number distributions."""

    @abc.abstractmethod
    def draw(self) -> float:
        """Draws a random number from the distribution."""


class Scalar(Distribution):
    """A scalar distribution (i.e. a constant).

    Args:
        value: The value of the scalar.

    """

    def __init__(self, value: float) -> None:
        self.value = value

    def draw(self) -> float:
        """Overrides `Distribution.draw()`."""
        return self.value


class Uniform(Distribution):
    """A uniform distribution.

    Args:
        low: The lower bound of the distribution.
        high: The upper bound of the distribution.

    Raises:
        ValueError: If low >= high.

    """

    def __init__(self, low: float, high: float) -> None:
        if low >= high:
            raise ValueError("low ({0}) must be less than high "
                             "({1}).".format(low, high))
        self.low = low
        self.high = high

    def draw(self) -> float:
        """Overrides `Distribution.draw()`."""
        return random.uniform(self.low, self.high)


class Gaussian(Distribution):
    """A gaussian distribution.

    Args:
        mean: The mean of the distribution (first moment).
        var: The variance of the distribution (second moment).

    Raises:
        ValueError: If var is negative.

    """

    def __init__(self, mean: float, var: float) -> None:
        if var < 0:
            raise ValueError("Variance ({0}) must be positive.".format(var))
        self.mu = mean
        self.sigma = math.sqrt(var)

    def draw(self) -> float:
        """Overrides `Distribution.draw()`."""
        return random.gauss(self.mu, self.sigma)


class LogNormal(Distribution):
    """A log normal distribution.

    Args:
        mean: The mean of the distribution's natural logarithm.
        var: The variance of the distribution's natural logarithm.

    Raises:
        ValueError: If var is negative.

    """

    def __init__(self, mean: float, var: float) -> None:
        if var < 0:
            raise ValueError("Variance ({0}) must be positive.".format(var))
        self.mu = mean
        self.sigma = math.sqrt(var)

    def draw(self) -> float:
        """Overrides `Distribution.draw()`."""
        return random.lognormvariate(self.mu, self.sigma)


class Exponential(Distribution):
    """An exponential distribution.

    Args:
        lambd: The rate parameter (1.0 / mean)

    """

    def __init__(self, lambd: float) -> None:
        self.lambd = lambd

    def draw(self) -> float:
        """Overrides `Distribution.draw()`."""
        return random.expovariate(self.lambd)
