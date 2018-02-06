"""Test random.py"""
import pytest

from leabra7 import random as rn


def test_distribution_cannot_be_instantiated():
    with pytest.raises(TypeError):
        rn.Distribution()


def test_scalar_is_always_equal_to_its_value():
    dist = rn.Scalar(3)
    for i in range(50):
        assert dist.draw() == 3


def test_uniform_is_always_within_its_range():
    dist = rn.Uniform(2, 3)
    for i in range(50):
        assert 2 <= dist.draw() <= 3


def test_uniform_checks_low_is_less_than_high():
    with pytest.raises(ValueError):
        rn.Uniform(3, 2)


def test_gaussian_can_be_used():
    dist = rn.Gaussian(2, 3)
    for i in range(50):
        dist.draw()


def test_gaussian_checks_variance_is_positive():
    with pytest.raises(ValueError):
        rn.Gaussian(mean=0, var=-1)


def test_lognormal_is_always_postiive():
    dist = rn.LogNormal(2, 3)
    for i in range(50):
        assert dist.draw() >= 0


def test_lognormal_checks_variance_is_positive():
    with pytest.raises(ValueError):
        rn.LogNormal(2, -3)


def test_exponential_can_be_used():
    dist = rn.Exponential(3)
    for i in range(50):
        dist.draw()
