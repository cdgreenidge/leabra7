"""Test utils.py"""
from leabra7 import utils as ut


def test_clip_float_returns_x_if_it_is_within_the_range() -> None:
    assert ut.clip_float(low=0.2, high=0.5, x=0.3) == 0.3


def test_clip_float_returns_the_lower_bound_if_x_is_below_it() -> None:
    assert ut.clip_float(low=0.2, high=0.5, x=-1) == 0.2


def test_clip_float_returns_the_upper_bound_if_x_is_above_it() -> None:
    assert ut.clip_float(low=0.2, high=0.5, x=1) == 0.5
