"""Tests events.py"""
import pytest

from leabra7 import events as ev


def test_clamp_checks_if_acts_contains_values_outside_0_1() -> None:
    with pytest.raises(ValueError):
        ev.Clamp(layer_name="lr1", acts=(1, 2), hard=True)
