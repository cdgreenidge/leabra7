"""Test specs.py"""
from leabra7 import specs as sp


def test_spec_sets_attributes_from_the_constructor():
    spec = sp.Spec(a=3)
    assert spec.a == 3
