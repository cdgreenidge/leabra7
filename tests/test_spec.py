"""Test specs.py"""
import pytest

from leabra7 import specs as sp


class Foo(sp.Spec):
    a = 3


def test_spec_sets_attributes_from_the_constructor():
    spec = Foo(a=4)
    assert spec.a == 4


def test_spec_throws_an_error_if_you_set_a_nonexistent_attribute():
    with pytest.raises(AttributeError):
        Foo(b=4)


def test_spec_supports_equality_comparison():
    a = Foo()
    b = Foo()
    assert a == b
