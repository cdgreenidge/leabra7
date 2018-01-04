"""Test specs.py"""
import pytest

from leabra7 import specs as sp


class Foo(sp.Spec):
    a = 3

    def validate(self) -> None:
        pass


def test_spec_sets_attributes_from_the_constructor():
    spec = Foo(a=4)
    assert spec.a == 4


def test_spec_throws_an_error_if_you_set_a_nonexistent_attribute():
    with pytest.raises(ValueError):
        Foo(b=4)


def test_spec_supports_equality_comparison():
    a = Foo()
    b = Foo()
    assert a == b
    a.a = 4
    assert a != b


def test_spec_can_check_if_an_attribute_is_in_range():
    a = Foo()
    a.assert_in_range("a", low=0, high=4)
    with pytest.raises(sp.ValidationError):
        a.assert_in_range("a", low=0, high=1)


def test_spec_assert_in_range_checks_the_bounds_make_sense():
    a = Foo()
    with pytest.raises(ValueError):
        a.assert_in_range("a", low=3, high=2)


def test_spec_can_check_if_an_attribute_is_nan():
    a = Foo()
    a.assert_not_nan("a")
    a.a = float("nan")
    with pytest.raises(sp.ValidationError):
        a.assert_not_nan("a")
