"""Test specs.py"""
from hypothesis import given
import hypothesis.strategies as st
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
    a.assert_sane_float("a")
    a.a = float("nan")
    with pytest.raises(sp.ValidationError):
        a.assert_sane_float("a")


def test_spec_can_check_if_an_attribute_is_inf():
    a = Foo()
    a.assert_sane_float("a")
    a.a = float("Inf")
    with pytest.raises(sp.ValidationError):
        a.assert_sane_float("a")


def test_spec_can_check_if_an_attribute_is_negative_inf():
    a = Foo()
    a.assert_sane_float("a")
    a.a = float("-Inf")
    with pytest.raises(sp.ValidationError):
        a.assert_sane_float("a")


def float_outside_range(low=None, high=None):
    """A Hypothesis strategy that generates a float outside [low, high]."""
    return st.one_of(
        st.floats(max_value=low),
        st.floats(min_value=high)).filter(lambda x: x != low and x != high)


insane_float = st.one_of(
    st.just(float("NaN")), st.just(float("Inf")), st.just(float("-Inf")))
"""A Hypothesis strategy that generates one of NaN, +Inf, or -Inf."""


@given(float_outside_range(0, float("Inf")))
def test_spec_validates_integ(f):
    with pytest.raises(sp.ValidationError):
        sp.UnitSpec(integ=f).validate()


# Test UnitSpec validation
@given(insane_float)
def test_it_should_validate_e_rev_e_for_insane_floats(f):
    with pytest.raises(sp.ValidationError):
        sp.UnitSpec(e_rev_e=f).validate()


@given(insane_float)
def test_it_should_validate_e_rev_i_for_insane_floats(f):
    with pytest.raises(sp.ValidationError):
        sp.UnitSpec(e_rev_i=f).validate()


@given(insane_float)
def test_it_should_validate_e_rev_l_for_insane_floats(f):
    with pytest.raises(sp.ValidationError):
        sp.UnitSpec(e_rev_l=f).validate()


@given(insane_float)
def test_it_should_validate_gc_l_for_insane_floats(f):
    with pytest.raises(sp.ValidationError):
        sp.UnitSpec(gc_l=f).validate()


@given(insane_float)
def test_it_should_validate_spk_thr_for_insane_floats(f):
    with pytest.raises(sp.ValidationError):
        sp.UnitSpec(spk_thr=f).validate()


@given(insane_float)
def test_it_should_validate_v_m_r_for_insane_floats(f):
    with pytest.raises(sp.ValidationError):
        sp.UnitSpec(v_m_r=f).validate()


def test_it_should_validate_v_m_r_is_less_than_spk_thr():
    with pytest.raises(sp.ValidationError):
        sp.UnitSpec(v_m_r=1, spk_thr=0.5).validate()


@given(insane_float)
def test_it_should_validate_spike_gain_for_insane_floats(f):
    with pytest.raises(sp.ValidationError):
        sp.UnitSpec(spike_gain=f).validate()


@given(float_outside_range(0, float("Inf")))
def test_it_should_validate_net_dt(f):
    with pytest.raises(sp.ValidationError):
        sp.UnitSpec(net_dt=f).validate()


@given(float_outside_range(0, float("Inf")))
def test_it_should_validate_vm_dt(f):
    with pytest.raises(sp.ValidationError):
        sp.UnitSpec(vm_dt=f).validate()


@given(float_outside_range(0, float("Inf")))
def test_it_should_validate_adapt_dt(f):
    with pytest.raises(sp.ValidationError):
        sp.UnitSpec(adapt_dt=f).validate()
