"""Test specs.py"""
from hypothesis import example
from hypothesis import given
import hypothesis.strategies as st
import pytest

from leabra7 import events as ev
from leabra7 import layer as lr
from leabra7 import specs as sp


class Foo(sp.Spec):
    """Dummy spec for testing."""
    a = 3

    def validate(self) -> None:
        """Validates the spec."""
        pass


def test_spec_sets_attributes_from_the_constructor() -> None:
    spec = Foo(a=4)
    assert spec.a == 4


def test_spec_throws_an_error_if_you_set_a_nonexistent_attribute() -> None:
    with pytest.raises(ValueError):
        Foo(b=4)


def test_spec_supports_equality_comparison() -> None:
    a = Foo()
    b = Foo()
    assert a == b
    a.a = 4
    assert a != b


def test_spec_can_check_if_an_attribute_is_in_range() -> None:
    a = Foo()
    a.assert_in_range("a", low=0, high=4)
    with pytest.raises(sp.ValidationError):
        a.assert_in_range("a", low=0, high=1)


def test_spec_assert_in_range_checks_the_bounds_make_sense() -> None:
    a = Foo()
    with pytest.raises(ValueError):
        a.assert_in_range("a", low=3, high=2)


def test_spec_can_check_if_an_attribute_is_nan() -> None:
    a = Foo()
    a.assert_sane_float("a")
    a.a = float("nan")
    with pytest.raises(sp.ValidationError):
        a.assert_sane_float("a")


def test_spec_can_check_if_an_attribute_is_inf() -> None:
    a = Foo()
    a.assert_sane_float("a")
    a.a = float("Inf")
    with pytest.raises(sp.ValidationError):
        a.assert_sane_float("a")


def test_spec_can_check_if_an_attribute_is_negative_inf() -> None:
    a = Foo()
    a.assert_sane_float("a")
    a.a = float("-Inf")
    with pytest.raises(sp.ValidationError):
        a.assert_sane_float("a")


def float_outside_range(low=None, high=None) -> None:
    """A Hypothesis strategy that generates a float outside [low, high]."""
    return st.one_of(
        st.floats(max_value=low),
        st.floats(min_value=high)).filter(lambda x: x != low and x != high)


insane_float = st.one_of(
    st.just(float("NaN")), st.just(float("Inf")), st.just(float("-Inf")))
"""A Hypothesis strategy that generates one of NaN, +Inf, or -Inf."""


# Test UnitSpec validation
@given(float_outside_range(0, float("Inf")))
def test_unit_spec_validates_integ(f) -> None:
    with pytest.raises(sp.ValidationError):
        sp.UnitSpec(integ=f).validate()


@given(insane_float)
def test_it_should_validate_e_rev_e_for_insane_floats(f) -> None:
    with pytest.raises(sp.ValidationError):
        sp.UnitSpec(e_rev_e=f).validate()


@given(insane_float)
def test_it_should_validate_e_rev_i_for_insane_floats(f) -> None:
    with pytest.raises(sp.ValidationError):
        sp.UnitSpec(e_rev_i=f).validate()


@given(insane_float)
def test_it_should_validate_e_rev_l_for_insane_floats(f) -> None:
    with pytest.raises(sp.ValidationError):
        sp.UnitSpec(e_rev_l=f).validate()


@given(insane_float)
def test_it_should_validate_gc_l_for_insane_floats(f) -> None:
    with pytest.raises(sp.ValidationError):
        sp.UnitSpec(gc_l=f).validate()


@given(insane_float)
def test_it_should_validate_spk_thr_for_insane_floats(f) -> None:
    with pytest.raises(sp.ValidationError):
        sp.UnitSpec(spk_thr=f).validate()


@given(insane_float)
def test_it_should_validate_v_m_r_for_insane_floats(f) -> None:
    with pytest.raises(sp.ValidationError):
        sp.UnitSpec(v_m_r=f).validate()


def test_it_should_validate_v_m_r_is_less_than_spk_thr() -> None:
    with pytest.raises(sp.ValidationError):
        sp.UnitSpec(v_m_r=1, spk_thr=0.5).validate()


@given(insane_float)
def test_it_should_validate_spike_gain_for_insane_floats(f) -> None:
    with pytest.raises(sp.ValidationError):
        sp.UnitSpec(spike_gain=f).validate()


@given(float_outside_range(0, float("Inf")))
def test_it_should_validate_net_dt(f) -> None:
    with pytest.raises(sp.ValidationError):
        sp.UnitSpec(net_dt=f).validate()


@given(float_outside_range(0, float("Inf")))
def test_it_should_validate_vm_dt(f) -> None:
    with pytest.raises(sp.ValidationError):
        sp.UnitSpec(vm_dt=f).validate()


@given(float_outside_range(0, float("Inf")))
def test_it_should_validate_adapt_dt(f) -> None:
    with pytest.raises(sp.ValidationError):
        sp.UnitSpec(adapt_dt=f).validate()


@given(float_outside_range(0, 1.0))
def test_it_should_validate_syn_tr(f) -> None:
    with pytest.raises(sp.ValidationError):
        sp.UnitSpec(syn_tr=f).validate()


@given(float_outside_range(0, 1.0))
def test_it_should_validate_act_min(f) -> None:
    with pytest.raises(sp.ValidationError):
        sp.UnitSpec(act_min=f).validate()


@given(float_outside_range(0, 1.0))
def test_it_should_validate_act_max(f) -> None:
    with pytest.raises(sp.ValidationError):
        sp.UnitSpec(act_max=f).validate()


@given(insane_float)
def test_it_should_validate_vm_min_sanity(f) -> None:
    with pytest.raises(sp.ValidationError):
        sp.UnitSpec(vm_min=f).validate()


@given(insane_float)
def test_it_should_validate_vm_max_sanity(f) -> None:
    with pytest.raises(sp.ValidationError):
        sp.UnitSpec(vm_max=f).validate()


@given(float_outside_range(0, float("Inf")))
def test_it_should_validate_act_gain_positive(f) -> None:
    with pytest.raises(sp.ValidationError):
        sp.UnitSpec(act_gain=f).validate()


def test_it_should_validate_act_min_is_less_than_act_max() -> None:
    with pytest.raises(sp.ValidationError):
        sp.UnitSpec(act_min=1.0, act_max=0.5).validate()


def test_it_should_validate_vm_min_is_less_than_vm_max() -> None:
    with pytest.raises(sp.ValidationError):
        sp.UnitSpec(vm_min=1.0, vm_max=0.5).validate()


def test_it_should_validate_v_m_r_in_range() -> None:
    with pytest.raises(sp.ValidationError):
        sp.UnitSpec(vm_min=1.0, vm_max=1.5, v_m_r=0.3).validate()
    with pytest.raises(sp.ValidationError):
        sp.UnitSpec(vm_min=0.1, vm_max=0.15, v_m_r=0.3).validate()


@given(float_outside_range(0, float("Inf")))
def test_it_should_validate_ss_dt(f) -> None:
    with pytest.raises(sp.ValidationError):
        sp.UnitSpec(ss_dt=f).validate()


@given(float_outside_range(0, float("Inf")))
def test_it_should_validate_s_dt(f) -> None:
    with pytest.raises(sp.ValidationError):
        sp.UnitSpec(s_dt=f).validate()


@given(float_outside_range(0, float("Inf")))
def test_it_should_validate_m_dt(f) -> None:
    with pytest.raises(sp.ValidationError):
        sp.UnitSpec(m_dt=f).validate()


@given(float_outside_range(0, float("Inf")))
def test_it_should_validate_l_dn_dt(f) -> None:
    with pytest.raises(sp.ValidationError):
        sp.UnitSpec(l_dn_dt=f).validate()


@given(float_outside_range(0, float("Inf")))
def test_it_should_validate_l_up_inc(f) -> None:
    with pytest.raises(sp.ValidationError):
        sp.UnitSpec(l_up_inc=f).validate()


# Test LayerSpec validation
@given(st.text())
@example("kwta")
@example("kwta_avg")
@example("fffb")
@example("none")
def test_layer_spec_validates_inhibition_type(f) -> None:
    if f not in ["kwta", "kwta_avg", "fffb", "none"]:
        with pytest.raises(sp.ValidationError):
            sp.LayerSpec(inhibition_type=f).validate()
    else:
        sp.LayerSpec(inhibition_type=f).validate()


@given(float_outside_range(0, 1.0))
def test_layer_spec_validates_kwta_pct(f) -> None:
    with pytest.raises(sp.ValidationError):
        sp.LayerSpec(kwta_pct=f).validate()


@given(float_outside_range(0, 1.0))
def test_layer_spec_validates_kwta_pt(f) -> None:
    with pytest.raises(sp.ValidationError):
        sp.LayerSpec(kwta_pt=f).validate()


@given(float_outside_range(0, float("Inf")))
def test_layer_spec_validates_integ(f) -> None:
    with pytest.raises(sp.ValidationError):
        sp.LayerSpec(integ=f).validate()


@given(insane_float)
def test_it_should_validate_feedforward_inhibition_multiplier(f) -> None:
    with pytest.raises(sp.ValidationError):
        sp.LayerSpec(ff=f).validate()


@given(insane_float)
def test_it_should_validate_feedback_inhibition_multiplier(f) -> None:
    with pytest.raises(sp.ValidationError):
        sp.LayerSpec(fb=f).validate()


@given(float_outside_range(0, float("Inf")))
def test_it_should_validate_feedback_integration_constant(f) -> None:
    with pytest.raises(sp.ValidationError):
        sp.LayerSpec(fb_dt=f).validate()


@given(insane_float)
def test_it_should_validate_global_inhibition_multiplier(f) -> None:
    with pytest.raises(sp.ValidationError):
        sp.LayerSpec(gi=f).validate()


def test_it_should_validate_the_unit_spec() -> None:
    with pytest.raises(sp.ValidationError):
        sp.LayerSpec(unit_spec=sp.UnitSpec(vm_dt=-1)).validate()


def test_every_valid_log_on_cycle_attribute_can_be_logged() -> None:
    valid_attrs = sp.LayerSpec()._valid_attrs_to_log
    lr1 = lr.Layer("lr1", 3)
    for attr in valid_attrs:
        lr1.validate_attr(attr)


def test_it_should_check_for_invalid_log_on_cycle_attrs() -> None:
    with pytest.raises(sp.ValidationError):
        sp.LayerSpec(log_on_cycle=("whales", )).validate()


def test_it_should_check_for_invalid_log_on_trial_attrs() -> None:
    with pytest.raises(sp.ValidationError):
        sp.LayerSpec(log_on_trial=("whales", )).validate()


def test_it_should_check_for_invalid_log_on_epoch_attrs() -> None:
    with pytest.raises(sp.ValidationError):
        sp.LayerSpec(log_on_epoch=("whales", )).validate()


def test_it_should_check_for_invalid_log_on_batch_attrs() -> None:
    with pytest.raises(sp.ValidationError):
        sp.LayerSpec(log_on_batch=("whales", )).validate()


# Test ProjnSpec validation
@given(float_outside_range(0, float("Inf")))
def test_projn_spec_validates_integ(f) -> None:
    with pytest.raises(sp.ValidationError):
        sp.ProjnSpec(integ=f).validate()


def test_projn_spec_validates_the_distribution() -> None:
    with pytest.raises(sp.ValidationError):
        sp.ProjnSpec(dist=3).validate()


@given(float_outside_range(0, 1))
def test_projn_spec_validates_sparsity(f) -> None:
    with pytest.raises(sp.ValidationError):
        sp.ProjnSpec(sparsity=f).validate()


@given(st.text())
@example("one_to_one")
@example("none")
def test_projn_spec_validates_projn_type(f) -> None:
    if f not in ["one_to_one", "full"]:
        with pytest.raises(sp.ValidationError):
            sp.ProjnSpec(projn_type=f).validate()
    else:
        sp.ProjnSpec(projn_type=f).validate()


@given(float_outside_range(0, float("Inf")))
def test_projn_spec_validates_wt_scale_abs(f) -> None:
    with pytest.raises(sp.ValidationError):
        sp.ProjnSpec(wt_scale_abs=f).validate()


@given(float_outside_range(0, float("Inf")))
def test_projn_spec_validates_wt_scale_rel(f) -> None:
    with pytest.raises(sp.ValidationError):
        sp.ProjnSpec(wt_scale_rel=f).validate()


@given(float_outside_range(0, float("Inf")))
def test_projn_spec_validates_lrate(f) -> None:
    with pytest.raises(sp.ValidationError):
        sp.ProjnSpec(lrate=f).validate()


@given(float_outside_range(0, float("Inf")))
def test_projn_spec_validates_sig_gain(f) -> None:
    with pytest.raises(sp.ValidationError):
        sp.ProjnSpec(sig_gain=f).validate()


@given(insane_float)
def test_projn_spec_validates_sig_offset(f) -> None:
    with pytest.raises(sp.ValidationError):
        sp.ProjnSpec(sig_offset=f).validate()


def test_projn_spec_validates_minus_phase_type() -> None:
    PhaseNone = ev.Phase(name="phase_none", phase_type="none")
    sp.ProjnSpec(minus_phase=PhaseNone).validate()

    PhaseMinus = ev.Phase(name="phase_minus", phase_type="minus")
    sp.ProjnSpec(minus_phase=PhaseMinus).validate()

    PhasePlus = ev.Phase(name="phase_plus", phase_type="plus")
    with pytest.raises(sp.ValidationError):
        sp.ProjnSpec(minus_phase=PhasePlus, plus_phase=PhaseNone).validate()


def test_projn_spec_validates_plus_phase_type() -> None:
    PhaseNone = ev.Phase(name="phase_none", phase_type="none")
    sp.ProjnSpec(plus_phase=PhaseNone).validate()

    PhasePlus = ev.Phase(name="phase_plus", phase_type="plus")
    sp.ProjnSpec(plus_phase=PhasePlus).validate()

    PhaseMinus = ev.Phase(name="phase_minus", phase_type="minus")
    with pytest.raises(sp.ValidationError):
        sp.ProjnSpec(minus_phase=PhaseNone, plus_phase=PhaseMinus).validate()


def test_projn_spec_validates_different_plus_and_minus_phases() -> None:
    with pytest.raises(sp.ValidationError):
        sp.ProjnSpec(
            minus_phase=ev.NonePhase, plus_phase=ev.NonePhase).validate()


def test_projn_spec_validates_attrs_to_log() -> None:
    with pytest.raises(sp.ValidationError):
        sp.ProjnSpec(log_on_cycle=("whales", )).validate()


def test_attrs_to_log_gets_the_attrs_to_log_by_frequency() -> None:
    spec = sp.LayerSpec(
        log_on_cycle=("unit_act", ),
        log_on_trial=("unit_v_m", ),
        log_on_epoch=("unit_spike", ),
        log_on_batch=("unit_i_net", ))
    assert spec.attrs_to_log(ev.CycleFreq) == ("unit_act", )
    assert spec.attrs_to_log(ev.TrialFreq) == ("unit_v_m", )
    assert spec.attrs_to_log(ev.EpochFreq) == ("unit_spike", )
    assert spec.attrs_to_log(ev.BatchFreq) == ("unit_i_net", )
