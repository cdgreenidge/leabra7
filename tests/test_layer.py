"""Test layer.py"""
import math

from hypothesis import given
import hypothesis.strategies as st
import numpy as np
import pytest
import torch  # type: ignore

from leabra7 import layer as lr
from leabra7 import events as ev
from leabra7 import specs as sp


def test_parse_unit_attribute_strips_the_unit_prefix() -> None:
    attr = "unit_act"
    assert lr._parse_unit_attr(attr) == "act"
    with pytest.raises(ValueError):
        lr._parse_unit_attr("foobar")


def test_layer_init_uses_the_spec_you_pass_it() -> None:
    spec = sp.LayerSpec()
    layer = lr.Layer(name="in", spec=spec, size=1)
    assert layer.spec is spec


def test_layer_has_a_name() -> None:
    layer = lr.Layer(name="in", size=1)
    assert layer.name == "in"


def test_layer_has_a_size() -> None:
    layer = lr.Layer(name="in", size=1)
    assert layer.size == 1


def test_layer_should_be_able_to_compute_its_average_activation() -> None:
    layer = lr.Layer(name="in", size=2)
    layer.units.act[0] = 0
    layer.units.act[1] = 1
    assert layer.avg_act == 0.5


def test_layer_should_be_able_to_compute_its_average_net_input() -> None:
    layer = lr.Layer(name="in", size=2)
    layer.units.net[0] = 0
    layer.units.net[1] = 1
    assert layer.avg_net == 0.5


@given(
    n=st.integers(min_value=0, max_value=10),
    d=st.integers(min_value=1, max_value=10))
def test_layer_can_add_input(n, d) -> None:
    layer = lr.Layer(name="in", size=d)
    wt_scales = np.random.uniform(low=0.0, size=(n, ))
    for i in range(n):
        layer.add_input(torch.Tensor((d)), wt_scales[i])
        assert math.isclose(
            layer.wt_scale_rel_sum, sum(wt_scales[0:i + 1]), abs_tol=1e-6)


def test_layer_should_be_able_to_update_its_units_net_input(mocker) -> None:
    layer = lr.Layer(name="in", size=3)
    layer.units = mocker.Mock()
    layer.update_net()
    layer.units.update_net.assert_called_once()


def test_layer_should_be_able_to_update_its_units_fffb_inhibition() -> None:
    layer_spec = sp.LayerSpec(inhibition_type="fffb")
    layer = lr.Layer(name="in", size=3, spec=layer_spec)
    layer.update_inhibition()


def test_layer_should_be_able_to_update_its_units_kwta_inhibition() -> None:
    layer_spec = sp.LayerSpec(inhibition_type="kwta")
    layer = lr.Layer(name="in", size=3, spec=layer_spec)
    layer.update_inhibition()


def test_layer_should_be_able_to_update_its_units_kwta_avg_inhibition(
) -> None:
    layer_spec = sp.LayerSpec(inhibition_type="kwta_avg")
    layer = lr.Layer(name="in", size=3, spec=layer_spec)
    layer.update_inhibition()


def test_layer_should_be_able_to_do_an_activation_cycle() -> None:
    layer = lr.Layer(name="in", size=3)
    layer.activation_cycle()


def test_layer_should_be_able_to_observe_whole_attributes() -> None:
    layer = lr.Layer(name="in", size=3)
    assert layer.observe_whole_attr("avg_act") == ("avg_act", 0.0)


def test_layer_should_be_able_to_observe_parts_attributes() -> None:
    layer = lr.Layer(name="in", size=3)
    assert layer.observe_parts_attr("unit_act") == {
        "unit": [0, 1, 2],
        "act": [0.0, 0.0, 0.0]
    }


def test_observing_invalid_parts_attribute_should_raise_error() -> None:
    layer = lr.Layer(name="in", size=3)
    with pytest.raises(ValueError):
        layer.observe_parts_attr("whales")


def test_layer_can_update_learning_averages_when_hard_clamped(mocker) -> None:
    layer = lr.Layer(name="layer1", size=3)
    mocker.spy(layer, "update_trial_learning_averages")
    mocker.spy(layer.units, "update_cycle_learning_averages")

    layer.hard_clamp([1.0])
    layer.activation_cycle()
    layer.handle(ev.EndPlusPhase())

    layer.units.update_cycle_learning_averages.assert_called_once()
    layer.update_trial_learning_averages.assert_called_once()


def test_layer_hard_clamping_should_change_the_unit_activations() -> None:
    layer = lr.Layer(name="in", size=4)
    layer.hard_clamp([0, 1])
    expected = [0, 1, 0, 1]
    for i in range(4):
        assert math.isclose(layer.units.act[i], expected[i], abs_tol=1e-6)


def test_layer_set_hard_clamp() -> None:
    layer = lr.Layer(name="in", size=3)
    layer.hard_clamp(act_ext=[0, 1])
    layer.activation_cycle()
    expected = [0, 1, 0]
    for i in range(3):
        assert math.isclose(layer.units.act[i], expected[i], abs_tol=1e-6)


def test_layer_can_unclamp() -> None:
    layer = lr.Layer(name="in", size=4)
    layer.hard_clamp([0, 1])
    layer.unclamp()
    assert not layer.clamped
    assert list(layer.units.act) == [0, 1, 0, 1]


def test_hard_clamp_event_hard_clamps_a_layer_if_the_names_match() -> None:
    clamp = ev.HardClamp(layer_name="lr1", acts=[0.7, 0.7])
    layer = lr.Layer("lr1", 3)
    layer.handle(clamp)
    assert layer.clamped
    assert all(layer.units.act == 0.7)


def test_hard_clamp_event_does_nothing_if_the_names_do_not_match() -> None:
    clamp = ev.HardClamp(layer_name="lr1", acts=[0.7, 0.7])
    layer = lr.Layer("WHALES", 3)
    layer.handle(clamp)
    assert not layer.clamped


def test_end_plus_phase_event_saves_activations() -> None:
    layer = lr.Layer("lr1", 3)
    layer.hard_clamp([1, 0, 1])
    layer.handle(ev.EndPlusPhase())
    assert (layer.acts_p == torch.Tensor([1, 0, 1])).all()
