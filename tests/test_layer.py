"""Test layer.py"""
import pytest

from leabra7 import layer as lr
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


def test_layer_should_be_able_to_update_its_units_net_input(mocker) -> None:
    layer = lr.Layer(name="in", size=3)
    layer.units = mocker.Mock()
    layer.update_net()
    layer.units.update_net.assert_called_once()


def test_layer_should_be_able_to_update_its_units_inhibition() -> None:
    layer = lr.Layer(name="in", size=3)
    layer.update_inhibition()


def test_layer_should_be_able_to_update_its_units_activation() -> None:
    layer = lr.Layer(name="in", size=3)
    layer.update_activation()


def test_layer_should_be_able_to_do_an_activation_cycle() -> None:
    layer = lr.Layer(name="in", size=3)
    layer.activation_cycle()


def test_layer_should_be_able_to_observe_whole_attributes() -> None:
    layer = lr.Layer(name="in", size=3)
    assert layer.observe_whole_attr("avg_act") == ("avg_act", 0.0)


def test_layer_shuld_be_able_to_observe_parts_attributes() -> None:
    layer = lr.Layer(name="in", size=3)
    assert layer.observe_parts_attr("unit_act") == {
        "unit": [0, 1, 2],
        "act": [0.0, 0.0, 0.0]
    }


def test_observing_invalid_parts_attribute_should_raise_error() -> None:
    layer = lr.Layer(name="in", size=3)
    with pytest.raises(ValueError):
        layer.observe_parts_attr("whales")


def test_layer_forcing_should_change_the_unit_activations() -> None:
    layer = lr.Layer(name="in", size=4)
    layer.force([0, 1])
    assert list(layer.units.act) == [0, 1, 0, 1]


def test_layer_forcing_should_not_change_after_cycles() -> None:
    layer = lr.Layer(name="in", size=4)
    layer.force([0, 1])
    layer.activation_cycle()
    assert list(layer.units.act) == [0, 1, 0, 1]
