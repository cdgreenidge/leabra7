"""Test layer.py"""
from leabra7 import layer as lr
from leabra7 import specs as sp
from leabra7 import unit as un


def test_layer_init_uses_the_spec_you_pass_it():
    spec = sp.LayerSpec()
    layer = lr.Layer(name="in", spec=spec, size=1)
    assert layer.spec is spec


def test_layer_has_a_name():
    layer = lr.Layer(name="in", size=1)
    assert layer.name == "in"


def test_layer_has_a_size():
    layer = lr.Layer(name="in", size=1)
    assert layer.size == 1


def test_layer_has_a_list_of_units():
    layer = lr.Layer(name="in", size=3)
    assert len(layer.units) == 3
    for unit in layer.units:
        assert isinstance(unit, un.Unit)


def test_layer_should_be_able_to_compute_its_average_activation():
    layer = lr.Layer(name="in", size=2)
    layer.units[0].act = 0
    layer.units[1].act = 1
    assert layer.avg_act() == 0.5
