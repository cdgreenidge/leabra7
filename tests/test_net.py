"""Test net.py"""
import pytest

from leabra7 import net
from leabra7 import specs


def test_the_network_can_check_if_an_object_exists_within_it():
    n = net.Net()
    n.new_layer("layer1", 3)
    assert n._assert_obj_exits("layer1")
    with pytest.raises(AssertionError)

def test_a_new_layer_validates_its_spec():
    n = net.Net()
    with pytest.raises(specs.ValidationError):
        n.new_layer("layer1", 3, spec=specs.LayerSpec(integ=-1))


def test_you_can_create_a_layer_with_a_default_spec():
    n = net.Net()
    n.new_layer("layer1", 3)


def test_you_can_force_a_layer():
    n = net.Net()
    n.new_layer("layer1", 4)
    n.force_layer("layer1", [0, 1])
    n.cycle()
    assert [u.act for u in n.objs["layer1"].units] == [0, 1, 0, 1]


def test_forcing_a_layer_validates_its_name():
    with pytest.raises(ValueError):
        net.Net().force_layer("abcd", [0])


def test_a_new_projn_validates_its_spec():
    n = net.Net()
    n.new_layer("layer1", 3)
    n.new_layer("layer2", 3)
    with pytest.raises(specs.ValidationError):
        n.new_projn(
            "projn1", "layer1", "layer2", spec=specs.ProjnSpec(integ=-1))


def test_you_can_create_a_projn_with_a_default_spec():
    n = net.Net()
    n.new_layer("layer1", 3)
    n.new_layer("layer2", 3)
    n.new_projn("projn1", "layer1", "layer2")


def test_projn_checks_if_the_sending_layer_name_is_valid():
    n = net.Net()
    n.new_layer("layer2", 3)
    with pytest.raises(ValueError):
        n.new_projn("projn1", "layer1", "layer2")


def test_projn_checks_if_the_receiving_layer_name_is_valid():
    n = net.Net()
    n.new_layer("layer1", 3)
    with pytest.raises(ValueError):
        n.new_projn("projn1", "layer1", "layer2")


# Right now, it's difficult to test net.cycle(), because it's the core of the
# stateful updates. Eventually, we'll add some regression tests for it.


def test_net_logs_checks_whether_the_frequency_name_is_valid():
    n = net.Net()
    with pytest.raises(ValueError):
        n.logs(freq="guitar", name="layer1")


def test_net_logs_checks_whether_the_object_name_is_valid():
    n = net.Net()
    with pytest.raises(ValueError):
        n.logs(freq="cycle", name="layer1")


def test_you_can_retrieve_the_logs_for_a_layer():
    n = net.Net()
    n.new_layer("layer1", 3, spec=specs.LayerSpec(log_on_cycle=("avg_act", )))
    n.cycle()
    assert "avg_act" in n.logs("cycle", "layer1").columns
