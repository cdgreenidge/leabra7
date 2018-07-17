"""Test net.py"""
import pytest

from leabra7 import events
from leabra7 import net
from leabra7 import specs


def test_the_network_can_check_if_an_object_exists_within_it() -> None:
    n = net.Net()
    n.new_layer("layer1", 3)
    n._validate_obj_name("layer1")
    with pytest.raises(ValueError):
        n._validate_obj_name("whales")


def test_the_network_can_get_a_layer_by_name() -> None:
    n = net.Net()
    n.new_layer("layer1", 3)
    layer1 = n.layers["layer1"]
    n.new_layer("layer2", 3)
    assert n._get_layer("layer1") is layer1


def test_the_network_can_validate_layer_names() -> None:
    n = net.Net()
    n.new_layer("layer1", 3)
    n._validate_layer_name("layer1")
    with pytest.raises(ValueError):
        n._validate_layer_name("whales")


def test_getting_an_invalid_layer_name_raises_value_error() -> None:
    n = net.Net()
    n.new_layer("layer1", 3)
    n.new_layer("layer2", 3)
    n.new_projn("proj1", "layer1", "layer2")
    with pytest.raises(ValueError):
        n._get_layer("whales")
    with pytest.raises(ValueError):
        n._get_layer("proj1")


def test_a_new_layer_validates_its_spec() -> None:
    n = net.Net()
    with pytest.raises(specs.ValidationError):
        n.new_layer("layer1", 3, spec=specs.LayerSpec(integ=-1))


def test_you_can_create_a_layer_with_a_default_spec() -> None:
    n = net.Net()
    n.new_layer("layer1", 3)


def test_a_new_projn_validates_its_spec() -> None:
    n = net.Net()
    n.new_layer("layer1", 3)
    n.new_layer("layer2", 3)
    with pytest.raises(specs.ValidationError):
        n.new_projn(
            "projn1", "layer1", "layer2", spec=specs.ProjnSpec(integ=-1))


def test_you_can_create_a_projn_with_a_default_spec() -> None:
    n = net.Net()
    n.new_layer("layer1", 3)
    n.new_layer("layer2", 3)
    n.new_projn("projn1", "layer1", "layer2")


def test_projn_checks_if_the_sending_layer_name_is_valid() -> None:
    n = net.Net()
    n.new_layer("layer2", 3)
    with pytest.raises(ValueError):
        n.new_projn("projn1", "layer1", "layer2")


def test_projn_checks_if_the_receiving_layer_name_is_valid() -> None:
    n = net.Net()
    n.new_layer("layer1", 3)
    with pytest.raises(ValueError):
        n.new_projn("projn1", "layer1", "layer2")


# Right now, it's difficult to test net.cycle(), because it's the core of the
# stateful updates. Eventually, we'll add some regression tests for it.


def test_you_can_force_a_layer() -> None:
    n = net.Net()
    n.new_layer("layer1", 4)
    n.force_layer("layer1", [0, 1])
    n.cycle()
    assert list(n.objs["layer1"].units.act) == [0, 1, 0, 1]


def test_forcing_a_layer_validates_its_name() -> None:
    with pytest.raises(ValueError):
        net.Net().force_layer("abcd", [0])


def test_running_a_minus_phase_raises_error_if_num_cycles_less_than_one(
) -> None:
    with pytest.raises(ValueError):
        net.Net().minus_phase_cycle(-1)


def test_running_a_minus_phase_broadcasts_minus_phase_event_markers(
        mocker) -> None:
    n = net.Net()
    n.new_layer("layer1", 1)
    mocker.spy(n, "handle")
    n.minus_phase_cycle(num_cycles=1)
    assert isinstance(n.handle.call_args_list[0][0][0], events.BeginMinusPhase)
    assert isinstance(n.handle.call_args_list[-1][0][0], events.EndMinusPhase)


def test_running_a_minus_phase_runs_the_correct_number_of_cycles(
        mocker) -> None:
    n = net.Net()
    n.new_layer("layer1", 1)
    mocker.spy(n, "handle")
    n.minus_phase_cycle(num_cycles=42)
    assert all(
        isinstance(i, events.Cycle)
        for i in n.handle.call_args_list[1:43][0][0])


def test_running_a_plus_phase_raises_error_if_num_cycles_less_than_one(
) -> None:
    with pytest.raises(ValueError):
        net.Net().plus_phase_cycle(-1)


def test_running_a_plus_phase_broadcasts_plus_phase_event_markers(
        mocker) -> None:
    n = net.Net()
    n.new_layer("layer1", 1)
    mocker.spy(n, "handle")
    n.plus_phase_cycle(num_cycles=1)
    assert isinstance(n.handle.call_args_list[0][0][0], events.BeginPlusPhase)
    assert isinstance(n.handle.call_args_list[-1][0][0], events.EndPlusPhase)


def test_running_a_plus_phase_runs_the_correct_number_of_cycles(
        mocker) -> None:
    n = net.Net()
    n.new_layer("layer1", 1)
    mocker.spy(n, "handle")
    n.plus_phase_cycle(num_cycles=42)
    assert all(
        isinstance(i, events.Cycle)
        for i in n.handle.call_args_list[1:43][0][0])


def test_net_logs_checks_whether_the_frequency_name_is_valid() -> None:
    n = net.Net()
    with pytest.raises(ValueError):
        n.logs(freq="guitar", name="layer1")


def test_net_logs_checks_whether_the_object_name_is_valid() -> None:
    n = net.Net()
    with pytest.raises(ValueError):
        n.logs(freq="cycle", name="layer1")


def test_you_can_retrieve_the_logs_for_a_layer() -> None:
    n = net.Net()
    n.new_layer("layer1", 3, spec=specs.LayerSpec(log_on_cycle=("avg_act", )))
    n.cycle()
    assert "avg_act" in n.logs("cycle", "layer1").whole.columns


def test_network_triggers_cycle_on_cycle_event(mocker) -> None:
    n = net.Net()
    mocker.spy(n, "cycle")
    n.handle(events.Cycle())
    assert n.cycle.call_count == 1


def test_network_passes_non_cycle_events_to_every_object(mocker) -> None:
    n = net.Net()
    n.new_layer("layer1", 3)
    n.new_layer("layer2", 3)
    n.new_projn("projn1", "layer1", "layer2")

    for _, obj in n.objs.items():
        mocker.spy(obj, "handle")

    n.handle(events.BeginPlusPhase)

    for _, obj in n.objs.items():
        assert obj.handle.call_count == 1
