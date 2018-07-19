"""Test net.py"""
import math
from typing import Iterable
from typing import List

import numpy as np
import pytest

from leabra7 import events
from leabra7 import net
from leabra7 import specs
from leabra7 import rand


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


# Test kWTA Inhibition
def test_kwta_suppresses_all_but_k_units() -> None:
    for p in np.linspace(0.0, 1.0, 10):
        n = net.Net()
        n.new_layer(name="lr1", size=1)

        lr2_spec = specs.LayerSpec(
            inhibition_type="kwta",
            kwta_pct=p,
            log_on_cycle=("unit_act", ),
            unit_spec=specs.UnitSpec(adapt_dt=0, spike_gain=0))
        n.new_layer(name="lr2", size=10, spec=lr2_spec)

        pr1_spec = specs.ProjnSpec(dist=rand.Uniform(low=0.25, high=0.75))
        n.new_projn("proj1", "lr1", "lr2", pr1_spec)

        pr2_spec = specs.ProjnSpec(dist=rand.Uniform(low=0.25, high=0.75))
        n.new_projn("proj2", "lr1", "lr2", pr2_spec)

        n.clamp_layer("lr1", [1])
        for i in range(50):
            n.cycle()

        logs = n.logs("cycle", "lr2").parts
        acts = logs[logs.time == 49]["act"]

        assert (acts > 0.5).sum() == max(1, int(round(10 * p)))


# TODO: precisely define this
def test_kwta_avg_suppresses_all_but_k_units() -> None:
    for p in np.linspace(0.4, 1.0, 10):
        n = net.Net()
        n.new_layer(name="lr1", size=1)

        lr2_spec = specs.LayerSpec(
            inhibition_type="kwta_avg",
            kwta_pct=p,
            log_on_cycle=("unit_act", ),
            unit_spec=specs.UnitSpec(adapt_dt=0, spike_gain=0))
        n.new_layer(name="lr2", size=10, spec=lr2_spec)

        pr1_spec = specs.ProjnSpec(dist=rand.Uniform(low=0.25, high=0.75))
        n.new_projn("proj1", "lr1", "lr2", pr1_spec)

        pr2_spec = specs.ProjnSpec(dist=rand.Uniform(low=0.25, high=0.75))
        n.new_projn("proj2", "lr1", "lr2", pr2_spec)

        n.clamp_layer("lr1", [1])
        for i in range(100):
            n.cycle()

        logs = n.logs("cycle", "lr2").parts
        acts = logs[logs.time == 99]["act"]

        # Can't exactly be exact about how many winners selected
        # assert (acts > 0.5).sum() == max(1, int(round(10 * p)))


def test_you_can_hard_clamp_a_layer() -> None:
    n = net.Net()
    n.new_layer("layer1", 4)
    n.clamp_layer("layer1", [0, 1])
    n.cycle()
    expected = [0, 1, 0, 1]
    for i in range(4):
        assert math.isclose(
            n.objs["layer1"].units.act[i], expected[i], abs_tol=1e-6)


def test_you_can_unclamp_a_layer() -> None:
    n = net.Net()
    n.new_layer("layer1", 1)
    n.new_layer("layer2", 2)
    n.new_projn("projn1", pre="layer1", post="layer2")

    n.clamp_layer("layer2", [0])
    n.clamp_layer("layer1", [0.7])
    n.unclamp_layer("layer2")

    # Drive layer 2 so it should spike
    for _ in range(50):
        n.cycle()

    assert (n.layers["layer2"].units.act > 0).all()


def test_clamping_a_layer_validates_its_name() -> None:
    with pytest.raises(ValueError):
        net.Net().clamp_layer("abcd", [0])


def test_unclamping_a_layer_validates_its_name() -> None:
    with pytest.raises(ValueError):
        net.Net().unclamp_layer("abcd")


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
    assert isinstance(n.handle.call_args_list[-2][0][0], events.EndPlusPhase)
    assert isinstance(n.handle.call_args_list[-1][0][0], events.EndTrial)


def test_running_a_plus_phase_runs_the_correct_number_of_cycles(
        mocker) -> None:
    n = net.Net()
    n.new_layer("layer1", 1)
    mocker.spy(n, "handle")
    n.plus_phase_cycle(num_cycles=42)
    assert all(
        isinstance(i, events.Cycle)
        for i in n.handle.call_args_list[1:43][0][0])


# Test XCAL
def trial(network: net.Net, input_pattern: Iterable[float],
          output_pattern: Iterable[float]) -> None:
    """Runs a trial."""
    network.clamp_layer("input", input_pattern)
    network.minus_phase_cycle(num_cycles=50)
    network.clamp_layer("output", output_pattern)
    network.plus_phase_cycle(num_cycles=25)
    network.unclamp_layer("input")
    network.unclamp_layer("output")
    network.learn()


def epoch(network: net.Net, input_patterns: Iterable[Iterable[float]],
          output_patterns: Iterable[Iterable[float]]) -> None:
    """Runs an epoch."""
    for in_pattern, out_pattern in zip(input_patterns, output_patterns):
        for _ in range(10):
            trial(network, in_pattern, out_pattern)
    network.end_epoch()


def batch(network: net.Net, input_patterns: Iterable[Iterable[float]],
          output_patterns: Iterable[Iterable[float]]) -> None:
    """Runs a training batch."""
    num_epochs = 2
    for _ in range(num_epochs):
        epoch(network, input_patterns, output_patterns)
    network.end_batch()


def output(network: net.Net, pattern: Iterable[float]) -> List[float]:
    """Runs the network with an input pattern and cleans up the output."""
    network.clamp_layer("input", pattern)
    for _ in range(50):
        network.cycle()
    # We skip logging for speed
    out = network.layers["output"].units.act.numpy()
    out[out > 0.7] = 1
    out[out < 0.1] = 0
    return list(out)


def test_a_simple_network_can_learn_simple_things() -> None:
    network = net.Net()
    network.new_layer("input", size=2)
    network.new_layer("output", size=2)
    projn_spec = specs.ProjnSpec(lrate=0.2)
    network.new_projn(
        "input_to_output", pre="input", post="output", spec=projn_spec)

    input_patterns = [[1, 0], [0, 1]]
    output_patterns = [[0, 1], [1, 0]]
    batch(network, input_patterns, output_patterns)

    actual = [output(network, i) for i in input_patterns]

    for act, exp in zip(actual, output_patterns):
        assert act == exp


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
    n.new_layer(
        name="layer1",
        size=3,
        spec=specs.LayerSpec(
            log_on_cycle=("avg_act", ),
            log_on_trial=("avg_act", ),
            log_on_epoch=("avg_act", ),
            log_on_batch=("avg_act", )))
    n.plus_phase_cycle(1)
    n.end_epoch()
    n.end_batch()
    for freq in ("cycle", "trial", "epoch", "batch"):
        assert "avg_act" in n.logs(freq, "layer1").whole.columns


def test_network_triggers_cycle_on_cycle_event(mocker) -> None:
    n = net.Net()
    mocker.spy(n, "_cycle")
    n.handle(events.Cycle())
    assert n._cycle.call_count == 1


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


def test_learn_broadcasts_learn_events_to_each_object(mocker) -> None:
    n = net.Net()
    mocker.spy(n, "handle")
    n.learn()
    assert isinstance(n.handle.call_args_list[0][0][0], events.Learn)


def test_you_can_signal_the_end_of_an_epoch(mocker) -> None:
    n = net.Net()
    mocker.spy(n, "handle")
    n.end_epoch()
    assert isinstance(n.handle.call_args_list[0][0][0], events.EndEpoch)


def test_you_can_signal_the_end_of_a_batch(mocker) -> None:
    n = net.Net()
    mocker.spy(n, "handle")
    n.end_batch()
    assert isinstance(n.handle.call_args_list[0][0][0], events.EndBatch)
