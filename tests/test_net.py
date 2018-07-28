"""Test net.py"""
import math

from hypothesis import example
from hypothesis import given
import hypothesis.strategies as st
import numpy as np
import pandas as pd
import pytest
import torch

from leabra7 import events
from leabra7 import net
from leabra7 import specs


def test_network_can_be_saved() -> None:
    n = net.Net()
    location = "tests/mynet.pkl"
    n.save(location)


def test_network_can_be_retrieved_and_continue_logging() -> None:
    n = net.Net()
    n.new_layer(
        "layer1",
        2,
        spec=specs.LayerSpec(log_on_cycle=(
            "unit_act",
            "avg_act",
        )))
    for i in range(2):
        n.cycle()
    n.pause_logging()

    location = "tests/mynet.pkl"
    n.save(location)
    m = net.Net()
    m.load(filename=location)

    before_parts_n = n.logs("cycle", "layer1").parts
    before_parts_m = m.logs("cycle", "layer1").parts

    before_whole_n = n.logs("cycle", "layer1").whole
    before_whole_m = m.logs("cycle", "layer1").whole

    assert np.all((before_parts_n == before_parts_m).values)
    assert np.all((before_whole_n == before_whole_m).values)

    for i in range(2):
        m.cycle()
    m.resume_logging()
    for i in range(2):
        m.cycle()

    after_parts_time = torch.Tensor(m.logs("cycle", "layer1").parts["time"])
    after_whole_time = torch.Tensor(m.logs("cycle", "layer1").whole["time"])

    assert list(after_parts_time.size()) == [8]
    assert list(after_whole_time.size()) == [4]

    assert (after_parts_time == torch.Tensor([0, 0, 1, 1, 4, 4, 5, 5])).all()
    assert (after_whole_time == torch.Tensor([0, 1, 4, 5])).all()


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


def test_the_network_can_get_a_projn_by_name() -> None:
    n = net.Net()
    n.new_layer("lr1", 1)
    n.new_layer("lr2", 1)
    n.new_projn("pr1", "lr1", "lr2")

    assert n._get_projn("pr1") is n.projns["pr1"]


def test_the_network_can_validate_projn_names() -> None:
    n = net.Net()
    n.new_layer("lr1", 1)
    n.new_layer("lr2", 1)
    n.new_projn("pr1", "lr1", "lr2")

    n._validate_projn_name("pr1")

    with pytest.raises(ValueError):
        n._validate_projn_name("whales")


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


def test_net_can_inhibit_projns() -> None:
    n = net.Net()
    n.new_layer("lr1", size=1)
    n.new_layer("lr2", size=1)

    n.new_projn("pr1", "lr1", "lr2")
    n.new_projn("pr2", "lr1", "lr2")
    n.new_projn("pr3", "lr1", "lr2")

    n.inhibit_projns("pr1", "pr2", "pr3")


def test_net_catches_inhibit_bad_projn_name() -> None:
    n = net.Net()
    n.new_layer("lr1", size=1)
    n.new_layer("lr2", size=1)

    n.new_projn("pr1", "lr1", "lr2")
    n.new_projn("pr2", "lr1", "lr2")
    n.new_projn("pr3", "lr1", "lr2")

    with pytest.raises(ValueError):
        n.inhibit_projns("pr4")

    with pytest.raises(ValueError):
        n.inhibit_projns("pr1", "pr5", "pr3")


def test_net_can_uninhibit_projns() -> None:
    n = net.Net()
    n.new_layer("lr1", size=1)
    n.new_layer("lr2", size=1)

    n.new_projn("pr1", "lr1", "lr2")
    n.new_projn("pr2", "lr1", "lr2")
    n.new_projn("pr3", "lr1", "lr2")

    n.uninhibit_projns("pr1", "pr2", "pr3")


def test_net_catches_uninhibit_bad_projn_name() -> None:
    n = net.Net()
    n.new_layer("lr1", size=1)
    n.new_layer("lr2", size=1)

    n.new_projn("pr1", "lr1", "lr2")
    n.new_projn("pr2", "lr1", "lr2")
    n.new_projn("pr3", "lr1", "lr2")

    with pytest.raises(ValueError):
        n.uninhibit_projns("pr4")

    with pytest.raises(ValueError):
        n.uninhibit_projns("pr1", "pr5", "pr3")


# Right now, it's difficult to test net.cycle(), because it's the core of the
# stateful updates. Eventually, we'll add some regression tests for it.


def test_running_a_phase_raises_error_if_num_cycles_less_than_one() -> None:
    with pytest.raises(ValueError):
        net.Net().phase_cycle(phase=events.PlusPhase, num_cycles=-1)


def test_running_a_phase_broadcasts_phase_event_markers(mocker) -> None:
    for phase_name in events.Phase.names():
        phase = events.Phase.from_name(phase_name)

        n = net.Net()
        mocker.spy(n, "handle")

        if phase.type == events.PhaseType.none:
            with pytest.raises(ValueError):
                n.phase_cycle(phase=phase, num_cycles=1)
            return

        else:
            n.phase_cycle(phase=phase, num_cycles=1)

            assert n.handle.call_args_list[0][0][0] == phase.begin_event
            assert isinstance(n.handle.call_args_list[1][0][0], events.Cycle)
            assert n.handle.call_args_list[2][0][0] == phase.end_event


def test_running_a_phase_runs_the_correct_number_of_cycles(mocker) -> None:
    for phase_name in events.Phase.names():
        phase = events.Phase.from_name(phase_name)
        n = net.Net()
        mocker.spy(n, "handle")

        if phase.type == events.PhaseType.none:
            with pytest.raises(ValueError):
                n.phase_cycle(phase=phase, num_cycles=42)

        else:
            n.phase_cycle(phase=phase, num_cycles=42)

            assert all(
                isinstance(i, events.Cycle)
                for i in n.handle.call_args_list[1:43][0][0])


def test_you_can_observe_unlogged_attributes() -> None:
    n = net.Net()
    n.new_layer("layer1", 1)
    n.new_layer("layer2", 1)
    n.new_projn("projn1", "layer1", "layer2")
    pd.util.testing.assert_frame_equal(
        n.observe("projn1", "cos_diff_avg"),
        pd.DataFrame({
            "cos_diff_avg": (0.0, )
        }),
        check_like=True)


def test_observing_unlogged_attr_raises_error_if_obj_not_observable() -> None:
    n = net.Net()
    n.new_layer("layer1", 1, spec=specs.LayerSpec(log_on_cycle=("avg_act", )))
    with pytest.raises(ValueError):
        n.observe("layer1_cycle_logger", "cos_diff_avg")


def test_observing_unlogged_attr_checks_whether_attr_is_invalid() -> None:
    n = net.Net()
    n.new_layer("layer1", 1)
    with pytest.raises(ValueError):
        n.observe("layer1", "whales")


def test_observing_unlogged_attr_checks_whether_obj_exists() -> None:
    n = net.Net()
    with pytest.raises(ValueError):
        n.observe("layer1", "avg_act")


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
    n.phase_cycle(phase=events.PlusPhase, num_cycles=1)
    n.end_trial()
    n.end_epoch()
    n.end_batch()
    for freq in ("cycle", "trial", "epoch", "batch"):
        assert "avg_act" in n.logs(freq, "layer1").whole.columns


def test_net_can_pause_and_resume_logging() -> None:
    n = net.Net()
    n.new_layer(
        "layer1",
        2,
        spec=specs.LayerSpec(log_on_cycle=(
            "unit_act",
            "avg_act",
        )))
    for i in range(2):
        n.cycle()
    n.pause_logging()
    for i in range(2):
        n.cycle()
    n.resume_logging()
    for i in range(2):
        n.cycle()

    parts_time = torch.Tensor(n.logs("cycle", "layer1").parts["time"])
    whole_time = torch.Tensor(n.logs("cycle", "layer1").whole["time"])
    assert list(parts_time.size()) == [8]
    assert list(whole_time.size()) == [4]

    assert (parts_time == torch.Tensor([0, 0, 1, 1, 4, 4, 5, 5])).all()
    assert (whole_time == torch.Tensor([0, 1, 4, 5])).all()


def test_net_trial_log_pausing_and_resuming() -> None:
    n = net.Net()
    n.new_layer(
        "layer1",
        2,
        spec=specs.LayerSpec(log_on_trial=(
            "unit_act",
            "avg_act",
        )))

    n.phase_cycle(phase=events.MinusPhase, num_cycles=5)
    n.phase_cycle(phase=events.PlusPhase, num_cycles=5)
    n.end_trial()

    n.pause_logging("trial")

    n.phase_cycle(phase=events.MinusPhase, num_cycles=5)
    n.phase_cycle(phase=events.PlusPhase, num_cycles=5)
    n.end_trial()

    n.resume_logging("trial")

    n.phase_cycle(phase=events.MinusPhase, num_cycles=5)
    n.phase_cycle(phase=events.PlusPhase, num_cycles=5)
    n.end_trial()

    parts_time = torch.Tensor(n.logs("trial", "layer1").parts["time"])
    whole_time = torch.Tensor(n.logs("trial", "layer1").whole["time"])

    assert list(parts_time.size()) == [4]
    assert list(whole_time.size()) == [2]

    assert (parts_time == torch.Tensor([0, 0, 2, 2])).all()
    assert (whole_time == torch.Tensor([0, 2])).all()


def test_net_epoch_log_pausing_and_resuming() -> None:
    n = net.Net()
    n.new_layer(
        "layer1",
        2,
        spec=specs.LayerSpec(log_on_epoch=(
            "unit_act",
            "avg_act",
        )))

    n.end_epoch()
    n.pause_logging("epoch")
    n.end_epoch()
    n.resume_logging("epoch")
    n.end_epoch()

    parts_time = torch.Tensor(n.logs("epoch", "layer1").parts["time"])
    whole_time = torch.Tensor(n.logs("epoch", "layer1").whole["time"])

    assert list(parts_time.size()) == [4]
    assert list(whole_time.size()) == [2]

    assert (parts_time == torch.Tensor([0, 0, 2, 2])).all()
    assert (whole_time == torch.Tensor([0, 2])).all()


def test_net_batch_log_pausing_and_resuming() -> None:
    n = net.Net()
    n.new_layer(
        "layer1",
        2,
        spec=specs.LayerSpec(log_on_batch=(
            "unit_act",
            "avg_act",
        )))

    n.end_batch()
    n.pause_logging("batch")
    n.end_batch()
    n.resume_logging("batch")
    n.end_batch()

    parts_time = torch.Tensor(n.logs("batch", "layer1").parts["time"])
    whole_time = torch.Tensor(n.logs("batch", "layer1").whole["time"])

    assert list(parts_time.size()) == [4]
    assert list(whole_time.size()) == [2]

    assert (parts_time == torch.Tensor([0, 0, 2, 2])).all()
    assert (whole_time == torch.Tensor([0, 2])).all()


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

    n.handle(events.PlusPhase.begin_event)

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
