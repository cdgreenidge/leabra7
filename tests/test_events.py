"""Tests events.py"""
import pytest

from leabra7 import events as ev


def test_clamp_checks_if_acts_contains_values_outside_0_1() -> None:
    with pytest.raises(ValueError):
        ev.HardClamp(layer_name="lr1", acts=(1, 2))


def test_pause_logging_checks_for_valid_frequency_name() -> None:
    with pytest.raises(ValueError):
        ev.PauseLogging(freq_name="whales")


def test_resume_logging_checks_for_valid_frequency_name() -> None:
    with pytest.raises(ValueError):
        ev.ResumeLogging(freq_name="whales")


def test_initializing_frequency_with_incorrect_type_raises_error() -> None:
    with pytest.raises(TypeError):
        ev.Frequency(name="cycle", end_event_type=ev.Cycle())


def test_you_can_get_the_names_of_all_defined_frequencies() -> None:
    actual = set(ev.Frequency.names())
    expected = set(("cycle", "trial", "epoch", "batch"))
    assert actual == expected


def test_you_can_get_frequency_objects_by_name() -> None:
    assert ev.Frequency.from_name("cycle") is ev.CycleFreq
    assert ev.Frequency.from_name("trial") is ev.TrialFreq
    assert ev.Frequency.from_name("epoch") is ev.EpochFreq
    assert ev.Frequency.from_name("batch") is ev.BatchFreq


def test_getting_a_frequency_with_undefined_name_raises_error() -> None:
    with pytest.raises(ValueError):
        ev.Frequency.from_name("whales")


def test_initializing_phase_with_incorrect_type_raises_error() -> None:
    with pytest.raises(TypeError):
        ev.Phase(
            name="plus_cycle",
            begin_event_type=ev.BeginPlusPhase,
            end_event_type=ev.Cycle())
    with pytest.raises(TypeError):
        ev.Phase(
            name="plus_cycle",
            begin_event_type=ev.BeginPlusPhase(),
            end_event_type=ev.Cycle)


def test_you_can_get_the_names_of_all_defined_phases() -> None:
    actual = set(ev.Phase.names())
    expected = set(("none", "plus", "minus", "theta_trough", "theta_peak",
                    "theta_plus"))
    assert actual == expected


def test_you_can_get_phase_objects_by_name() -> None:
    assert ev.Phase.from_name("plus") is ev.PlusPhase
    assert ev.Phase.from_name("minus") is ev.MinusPhase
    assert ev.Phase.from_name("theta_trough") is ev.ThetaTrough
    assert ev.Phase.from_name("theta_peak") is ev.ThetaPeak
    assert ev.Phase.from_name("theta_plus") is ev.ThetaPlus


def test_getting_a_phase_with_undefined_name_raises_error() -> None:
    with pytest.raises(ValueError):
        ev.Phase.from_name("whales")
