"""Tests events.py"""
from hypothesis import example
from hypothesis import given
import hypothesis.strategies as st
import pytest

from leabra7 import events as ev


def test_clamp_checks_if_acts_contains_values_outside_0_1() -> None:
    with pytest.raises(ValueError):
        ev.HardClamp(layer_name="lr1", acts=(1, 2))


def test_pause_logging_checks_for_valid_frequency_names() -> None:
    with pytest.raises(ValueError):
        ev.PauseLogging("whales")

    with pytest.raises(ValueError):
        ev.PauseLogging("cycle", "whales")

    with pytest.raises(ValueError):
        ev.PauseLogging("whales", "cycle")

    with pytest.raises(ValueError):
        ev.PauseLogging("cycle", "whales", "cycle")


def test_resume_logging_checks_for_valid_frequency_name() -> None:
    with pytest.raises(ValueError):
        ev.ResumeLogging("whales")

    with pytest.raises(ValueError):
        ev.ResumeLogging("cycle", "whales")

    with pytest.raises(ValueError):
        ev.ResumeLogging("whales", "cycle")

    with pytest.raises(ValueError):
        ev.ResumeLogging("cycle", "whales", "cycle")


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


@given(st.one_of(st.integers(), st.floats(), st.text()))
def test_freq_object_inequality_with_non_freq(t) -> None:
    for freq in ev.Frequency.freqs():
        assert freq != t


def test_getting_a_frequency_with_undefined_name_raises_error() -> None:
    with pytest.raises(ValueError):
        ev.Frequency.from_name("whales")


def test_you_can_get_the_names_of_all_defined_phases() -> None:
    actual = set(ev.Phase.names())
    expected = set(("none", "plus", "minus"))
    assert actual == expected


def test_phase_type_non_equality() -> None:
    assert ev.PhaseType.PLUS != ev.PhaseType.MINUS
    assert ev.PhaseType.PLUS != ev.PhaseType.NONE
    assert ev.PhaseType.MINUS != ev.PhaseType.NONE


@given(t=st.text())
@example("plus")
@example("minus")
@example("none")
def test_phase_type_retrieval(t) -> None:
    if t == "plus":
        assert ev.PhaseType.get_phase_type(t) == ev.PhaseType.PLUS
    elif t == "minus":
        assert ev.PhaseType.get_phase_type(t) == ev.PhaseType.MINUS
    elif t == "none":
        assert ev.PhaseType.get_phase_type(t) == ev.PhaseType.NONE
    else:
        with pytest.raises(ValueError):
            ev.PhaseType.get_phase_type(t)


def test_you_can_get_phase_objects_by_name() -> None:
    assert ev.Phase.from_name("none") is ev.NonePhase
    assert ev.Phase.from_name("plus") is ev.PlusPhase
    assert ev.Phase.from_name("minus") is ev.MinusPhase


@given(st.one_of(st.integers(), st.floats(), st.text()))
def test_phase_object_inequality_with_non_phase(t) -> None:
    for phase in ev.Phase.phases():
        assert phase != t


def test_can_retrive_phase_names() -> None:
    for phase in ev.Phase.phases():
        assert phase.name == phase.get_name()


def test_getting_a_phase_with_undefined_name_raises_error() -> None:
    with pytest.raises(ValueError):
        ev.Phase.from_name("whales")
