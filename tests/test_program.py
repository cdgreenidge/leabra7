"""Tests program.py"""
import pytest

from leabra7 import program as pr
from leabra7 import layer as lr


def test_node_checks_that_children_is_an_iterable() -> None:
    with pytest.raises(ValueError):
        pr.Node(children=(pr.Node()))


def test_nodes_can_be_names() -> None:
    a = pr.Node(name="bob")
    assert a.name == "bob"


def test_nodes_provide_default_unique_names() -> None:
    n = 25
    assert len(set(pr.Node().name for i in range(n))) == n


def test_nodes_can_provide_a_dfs_preordering_of_their_children() -> None:
    # yapf: disable
    root = pr.Node(name="A", children=(
        pr.Node(name="B", children=(
            pr.Node(name="D"),
        )),
        pr.Node(name="C", children=(
            pr.Node(name="E"),
        ))
    ))
    # yapf: enable
    order = root.preorder()
    names = list(u.name for u in order)
    assert names == ["A", "B", "D", "C", "E"]


def test_nodes_can_stream_their_atomic_event_children() -> None:
    root = pr.Node(
        name="A",
        children=(pr.Node(
            name="B",
            children=(pr.BeginPlusPhase(name="D"), pr.Cycle(name="E"))),
                  pr.Node(name="C", children=(pr.EndPlusPhase(name="F"), ))))
    assert list(u.name for u in root.atomic_stream()) == ["D", "E", "F"]


def test_the_loop_node_loops_its_children() -> None:
    a = pr.Node()
    b = pr.Node()
    assert list(pr.Loop((a, b), num_iter=2).children) == [a, b, a, b]


def test_atomic_events_raise_valueerror_if_they_are_assigned_children(
) -> None:
    with pytest.raises(ValueError):
        pr.Cycle(children=(pr.Node(), ))
    with pytest.raises(ValueError):
        pr.BeginPlusPhase(children=(pr.Node(), ))


def test_hard_clamp_checks_if_acts_contains_values_outside_0_1() -> None:
    with pytest.raises(ValueError):
        pr.HardClamp(layer_name="lr1", acts=(1, 2))
