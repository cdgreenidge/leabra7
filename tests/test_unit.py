"""Test unit.py"""
import math
from typing import Tuple

from hypothesis import given
import hypothesis.strategies as st
import numpy as np
import pytest
import torch  # type: ignore

from leabra7 import specs as sp
from leabra7 import unit as un


# Test un.gaussian(res, std)
def test_gaussian_returns_a_probability_density() -> None:
    assert math.isclose(sum(un.gaussian(res=0.001, std=0.01)), 1)


def test_gaussian_raises_an_error_if_std_is_too_small() -> None:
    with pytest.raises(ValueError):
        un.gaussian(res=0.001, std=0.9e-3)


def test_gaussian_raises_an_error_if_res_is_too_big() -> None:
    with pytest.raises(ValueError):
        un.gaussian(res=3, std=0.9)


# Test un.xx1(res, xmin, xmax)
def test_xx1_returns_the_correct_array() -> None:
    # This is a regression test on Fabien Benreau's prototype
    reference = np.load("tests/xx1.npy")
    assert np.allclose(un.xx1(0.001, -3, 3), reference)


@pytest.fixture(scope="module", name="nxx1_table")
def nxx1_table_fixture() -> Tuple[np.ndarray, np.ndarray]:
    """Returns a lookup table for the NXX1 function."""
    return un.nxx1_table()


# Test un.nxx1_table()
def test_nxx1_table_returns_the_correct_arrays(nxx1_table) -> None:
    # This is a regression test on Fabien Benreau's prototype
    file = np.load("tests/nxx1.npz")
    reference_xs = file["xs"]
    reference_conv = file["conv"]
    file.close()
    xs, conv = nxx1_table
    assert np.allclose(reference_xs, xs)
    assert np.allclose(reference_conv, conv)


# Test nxx1_interpolator() and nxx1()
def test_nxx1_equals_the_lookup_table(nxx1_table) -> None:
    unit = un.UnitGroup(size=1)
    xs, conv = nxx1_table
    for i in range(0, xs.size, 50):
        assert math.isclose(unit.nxx1(xs[i]), conv[i], abs_tol=1e-6)


def test_nxx1_equals_the_min_value_outside_the_min_bound(nxx1_table) -> None:
    unit = un.UnitGroup(size=1)
    xs, conv = nxx1_table
    assert unit.nxx1(xs[0] - 1) == conv[0]


def test_nxx1_equals_the_max_value_outside_the_max_bound(nxx1_table) -> None:
    unit = un.UnitGroup(size=1)
    xs, conv = nxx1_table
    assert unit.nxx1(xs[-1] + 1) == conv[-1]


@given(
    vals=st.lists(
        elements=st.floats(min_value=0.0, max_value=1.0),
        min_size=1,
        max_size=10),
    minimum=st.floats(min_value=0.0, max_value=1.0),
    maximum=st.floats(min_value=0.0, max_value=1.0))
def test_clip_method(vals, minimum, maximum):
    if minimum < maximum:
        clipped = un.clip(torch.Tensor(vals), minimum, maximum)
        for i in range(len(clipped)):
            assert minimum <= clipped[i] <= maximum


# Test Unit Group
def test_unitgroup_init_checks_that_size_is_positive() -> None:
    with pytest.raises(ValueError):
        un.UnitGroup(size=0)


def test_unitgroup_update_net_checks_input_dimensions() -> None:
    ug = un.UnitGroup(size=3)
    with pytest.raises(AssertionError):
        ug.add_input(torch.Tensor([1, 2]))


def test_unitgroup_update_inhibition_checks_input_dimensions() -> None:
    ug = un.UnitGroup(size=3)
    with pytest.raises(AssertionError):
        ug.update_inhibition(torch.Tensor([1, 2]))


def test_you_can_observe_parts_attrs_from_the_unit_group() -> None:
    n = 2
    ug = un.UnitGroup(size=n)
    for attr in ug.loggable_attrs:
        logs = ug.observe(attr)
        logs == {"unit": [0, 1], attr: getattr(ug, attr).numpy().tolist()}


def test_observing_an_invalid_attr_raises_an_error() -> None:
    ug = un.UnitGroup(size=3)
    with pytest.raises(ValueError):
        ug.observe("whales")


def test_unitgroup_can_calculate_the_threshold_inhibition() -> None:
    group = un.UnitGroup(size=10)
    group.add_input(torch.Tensor(np.linspace(0.3, 0.8, 10)))
    group.update_net()
    g_i_thr = group.g_i_thr(unit_idx=2)
    group.update_inhibition(torch.Tensor(10).fill_(g_i_thr))

    for i in range(200):
        group.update_membrane_potential()

    assert (torch.abs(group.v_m - group.spec.spk_thr) < 1e-6)[2]


def test_unitgroup_can_calculate_each_units_threshold_inhibition() -> None:
    group = un.UnitGroup(size=10)
    group.add_input(torch.Tensor(np.linspace(0.3, 0.8, 10)))
    group.update_net()
    g_i_thr = group.group_g_i_thr()
    group.update_inhibition(g_i_thr)

    for i in range(200):
        group.update_membrane_potential()

    assert torch.sum(torch.abs(group.v_m - group.spec.spk_thr) > 1e-6) == 0


def test_unitgroup_uses_the_default_spec_if_none_is_provided() -> None:
    group = un.UnitGroup(size=3)
    assert group.spec == sp.UnitSpec()


def test_unitgroup_sets_the_spec_you_provide() -> None:
    spec = sp.UnitSpec()
    assert un.UnitGroup(size=3, spec=spec).spec is spec


def test_unitgroup_can_return_the_top_k_net_input_values() -> None:
    group = un.UnitGroup(size=10)
    group.net = torch.Tensor([9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
    assert (group.top_k_net_indices(3) == torch.Tensor([0, 1, 2]).long()).all()
