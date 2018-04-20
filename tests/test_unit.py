"""Test unit.py"""
import math
from typing import Tuple

import numpy as np
import pytest
import torch

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
    unit = un.Unit()
    xs, conv = nxx1_table
    for i in range(0, xs.size, 50):
        assert math.isclose(unit.nxx1(xs[i]), conv[i])


def test_nxx1_equals_the_min_value_outside_the_min_bound(nxx1_table) -> None:
    unit = un.Unit()
    xs, conv = nxx1_table
    assert unit.nxx1(xs[0] - 1) == conv[0]


def test_nxx1_equals_the_max_value_outside_the_max_bound(nxx1_table) -> None:
    unit = un.Unit()
    xs, conv = nxx1_table
    assert unit.nxx1(xs[-1] + 1) == conv[-1]


# Test Unit class
def test_unit_init_uses_the_spec_you_pass_it() -> None:
    foo = sp.UnitSpec()
    unit = un.Unit(spec=foo)
    assert unit.spec is foo


def test_unit_init_can_make_a_defaut_spec_for_you() -> None:
    unit = un.Unit()
    assert unit.spec == sp.UnitSpec()


def test_unit_has_0_raw_un_input_at_first() -> None:
    unit = un.Unit()
    assert unit.net_raw == 0


def test_unit_can_add_inputs_to_the_raw_un_input() -> None:
    unit = un.Unit()
    unit.add_input(3)
    assert unit.net_raw == 3


def test_unit_can_update_its_membrane_potential() -> None:
    unit = un.Unit()
    unit.update_membrane_potential()


def test_unit_can_update_its_activation() -> None:
    unit = un.Unit()
    unit.update_activation()


def test_unit_can_observe_its_attributes() -> None:
    unit = un.Unit()
    assert unit.observe("act") == {"act": 0.0}


def test_unit_raises_valueerror_if_attr_is_unobservable() -> None:
    unit = un.Unit()
    with pytest.raises(ValueError):
        unit.observe("banannas")


def test_unit_can_calculate_the_inhibition_to_put_it_at_threshold() -> None:
    unit = un.Unit()
    unit.add_input(3)
    unit.update_net()
    unit.update_inhibition(unit.g_i_thr())

    for i in range(200):
        unit.update_membrane_potential()

    assert math.isclose(unit.v_m, unit.spec.spk_thr, rel_tol=1e-4)


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


def test_you_can_observe_attrs_from_the_unit_group() -> None:
    n = 3
    ug = un.UnitGroup(size=n)
    for i in ug.loggable_attrs:
        logs = ug.observe(i)
        assert logs["unit"] == list(range(n))
        assert logs[i] == list(getattr(ug, i))


def test_it_checks_for_unobservable_attrs() -> None:
    ug = un.UnitGroup(3)
    with pytest.raises(ValueError):
        ug.observe("rabbit cm")


def test_unitgroup_can_calculate_the_threshold_inhibition() -> None:
    group = un.UnitGroup(size=10)
    group.add_input(torch.Tensor(np.linspace(0.3, 0.8, 10)))
    group.update_net()
    group.update_inhibition(group.g_i_thr())

    for i in range(200):
        group.update_membrane_potential()

    assert (torch.abs(group.v_m - group.spec.spk_thr) < 1e-6).all()


def test_unitgroup_uses_the_default_spec_if_none_is_provided() -> None:
    group = un.UnitGroup(size=3)
    assert group.spec == sp.UnitSpec()


def test_unitgroup_sets_the_spec_you_provide() -> None:
    spec = sp.UnitSpec()
    assert un.UnitGroup(size=3, spec=spec).spec is spec


def test_unitgroup_has_the_same_behavior_as_unit() -> None:
    def units_are_equal(u0: un.Unit, u1: un.Unit) -> bool:
        """Returns true if two units have the same state."""
        attrs = ("net_raw", "net", "gc_i", "act", "i_net", "i_net_r", "v_m",
                 "v_m_eq", "adapt", "spike")
        for i in attrs:
            assert getattr(u0, i) == getattr(u1, i)

    unit0 = un.Unit()
    unit1 = un.Unit()
    group = un.UnitGroup(size=2)

    for i in range(500):
        unit0.add_input(0.3)
        unit1.add_input(0.5)
        group.add_input(torch.Tensor([0.3, 0.5]))
        unit0.update_net()
        unit1.update_net()
        group.update_net()
        unit0.update_inhibition(0.1)
        unit1.update_inhibition(0.1)
        group.update_inhibition(torch.Tensor([0.1, 0.1]))
        unit0.update_membrane_potential()
        unit1.update_membrane_potential()
        group.update_membrane_potential()
        unit0.update_activation()
        unit1.update_activation()
        group.update_activation()

        attrs = ("net_raw", "net", "gc_i", "act", "i_net", "i_net_r", "v_m",
                 "v_m_eq", "adapt", "spike")
        for i in attrs:
            group_attr = getattr(group, i)
            assert math.isclose(getattr(unit0, i), group_attr[0], abs_tol=1e-6)
            assert math.isclose(getattr(unit1, i), group_attr[1], abs_tol=1e-6)
