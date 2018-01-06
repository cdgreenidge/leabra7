"""Test unit.py"""
import math

import numpy as np
import pytest

from leabra7 import specs as sp
from leabra7 import unit as un


# Test un.gaussian(res, std)
def test_gaussian_returns_a_probability_density():
    assert math.isclose(sum(un.gaussian(res=0.001, std=0.01)), 1)


def test_gaussian_raises_an_error_if_std_is_too_small():
    with pytest.raises(ValueError):
        un.gaussian(res=0.001, std=0.9e-3)


def test_gaussian_raises_an_error_if_res_is_too_big():
    with pytest.raises(ValueError):
        un.gaussian(res=3, std=0.9)


# Test un.xx1(res, xmin, xmax)
def test_xx1_returns_the_correct_array():
    # This is a regression test on Fabien Benreau's prototype
    reference = np.load("tests/xx1.npy")
    assert np.allclose(un.xx1(0.001, -3, 3), reference)


@pytest.fixture(scope="module")
def nxx1_table():
    return un.nxx1_table()


# Test un.nxx1_table()
def test_nxx1_table_returns_the_correct_arrays(nxx1_table):
    # This is a regression test on Fabien Benreau's prototype
    file = np.load("tests/nxx1.npz")
    reference_xs = file["xs"]
    reference_conv = file["conv"]
    file.close()
    xs, conv = nxx1_table
    assert np.allclose(reference_xs, xs)
    assert np.allclose(reference_conv, conv)


# Test nxx1_interpolator() and nxx1()
def test_nxx1_equals_the_lookup_table(nxx1_table):
    unit = un.Unit()
    xs, conv = nxx1_table
    for i in range(0, xs.size, 50):
        assert math.isclose(unit.nxx1(xs[i]), conv[i])


def test_nxx1_equals_the_min_table_value_outside_the_min_boundary(nxx1_table):
    unit = un.Unit()
    xs, conv = nxx1_table
    assert unit.nxx1(xs[0] - 1) == conv[0]


def test_nxx1_equals_the_max_table_value_outside_the_max_boundary(nxx1_table):
    unit = un.Unit()
    xs, conv = nxx1_table
    assert unit.nxx1(xs[-1] + 1) == conv[-1]


# Test Unit class
def test_unit_init_uses_the_spec_you_pass_it():
    unit = un.Unit(spec=3)
    assert unit.spec == 3


def test_unit_init_can_make_a_defaut_spec_for_you():
    unit = un.Unit()
    assert unit.spec == sp.UnitSpec()


def test_unit_has_0_raw_un_input_at_first():
    unit = un.Unit()
    assert unit.net_raw == 0


def test_unit_can_add_inputs_to_the_raw_un_input():
    unit = un.Unit()
    unit.add_input(3)
    assert unit.net_raw == 3


def test_unit_can_update_its_membrane_potential():
    unit = un.Unit()
    unit.update_membrane_potential()


def test_unit_can_update_its_activation():
    unit = un.Unit()
    unit.update_activation()


def test_unit_can_observe_its_attributes():
    unit = un.Unit()
    assert unit.observe("act") == [("act", 0.0)]


def test_unit_raises_valueerror_if_attr_is_unobservable():
    unit = un.Unit()
    with pytest.raises(ValueError):
        unit.observe("banannas")
