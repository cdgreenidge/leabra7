"""Test unit.py"""
from leabra7 import specs as sp
from leabra7 import unit as un


def test_unit_init_uses_the_spec_you_pass_it():
    unit = un.Unit(spec=3)
    assert unit.spec == 3


def test_unit_init_can_make_a_defaut_spec_for_you():
    unit = un.Unit()
    assert unit.spec == sp.UnitSpec()


def test_unit_has_0_raw_net_input_at_first():
    unit = un.Unit()
    assert unit.net_raw == 0


def test_unit_can_add_inputs_to_the_raw_net_input():
    unit = un.Unit()
    unit.add_input(3)
    assert unit.net_raw == 3
