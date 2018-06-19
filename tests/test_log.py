"""Test log.py"""
from typing import Any
from typing import List
from typing import Dict

import pytest

import pandas as pd  # type: ignore

from leabra7 import log


# Test log.whole_observations_to_dict on an emtpy list
def test_whole_observations_to_dict_on_empty_list() -> None:
    empty_list = []
    empty_expect = dict()

    empty_result = log.whole_observations_to_dict(empty_list)

    assert empty_expect == empty_result


# Test log.whole_observations_to_dict on a unique keyed list
def test_whole_observations_to_dict_on_unique_key_list() -> None:
    standard_list = [("a", 1), ("b", 2), ("c", 4)]
    standard_expect = {"a": 1, "b": 2, "c": 4}

    standard_result = log.whole_observations_to_dict(standard_list)

    assert standard_expect == standard_result


# Test log.whole_observations_to_dict on a repeat keyed list
def test_whole_observations_to_dict_on_repeat_key_list() -> None:
    repeat_list = [("a", 1), ("b", 2), ("a", 2), ("b", 3), ("c", 4)]

    with pytest.raises(
            AssertionError,
            message='Failed to Raise Asssertion for Repeated Key'):
        log.whole_observations_to_dict(repeat_list)


# Test log.DataFrameBuffer
def test_dataframebuffer_can_record_observations() -> None:
    dfb = log.DataFrameBuffer()
    dfb.append(pd.DataFrame({"unit": [0, 1], "act": [0.5, 0.3]}))
    dfb.append(pd.DataFrame({"unit": [0, 1], "act": [0.6, 0.7]}))
    expected = pd.DataFrame({
        "unit": [0, 1, 0, 1],
        "act": [0.5, 0.3, 0.6, 0.7],
        "time": [0, 0, 1, 1]
    })
    assert dfb.to_df().equals(expected)


class ObjToLog(log.ObservableMixin):
    """A dummy class with which to test logging."""

    def __init__(self, name: str, *args: Any, **kwargs: Any) -> None:
        self.unit = [0, 1]
        self.acts = [0.3, 0.5]
        super().__init__(
            name=name,
            whole_attrs=["avg_act"],
            parts_attrs=["unit_act"],
            *args,
            **kwargs)

    def observe(self, attr: str) -> List[Dict[str, Any]]:
        """Observes an attribute."""
        if attr == "unit0_act":
            return [{"unit": 0, "act": 0.3}]
        elif attr == "unit1_act":
            return [{"unit": 1, "act": 0.5}]
        elif attr == "avg_act":
            return [{"avg_act": 0.4}]
        else:
            raise ValueError("Unknown attribute.")

    def observe_parts_attr(self, attr: str) -> log.PartsObs:
        if attr not in self.parts_attrs:
            raise ValueError("{0} is not a parts attr.".format(attr))
        if attr == "unit_act":
            return {"unit": self.unit, "act": self.acts}


# Test log.ObservableMixin
def test_observable_has_whole_attrs() -> None:
    obj = ObjToLog("obj")
    assert obj.whole_attrs == {"avg_act"}


def test_observable_has_parts_attrs() -> None:
    obj = ObjToLog("obj")
    assert obj.parts_attrs == {"unit_act"}


def test_observable_can_validate_attributes() -> None:
    obj = ObjToLog("obj")
    obj.validate_attr("unit_act")
    obj.validate_attr("avg_act")
    with pytest.raises(ValueError):
        obj.validate_attr("whales")


def test_observing_whole_attrs_raises_error_if_invalid_attr() -> None:
    obj = ObjToLog("obj")
    with pytest.raises(ValueError):
        obj.observe_whole_attr("whales")
    with pytest.raises(ValueError):
        obj.observe_whole_attr("unit_act")


def test_you_can_observe_whole_attributes() -> None:
    obj = ObjToLog("obj")
    obj.avg_act = 0.3
    assert obj.observe_whole_attr("avg_act") == ("avg_act", 0.3)


def test_observing_parts_attrs_raises_error_if_invalid_attr() -> None:
    obj = ObjToLog("obj")
    with pytest.raises(ValueError):
        obj.observe_parts_attr("whales")
    with pytest.raises(ValueError):
        obj.observe_parts_attr("avg_act")


def test_you_can_observe_parts_attributes() -> None:
    obj = ObjToLog("obj")
    assert obj.observe_parts_attr("unit_act") == {
        "unit": [0, 1],
        "act": [0.3, 0.5]
    }


# Test log.Logger
def test_logger_can_record_attributes_from_an_object() -> None:
    obj = ObjToLog("obj")
    logger = log.Logger(obj, ["unit0_act", "unit1_act", "avg_act"])
    logger.record()
    expected = pd.DataFrame.from_dict({
        "act": [0.3, 0.5, None],
        "avg_act": [None, None, 0.4],
        "unit": [0, 1, None]
    })
    expected["time"] = 0
    assert logger.to_df().equals(expected)


def test_logger_has_a_name_property() -> None:
    obj = ObjToLog("obj")
    logger = log.Logger(obj, ["a"])
    logger.name = "obj"
    assert logger.name == "obj"
