"""Test log.py"""
from typing import Any

import pandas as pd  # type: ignore
import pytest

from leabra7 import log


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
        self.avg_act = 0.4
        super().__init__(
            name=name,
            whole_attrs=["avg_act"],
            parts_attrs=["unit_act"],
            *args,
            **kwargs)

    def observe_parts_attr(self, attr: str) -> log.PartsObs:
        if attr != "unit_act":
            raise ValueError("{0} is not a parts attr.".format(attr))
        else:
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


# Test log.merge_parts_observations()
def test_you_can_merge_parts_observations() -> None:
    obs1 = {"unit": [0, 1], "act": [0.2, 0.3]}
    obs2 = {"unit": [0, 1], "net": [0.5, 0.5]}
    merged_dict = {"unit": [0, 1], "act": [0.2, 0.3], "net": [0.5, 0.5]}
    pd.util.testing.assert_frame_equal(
        pd.DataFrame(merged_dict),
        log.merge_parts_observations([obs1, obs2]),
        check_like=True)


# Test log.merge_whole_observations()
def test_you_can_merge_whole_observations() -> None:
    observations = [("avg_act", 0.3), ("fbi", 0.5)]
    expected = pd.DataFrame({"avg_act": [0.3], "fbi": [0.5]})
    pd.util.testing.assert_frame_equal(
        expected, log.merge_whole_observations(observations))


# Test log.Logger
def test_logger_can_record_attributes_from_an_object() -> None:
    obj = ObjToLog("obj")
    logger = log.Logger(obj, ["unit_act", "avg_act"])
    logger.record()
    expected_parts = pd.DataFrame.from_dict({
        "time": [0, 0],
        "act": [0.3, 0.5],
        "unit": [0, 1]
    })
    expected_whole = pd.DataFrame({"avg_act": [0.4], "time": [0]})
    actual = logger.to_logs()
    pd.util.testing.assert_frame_equal(
        expected_parts, actual.parts, check_like=True)
    pd.util.testing.assert_frame_equal(
        expected_whole, actual.whole, check_like=True)


def test_logger_has_a_name_property() -> None:
    obj = ObjToLog("obj")
    logger = log.Logger(obj, ["a"])
    logger.name = "obj"
    assert logger.name == "obj"
