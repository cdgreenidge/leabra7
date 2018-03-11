"""Test log.py"""
from typing import Any
from typing import List
from typing import Tuple

import pandas as pd  # type: ignore

from leabra7 import log


# Test log.DataFrameBuffer
def test_dataframebuffer_can_record_new_attributes() -> None:
    dfb = log.DataFrameBuffer()
    dfb.append([("a", 1)])
    dfb.append([("a", 2), ("b", 2)])
    assert dfb.buf == {"a": [1, 2], "b": [None, 2]}


def test_dataframebuffer_can_record_partial_rows() -> None:
    dfb = log.DataFrameBuffer()
    dfb.append([("a", 1), ("b", 2)])
    dfb.append([("c", 3)])
    assert dfb.buf == {"a": [1, None], "b": [2, None], "c": [None, 3]}


def test_dataframebuffer_can_serialize_itself_to_a_dataframe() -> None:
    dfb = log.DataFrameBuffer()
    dfb.append([("a", 1), ("b", 3)])
    dfb.append([("a", 2), ("b", 4)])
    expected = pd.DataFrame.from_dict({"a": [1, 2], "b": [3, 4]})
    assert dfb.to_df().equals(expected)


# Test log.Logger
class ObjToLog(log.ObservableMixin):
    """A dummy class with which to test logging."""
    name = "obj"

    def observe(self, attr) -> List[Tuple[str, Any]]:
        """Observes an attribute."""
        return [(attr, getattr(self, attr))]


def test_logger_can_record_attributes_from_an_object() -> None:
    obj = ObjToLog("obj")
    logger = log.Logger(obj, ["a"])
    obj.a = 3
    logger.record()
    obj.a = 4
    logger.record()
    expected = pd.DataFrame.from_dict({"a": [3, 4]})
    assert logger.to_df().equals(expected)


def test_logger_has_a_name_property() -> None:
    obj = ObjToLog("obj")
    logger = log.Logger(obj, ["a"])
    logger.name = "obj"
    assert logger.name == "obj"
