"""Test log.py"""
import pandas as pd  # type: ignore
import pytest

from leabra7 import log


# Test log.DataFrameBuffer
def test_dataframebuffer_can_record_new_attributes():
    dfb = log.DataFrameBuffer()
    dfb.append([("a", 1)])
    dfb.append([("a", 2), ("b", 2)])
    assert dfb.buf == {"a": [1, 2], "b": [None, 2]}


def test_dataframebuffer_can_record_partial_rows():
    dfb = log.DataFrameBuffer()
    dfb.append([("a", 1), ("b", 2)])
    dfb.append([("c", 3)])
    assert dfb.buf == {"a": [1, None], "b": [2, None], "c": [None, 3]}


def test_dataframebuffer_can_serialize_itself_to_a_dataframe():
    dfb = log.DataFrameBuffer()
    dfb.append([("a", 1), ("b", 3)])
    dfb.append([("a", 2), ("b", 4)])
    expected = pd.DataFrame.from_dict({"a": [1, 2], "b": [3, 4]})
    assert dfb.to_df().equals(expected)


# Test log.Logger
class ObjToLog:
    """A dummy class with which to test logging."""
    name = "obj"

    def observe(self, attr):
        return [(attr, getattr(self, attr))]


def test_logger_can_record_attributes_from_an_object():
    obj = ObjToLog()
    logger = log.Logger(obj, "cycle", ["a"])
    obj.a = 3
    logger.record()
    obj.a = 4
    logger.record()
    expected = pd.DataFrame.from_dict({"a": [3, 4]})
    assert logger.to_df().equals(expected)


def test_logger_has_a_name_property():
    obj = ObjToLog()
    logger = log.Logger(obj, "cycle", ["a"])
    logger.name = "obj"
    assert logger.name == "obj"


def test_logger_has_a_frequency_property():
    obj = ObjToLog()
    logger = log.Logger(obj, "cycle", ["a"])
    assert logger.freq == "cycle"


def test_logger_throws_an_error_for_invalid_frequencies():
    obj = ObjToLog()
    with pytest.raises(ValueError):
        log.Logger(obj, "antenna", ["a"])
