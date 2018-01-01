"""Test log.py"""
import pandas as pd  # type: ignore

from leabra7 import log


# Test log.DataFrameBuffer
def test_if_there_is_a_new_attr_it_uses_nones_to_fill_in_missing_values():
    dfb = log.DataFrameBuffer()
    dfb.append([("a", 1)])
    dfb.append([("a", 2), ("b", 2)])
    assert dfb.buf == {"a": [1, 2], "b": [None, 2]}


def test_if_the_attr_list_is_incomplete_it_pads_the_missing_attrs_with_nones():
    dfb = log.DataFrameBuffer()
    dfb.append([("a", 1), ("b", 2)])
    dfb.append([("c", 3)])
    assert dfb.buf == {"a": [1, None], "b": [2, None], "c": [None, 3]}


def test_it_converts_the_buffer_to_a_dataframe_on_flush():
    dfb = log.DataFrameBuffer()
    dfb.append([("a", 1), ("b", 3)])
    dfb.append([("a", 2), ("b", 4)])
    expected = pd.DataFrame.from_dict({"a": [1, 2], "b": [3, 4]})
    assert dfb.to_df().equals(expected)


# Test log.Logger
class ObjToLog:
    """A dummy class with which to test logging."""

    def observe(self, attr):
        return getattr(self, attr)


def test_it_can_record_attributes_from_an_object():
    obj = ObjToLog()
    logger = log.Logger(obj, ["a"])
    obj.a = 3
    logger.record()
    obj.a = 4
    logger.record()
    expected = pd.DataFrame.from_dict({"a": [3, 4]})
    assert logger.to_df().equals(expected)
