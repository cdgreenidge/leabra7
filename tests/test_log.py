"""Test log.py"""
import pandas as pd  # type: ignore
import pytest

from leabra7 import unit as un
from leabra7 import layer as lr
from leabra7 import log


# Test log.DataFrameBuffer
def test_it_appends_nones_to_maintain_const_length_if_the_key_does_not_exist():
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


# Test log.simplify(bufkey)
def test_it_converts_the_frequency_to_a_string():
    key = ("name", log.Frequency.CYCLE)
    assert log.simplify(key) == ("name", "cycle")


# Test log.naive_attrreader(attr)
def test_it_reads_an_attribute_from_an_object(mocker):
    moby_dick = mocker.Mock()
    moby_dick.color = 'WHITE!'
    ahabs_lookout = log.naive_attrreader("color")
    assert ahabs_lookout(moby_dick) == ("color", "WHITE!")


# Test log.unit_attrreader(attr)
def test_it_raises_an_error_if_the_unit_attribute_is_invalid():
    with pytest.raises(ValueError):
        log.unit_attrreader("whales")


def test_it_returns_a_function_that_logs_an_attribute_in_a_unit():
    reader = log.unit_attrreader("act")
    assert reader(un.Unit()) == ("act", 0)


# Test log.unit_to_layer_attrreaders(f, n)
def test_it_returns_a_list_of_attrreaders_for_each_unit(mocker):
    layer = mocker.Mock()
    layer.units = [0, 1, 2, 3, 4, 5]
    readers = log.to_layer_attrreaders(lambda x: ("number", x), 6)
    assert [f(layer) for f in readers] == [("unit{0}_number".format(n), n)
                                           for n in range(6)]


# Test log.trim_unit_prefix(unitattr)
def test_it_trims_the_unit_prefix_from_an_attribute():
    assert log.trim_unit_prefix("unit_act") == "act"


# Test log.unit_attrs(attrs)
def test_it_filters_the_unit_attrs_from_a_list_of_attrs():
    attrs = ["unit_act", "whales", "unit_v_m", "porpoises"]
    assert log.unit_attrs(attrs) == ["act", "v_m"]


# Test log.layer_reader(attrs, lr_size)
def test_it_returns_an_object_observation_from_a_layer():
    layer = lr.Layer(name="lr1", size=2)
    f = log.layer_reader(["unit_act"], layer.size)
    exp = 0
    assert f(layer) == [("unit0_act", exp), ("unit1_act", exp)]


# Test log.layer_logger(attrs, freq, layer_size)
def test_it_returns_a_layer_observation_and_buffer_key():
    layer = lr.Layer(name="lr1", size=1)
    logger = log.layer_logger(["unit_act"], log.Frequency.CYCLE, layer.size)
    key, obs = logger(layer)
    assert key == ("lr1", log.Frequency.CYCLE)
    assert obs == [("unit0_act", 0)]


# Test log.naive_reader(attrs):
def test_it_reads_attributes_from_an_object(mocker):
    obj = mocker.Mock()
    obj.a = 3
    obj.b = 4
    reader = log.naive_reader(["a", "b"])
    assert reader(obj) == [("a", 3), ("b", 4)]
