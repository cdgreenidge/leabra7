"""Test utils.py"""
import unittest

from hypothesis import given
import hypothesis.strategies as st
import pytest

from leabra7 import utils as ut


@st.composite
def k_and_list(draw, elements=st.integers()):
    xs = draw(st.lists(elements, min_size=1))
    i = draw(st.integers(min_value=0, max_value=len(xs)))
    return (i, xs)


@given(k_and_list())
def test_partition_partitions_a_list_without_key_function(x):
    n, ys = x
    first, second = ut.partition(n, ys)

    # Check that we have a partition
    unittest.TestCase().assertCountEqual(first + second, ys)

    # Check that our partition is the correct size
    assert len(first) == n
    assert len(second) == len(ys) - n

    # Check that every element in first is greater than or equal to
    # every element in second
    if len(second) > 0:
        bound = max(second)
        for i in first:
            assert i >= bound


@given(k_and_list())
def test_partition_partitions_a_list_with_key_function(x):
    n, ys = x
    zs = [(-y, y) for y in ys]
    first, second = ut.partition(n, zs, key=lambda x: x[1])

    # Check that we have a partition
    unittest.TestCase().assertCountEqual(first + second, zs)

    # Check that our partition is the correct size
    assert len(first) == n
    assert len(second) == len(ys) - n

    # Check that every element in first is greater than or equal to
    # every element in second
    if len(second) > 0:
        bound = max(i[1] for i in second)
        for i in first:
            assert i[1] >= bound


def test_partition_validates_n():
    with pytest.raises(ValueError):
        ut.partition(5, [1, 2, 3])
