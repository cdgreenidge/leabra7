"""Test log.py"""
from typing import Any
from typing import List
from typing import Dict

import pandas as pd  # type: ignore

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
    print(expected)
    print(dfb.to_df())
    assert dfb.to_df().equals(expected)


# Test log.Logger
class ObjToLog(log.ObservableMixin):
    """A dummy class with which to test logging."""

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
