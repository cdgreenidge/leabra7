"""Tools to log data from the network."""
import collections
from typing import Any
from typing import Dict  # noqa pylint: disable=W0611
from typing import List
from typing import Tuple

import pandas as pd  # type: ignore

Attr = str
"""In our logs, we record attributes as strings."""

AttrObs = Tuple[Attr, Any]
"""
An observation of one of an object's attributes.

It is a tuple containing the attribute name, and the value of that attribute,
e.g. ("unit_act", 0.3).

"""

ObjObs = List[AttrObs]
"""An observation of an object (i.e. many of the object's attributes."""


class DataFrameBuffer:
    """A buffer of dataframe rows.

    This gives us constant time append. When we're done collecting rows, we
    can condense them all into a dataframe.
    """

    def __init__(self) -> None:
        self.length = 0

        def new() -> List[Any]:
            return [None] * self.length

        self.buf = collections.defaultdict(new)  # type: Dict[str, List[Any]]

    def append(self, row: ObjObs) -> None:
        """Appends a row to the dataframe buffer.

        If the row contains an attribute that hasn't been logged before, the
        column is filled in with Nones for previous time steps. If the row
        doesn't contain all the attributes in the dataframe, the missing
        attributes take None for their values.

        Args:
            row: The list of attribute observations to append to the buffer.

        """
        for k, v in row:
            self.buf[k].append(v)
        self.length += 1
        self._pad()

    def to_df(self) -> pd.DataFrame:
        """Returns a DataFrame containing the data in the buffer."""
        return pd.DataFrame.from_dict(self.buf)

    def _pad(self) -> None:
        """Pads all the columns in the buffer with Nones until they are the
        same length."""
        for _, v in self.buf.items():
            while len(v) < self.length:
                v.append(None)
