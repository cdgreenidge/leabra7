"""Tools to log data from the network."""
import collections
import enum
from typing import Any
from typing import Callable
from typing import Dict  # noqa pylint: disable=W0611
from typing import DefaultDict
from typing import List
from typing import Tuple
from typing import TypeVar

import pandas as pd  # type: ignore

from leabra7 import layer
from leabra7 import unit  # noqa pylint: disable=W0611

Attr = str
"""In our logs, we record attributes as strings."""

AttrObs = Tuple[Attr, Any]
"""A attribute observation is a tuple with an attribute name, and the
value of that attribute, e.g. ("unit_act", 0.3)."""

T = TypeVar('T')
AttrReader = Callable[[T], AttrObs]
"""A attribute reader is a function that takes an object of type T
and returns an attribute observation."""

ObjObs = List[AttrObs]
"""An observation of an object is just a list of observations of that
object's attributes, taken at one point in time."""

ObjReader = Callable[[T], ObjObs]
"""An object reader is a function that takes an object of type T and returns a
object observation."""


class DataFrameBuffer:
    """A buffer of dataframe rows (log entries) that are waiting to be
    appended to a target dataframe."""

    def __init__(self) -> None:
        self.length = 0

        def new() -> List[Any]:
            return [None] * self.length

        self.buf = collections.defaultdict(new)  # type: Dict[str, List[Any]]

    def append(self, row: ObjObs) -> None:
        """Appends a row to the dataframe buffer. If there are new keys in the
        row, it fills in the beginning of the "column" with None values. If the
        row does not contain all the columns already in the buffer, it appends
        None values to those columns. For best results, make sure you add rows
        with the same keys every time!"""
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


ObjName = str
"""Every object has a unique name."""


@enum.unique
class Frequency(enum.Enum):
    """Frequencies at which we can log objects."""
    CYCLE = "cycle"
    # In the future: TRIAL, EPOCH, BATCH


BufferKey = Tuple[ObjName, Frequency]
"""A BufferKey uniquely identifies the DataFrameBuffer where we should store
an object observation: there is one buffer for every object and frequency."""


def simplify(key: BufferKey) -> Tuple[str, str]:
    """Simplifies a BufferKey so that it is just a tuple of strings. This is
    used when we're exporting data, so that users don't have to dig around for
    the frequency enum, but can just write log[("layer1", "cycle")]."""
    name, freq = key
    return (name, freq.value)


Logger = Callable[[T], Tuple[BufferKey, ObjObs]]
"""A Logger is a function that, when applied to an object of type T, returns a
tuple with a BufferKey and an object observation."""

Logs = DefaultDict[BufferKey, DataFrameBuffer]
"""We store the logs in a dictionary of DataFrameBuffers. The DefaultDict
factory function should be `DataFrameBuffer`."""


def naive_attrreader(attr: str) -> AttrReader[T]:
    """Reads an attribute from an object."""

    def reader(obj: T) -> AttrObs:
        return (attr, getattr(obj, attr))

    return reader


def unit_attrreader(attr: str) -> AttrReader["unit.Unit"]:
    """Reads an attribute from a unit. Raises ValueError if the attribute
    is invalid."""
    valid_attrs = ("g_e", "act", "act_m", "act_p", "act_nd", "spike", "adapt",
                   "i_net", "v_m", "i_net_eq", "v_m_eq", "avg_ss", "avg_s",
                   "avg_m", "avg_l")
    if attr not in valid_attrs:
        raise ValueError("Invalid unit attribute {0}.".format(attr))

    return naive_attrreader(attr)


def to_layer_attrreader(reader: AttrReader["unit.Unit"],
                        n: int) -> AttrReader["layer.Layer"]:
    """Transforms a unit attrreader to a layer attrreader by applying
    it to the nth unit in a layer. Prepends `unitn_` to the description
    in the attribute observation so we can tell which unit it came from."""

    def lr_reader(lr: "layer.Layer") -> AttrObs:
        desc, val = reader(lr.units[n])
        return ("unit{0}_{1}".format(n, desc), val)

    return lr_reader


def to_layer_attrreaders(reader: AttrReader["unit.Unit"],
                         lr_size: int) -> List[AttrReader["layer.Layer"]]:
    """Transforms a unit varobserver to a list of layer varobservers that
    collectively perform the original unit observation for every unit
    in the layer (as many as the layer size). The nth layer observer
    observes the nth unit in the layer, and prepends `unitn_` to the
    original variable name."""
    return [to_layer_attrreader(reader, n) for n in range(lr_size)]


def trim_unit_prefix(unitattr: str) -> str:
    """Trims the `unit_` prefix from a string."""
    return unitattr[5:]  # unit_ is 5 chars long


def unit_attrs(attrs: List[str]) -> List[str]:
    """Filters out the unit variables (those prefixed with `unit_`)
    from the layer variables (e.g. mean_act) in a list of strings. Trims off
    the `unit_` prefix from each unit attribute."""
    return [trim_unit_prefix(x) for x in attrs if x.startswith("unit_")]


def naive_reader(attrs: List[str]) -> ObjReader[T]:
    """Makes a reader that reads attributes from an object--nothing special."""
    readers = [naive_attrreader(i) for i in attrs]  # type: List[AttrReader[T]]

    def f(obj: T) -> ObjObs:
        return [g(obj) for g in readers]

    return f


def layer_reader(attrs: List[str], lr_size: int) -> ObjReader["layer.Layer"]:
    """Makes a layer reader, given a list of attributes to log and
    the layer size."""
    readers = []
    for i in unit_attrs(attrs):
        readers.extend(to_layer_attrreaders(unit_attrreader(i), lr_size))

    def f(lr: layer.Layer) -> ObjObs:
        return [f(lr) for f in readers]

    return f


def layer_logger(attrs: List[str], freq: Frequency,
                 lr_size: int) -> Logger["layer.Layer"]:
    """Given a list of attributes and a frequency, returns a logger for the
    layer."""
    reader = layer_reader(attrs, lr_size)

    def logger(lr: "layer.Layer") -> Tuple[BufferKey, ObjObs]:
        obs = reader(lr)
        key = (lr.name, freq)
        return (key, obs)

    return logger
