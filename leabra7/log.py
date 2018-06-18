"""Tools to log data from the network."""
import abc
import functools
import itertools
from typing import Any
from typing import Dict
from typing import List
from typing import Iterable
from typing import Tuple

import pandas as pd  # type: ignore

Obs = Dict[str, Any]
"""An observation is a specially-formatted dict.

Here are some examples:

- An observation of 'unit0_act' for a layer could look
  like this: {"unit": 0, "act": 0.5}.

- An observation of 'avg_act' for a layer
  could look like this: {"avg_act": 0.333}.

"""

## TODO: Review comments for the WholeObs and PartsObs

WholeObs = Tuple[str, Any]
""" An observation of whole attributes is a tuple.

It stores the observations for an entire layer.

Here is an example:

- An observation of the average activation for a layer could
  look like this: ("avg_act", 0.333)

"""

PartsObs = Dict[str, List[Any]]
""" An observation of partial attributes is a dict.

It stores observations for units of a layer.

Here is an example:

- An observation of the activation in one unit of a layer
  could look like this: {"unit": [0, 1, 2], "act": [0.2, 0.3, 0.5]}

"""


def whole_observations_to_dict(observations: List[WholeObs]) -> Dict[str, Any]:
    """ Converts a list of whole observations into dictionary.

    Args:
        observations: The list of whole observations.

    Returns:
        A dictionary containing the keys and values for the whole observations.

    """

    observation_keys = [obs[0] for obs in observations]

    assert len(observation_keys) == len(set(observation_keys))

    return dict(observations)


class DataFrameBuffer:
    """A buffer of dataframe records.

    This gives us constant time append. When we're done collecting rows, we
    can condense them all into a dataframe.
    """

    def __init__(self) -> None:
        self.time = 0
        self.buffer: List[pd.DataFrame] = []

    def append(self, record: pd.DataFrame) -> None:
        """Appends a record to the dataframe buffer.

        A "time" column is added to each record.

        Args:
            record: A dataframe containing some rows in the output log,
                for a single time step.

        """
        df: pd.DataFrame = record.copy()
        df["time"] = self.time
        self.buffer.append(df)
        self.time += 1

    def to_df(self) -> pd.DataFrame:
        """Returns a DataFrame containing the data in the buffer."""
        return pd.concat(self.buffer, ignore_index=True)


class ObservableMixin(metaclass=abc.ABCMeta):
    """Defines the interface required by `Logger` to record attributes.

    Attributes:
        name (str): The name of the object.

    """

    def __init__(self, name: str, *args: Any, **kwargs: Any) -> None:
        self.name = name
        # noinspection PyArgumentList
        super().__init__(*args, **kwargs)  # type: ignore

    @property
    @abc.abstractmethod
    def whole_attrs(self) -> List[str]:
        """The attributes one can log on the whole object."""

    @property
    @abc.abstractmethod
    def parts_attrs(self) -> List[str]:
        """The attributes one can log on the object's parts."""

    @abc.abstractmethod
    def observe(self, attr: str) -> List[Obs]:
        """Observes an attribute on this object.

        Args:
            attr: The attribute to observe. The name "attribute" is a bit of a
                misnomer, because it could be something computed on-the-fly.

        Returns:
            A list of observations from the attribute. We need a list because
            some attributes produce more than one observation,
            e.g. observing "unit_act" on a layer.

        Raises:
            ValueError: if attr is not a loggable attribute.

        """


def merge_observations(observations: Iterable[Obs]) -> pd.DataFrame:
    """Merges a list of observations together.

    This is done using full outer joins.

    Args:
        observations: The list of observations.

    Returns:
        A DataFrame containing the merged observations.

    """

    def merge(df: pd.DataFrame, obs: Obs) -> pd.DataFrame:
        """Merges two observations."""
        df2 = pd.DataFrame(obs, index=[0])
        if df.columns.intersection(df2.columns).empty:
            return pd.concat((df, df2), sort=True)
        return pd.merge(df, df2, how="outer", copy=False)

    return functools.reduce(merge, observations, pd.DataFrame())


class Logger:
    """Records target attributes to an internal buffer.

    Args:
        target: The object from which to record attributes. It must inherit
            from `ObservableMixin`.
        attrs: A list of attribute names to log.

    Attrs:
        name (str): The name of the target object.

    """

    def __init__(self, target: ObservableMixin, attrs: Iterable[str]) -> None:
        self.target = target
        self.target_name = target.name
        self.attrs = attrs
        self.buffer = DataFrameBuffer()

    def record(self) -> None:
        """Records the attributes to an internal buffer."""
        observations = itertools.chain.from_iterable(
            self.target.observe(a) for a in self.attrs)
        self.buffer.append(merge_observations(observations))

    def to_df(self) -> pd.DataFrame:
        """Converts the internal buffer to a dataframe.

        Returns:
            A dataframe containing the contents of the internal buffer. The
            columns names are the attribute names, and each row contains the
            observations for one call of the record() method.

        """
        return self.buffer.to_df()
