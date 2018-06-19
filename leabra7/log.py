"""Tools to log data from the network."""
import abc
import functools
import itertools
from typing import Any
from typing import Dict
from typing import Iterable
from typing import List
from typing import NamedTuple
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

    Classes that use this mixin should inject `name`, `whole_attrs`, and
    `parts_attrs`into into `super().__init__()`. For example:

        class ObjToLog(log.ObservableMixin):

        def __init__(self, name, *args: Any, **kwargs: Any) -> None:
            super().__init__(
                name=name,
                whole_attrs=["avg_act"],
                parts_attrs=["unit0_act", "unit1_act"],
                *args,
                **kwargs)

    This way, we can instantiate `ObjToLog` with
    `ObjToLog(name="obj1"), but it will have `whole_attrs` and
    `parts_attrs` attributes.

    Args:
      name: The name of the object.
      whole_attrs: The whole attributes that we can log on the object.
      parts_attrs: The parts attributes that we can log on the object.

    Attributes:
      name (str): The name of the object.
      whole_attrs (Set[str]): The valid whole attributes to log.
      parts_attrs (Set[str]): The valid parts attributes to log.

    """

    def __init__(self, name: str, whole_attrs: List[str],
                 parts_attrs: List[str], *args: Any, **kwargs: Any) -> None:
        self.name = name
        self.whole_attrs = set(whole_attrs)
        self.parts_attrs = set(parts_attrs)
        # noinspection PyArgumentList
        super().__init__(*args, **kwargs)  # type: ignore

    def validate_attr(self, attr: str) -> None:
        """Checks if an attr is valid to log.

        Args:
          attr: The attribute to check

        Raises:
          ValueError: If the attr is not a valid attribute to observe.

        """
        if attr not in self.whole_attrs and attr not in self.parts_attrs:
            raise ValueError("{0} is not a valid observable attribute.")

    def observe_whole_attr(self, attr: str) -> WholeObs:
        """Observes a whole attribute.

        Args:
          attr: The attribute to observe.

        Returns:
          A WholeObs (`Tuple[str, Any]`) containing the attribute name and the
          value of the attribute.

        Raises:
          ValueError: If the attr is not a whole attr.

        """
        if attr not in self.whole_attrs:
            raise ValueError("{0} is not a whole attr.".format(attr))
        return (attr, getattr(self, attr))

    @abc.abstractmethod
    def observe_parts_attr(self, attr: str) -> PartsObs:
        """Observes a parts attribute.

        Args:
          attr: The attribute to observe.

        Returns:
          A PartsObs (`Dict[str, List[any]]`) containing the attribute name and
          the values of the attribute for each part.

        Raises:
          ValueError: If the attr is not a parts attribute.

        """

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


class Logs(NamedTuple):
    """A container for the logs collected on an object.

    Attributes:
      whole (pd.DataFrame): A DataFrame containing the logs for attributes
        logged on the whole object. Examples of these attributes for the layer
        object are "avg_act" or "fbi".
      parts (pd.DataFrame): A DataFrame containing the logs for attributes
        logged on parts of the object. Examples of these attributes for the
        layer object are "unit_act" or "unit_net".

    """
    whole: pd.DataFrame
    parts: pd.DataFrame


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
