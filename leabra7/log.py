"""Tools to log data from the network."""
import abc
import collections
from typing import Any
from typing import Dict
from typing import Iterable
from typing import List
from typing import NamedTuple
from typing import Tuple

import pandas as pd  # type: ignore

from leabra7 import events
from leabra7 import specs

WholeObs = Tuple[str, Any]
"""The observation of an entire object.

Example:

- An observation of the average activation for a layer could
  look like this: ("avg_act", 0.333)

"""

PartsObs = Dict[str, List[Any]]
"""An observation of the parts of an object, e.g. the units of a layer.

Example:

- An observation of the activation in one unit of a layer
  could look like this: {"unit": [0, 1, 2], "act": [0.2, 0.3, 0.5]}

"""


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

    def increment_time(self) -> None:
        """Increments the time counter."""
        self.time += 1

    def to_df(self) -> pd.DataFrame:
        """Returns a DataFrame containing the data in the buffer."""
        return pd.concat(self.buffer, ignore_index=True)


class ObservableMixin(metaclass=abc.ABCMeta):
    """Defines the interface required by `Logger` to record attributes.

    Classes that use this mixin should inject `whole_attrs`, and
    `parts_attrs`into into `super().__init__()`. For example:

        class ObjToLog(log.ObservableMixin):

        def __init__(self, name, *args: Any, **kwargs: Any) -> None:
            super().__init__(
                whole_attrs=["avg_act"],
                parts_attrs=["unit0_act", "unit1_act"],
                *args,
                **kwargs)

    This way, we can instantiate `ObjToLog` with
    `ObjToLog(name="obj1"), but it will have `whole_attrs` and
    `parts_attrs` attributes.

    Classes that use this mixin must also provide the `name` and `spec`
    properties.

    Args:
        * whole_attrs: The whole attributes that we can log on the object.
        * parts_attrs: The parts attributes that we can log on the object.

    Attributes:
        * whole_attrs (Set[str]): The valid whole attributes to log.
        * parts_attrs (Set[str]): The valid parts attributes to log.

    Properties:
      name (str): The name of the object.
      spec (specs.Spec): The object spec.

    """

    def __init__(self, whole_attrs: List[str], parts_attrs: List[str],
                 *args: Any, **kwargs: Any) -> None:
        self.whole_attrs = set(whole_attrs)
        self.parts_attrs = set(parts_attrs)
        # noinspection PyArgumentList
        super().__init__(*args, **kwargs)  # type: ignore

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Returns the name of the object."""

    @property
    @abc.abstractmethod
    def spec(self) -> specs.ObservableSpec:
        """Returns the object spec."""

    def validate_attr(self, attr: str) -> None:
        """Checks if an attr is valid to log.

        Args:
          attr: The attribute to check

        Raises:
          ValueError: If the attr is not a valid attribute to observe.

        """
        if attr not in self.whole_attrs and attr not in self.parts_attrs:
            raise ValueError(
                "{0} is not a valid observable attribute.".format(attr))

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

    def observe(self, attr: str) -> pd.DataFrame:
        """Observes an attribute, returning a dataframe.

        This supports the Net.observe() method.

        Args:
          attr: The attribute to observe.

        Returns:
          A Pandas Dataframe containing the observation.

        Raises:
          ValueError: If the attr is not observable.

        """
        self.validate_attr(attr)
        if attr in self.parts_attrs:
            return pd.DataFrame(self.observe_parts_attr(attr))
        name, val = self.observe_whole_attr(attr)
        return pd.DataFrame({name: (val, )})


def merge_parts_observations(observations: Iterable[PartsObs]) -> pd.DataFrame:
    """Merges parts observations together into a dataframe.

    This dataframe can then be appended to the parts log dataframe. If
    any of the observations have identical keys, then the later keys
    in the sequence will override the earlier keys in the merged
    dataframe. This should be fine because the only identical keys
    will be the parts indices (e.g. "unit": [0, 1])).

    Args:
      observations: The parts observations to merge.

    Returns:
      A dataframe containing the merged observations.

    """
    return pd.DataFrame(data=dict(collections.ChainMap(*observations)))


def merge_whole_observations(observations: Iterable[WholeObs]) -> pd.DataFrame:
    """Merges whole observations together into a dataframe.

    This dataframe can then be appended to the whole log dataframe.

    Args:
      observations: The whole observations to merge.

    Returns:
      A dataframe containing the merged observations.

    """
    return pd.DataFrame(dict(observations), index=[0])


class Logs(NamedTuple):
    """A container for the logs collected on an object.

    Attributes:
        * whole (pd.DataFrame): A DataFrame containing the logs for attributes
            logged on the whole object. Examples of these attributes for the
            layer object are "avg_act" or "fbi".
        * parts (pd.DataFrame): A DataFrame containing the logs for attributes
            logged on parts of the object. Examples of these attributes for the
            layer object are "unit_act" or "unit_net".

    """
    whole: pd.DataFrame
    parts: pd.DataFrame


class Logger(events.EventListenerMixin):
    """Records target attributes to internal buffers.

    Args:
        * target: The object from which to record attributes. It must inherit
            from `ObservableMixin`.
        * attrs: A list of attribute names to log.
        * freq: The frequency at which this logger should record.

    Attrs:
        * name (str): The name of the target object.

    Raises:
      TypeError: If event_trigger is not a type.

    """

    def __init__(self, target: ObservableMixin, attrs: Iterable[str],
                 freq: events.Frequency) -> None:
        self.target = target
        self.target_name = target.name
        self.whole_attrs = [i for i in attrs if i in target.whole_attrs]
        self.parts_attrs = [i for i in attrs if i in target.parts_attrs]
        self.whole_buffer = DataFrameBuffer()
        self.parts_buffer = DataFrameBuffer()
        self.paused = False
        self.freq = freq

    def record(self) -> None:
        """Records the attributes to an internal buffer."""
        if self.paused:
            self.whole_buffer.increment_time()
            self.parts_buffer.increment_time()
            return

        whole_observations = [
            self.target.observe_whole_attr(a) for a in self.whole_attrs
        ]
        parts_observations = [
            self.target.observe_parts_attr(a) for a in self.parts_attrs
        ]
        self.whole_buffer.append(merge_whole_observations(whole_observations))
        self.parts_buffer.append(merge_parts_observations(parts_observations))

    def to_logs(self) -> Logs:
        """Converts the internal buffer to a Logs object.

        Returns:
          A Logs object containing the contents of the internal buffers.

        """
        return Logs(
            whole=self.whole_buffer.to_df(), parts=self.parts_buffer.to_df())

    def handle(self, event: events.Event) -> None:
        """Overrides `events.EventListnerMixin.handle()`."""
        if isinstance(event, self.freq.end_event_type):
            self.record()
        elif isinstance(event, events.PauseLogging):
            if event.freq == self.freq:
                self.paused = True
        elif isinstance(event, events.ResumeLogging):
            if event.freq == self.freq:
                self.paused = False
