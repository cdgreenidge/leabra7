"""Components for the AST that a leabra7 network can run."""
import abc
from enum import auto
from enum import Enum
import inspect
from typing import Dict
from typing import Set
from typing import Sequence
from typing import Tuple
from typing import Type


class Event():
    """An atomic event is an event that the network can execute directly.

    It's ok to use isinstance() to check if what kind of Event you get,
    because we're basically mimicing an algebraic datatype.

    """
    pass


class Cycle(Event):
    """The event that cycles the network."""
    pass


class EndTrial(Event):
    """The event that signals the end of a trial."""
    pass


class EndEpoch(Event):
    """The event that signals the end of an epoch."""
    pass


class EndBatch(Event):
    """The event that signals the end of a batch."""
    pass


class PauseLogging(Event):
    """The event that pauses logging in the network.

    Args:
      freq_names: The names of the frequencies for which to pause logging.

    Raises:
      ValueError: If no frequency exists with name in freq_names.

    """

    def __init__(self, *freq_names: str) -> None:
        for name in freq_names:
            Frequency.from_name(name)

        self.freqs: Set["Frequency"] = set()

        for name in freq_names:
            self.freqs.add(Frequency.from_name(name))


class ResumeLogging(Event):
    """The event that resumes logging in the network.

    Args:
      freq_names: The names of the frequencies for which to resume logging.

    Raises:
      ValueError: If no frequency exists with name in freq_names.

    """

    def __init__(self, *freq_names: str) -> None:
        for name in freq_names:
            Frequency.from_name(name)

        self.freqs: Set["Frequency"] = set()

        for name in freq_names:
            self.freqs.add(Frequency.from_name(name))


class InhibitProjns(Event):
    """The event that inhibits projections in the network.

    Args:
        projn_names: The names of projections to inhibit.

    """

    def __init__(self, *projn_names: str) -> None:
        self.projn_names = projn_names


class UninhibitProjns(Event):
    """The event that uninhibits projections in the network.

    Args:
        projn_names: The names of projections to uninhibit.

    """

    def __init__(self, *projn_names: str) -> None:
        self.projn_names = projn_names


class HardClamp(Event):
    """The event that hard clamps a layer.

    Args:
      layer_name: The name of the layer to hard clamp.
      acts: A sequence of the activations to clamp the layer to. If there are
        fewer values than the number of units in the layer, it will be tiled.

    Raises:
      ValueError: If any value of acts is outside the range [0, 1].

    """

    def __init__(self, layer_name: str, acts: Sequence[float]) -> None:
        self.layer_name = layer_name
        if not all(0 <= i <= 1 for i in acts):
            raise ValueError("All values of acts must be in [0, 1].")
        self.acts = acts


class Unclamp(Event):
    """The event that unclamps a layer.

    Args:
      layer_names: The names of the layers to unclamp.

    """

    def __init__(self, *layer_names: str) -> None:
        self.layer_names = layer_names


class Learn(Event):
    """The event that triggers learning in projections."""
    pass


class Frequency():
    """Defines event frequencies.

    Args:
      name: The name of the frequency.
      end_event_type: The event that marks the end of the frequency period.

    Raises:
      TypeError: if end_event_type is not a type (i.e. class variable)

    """
    # Stores a reference to each created frequency object, keyed by name
    registry: Dict[str, "Frequency"] = {}

    name: str
    end_event_type: Type[Event]

    def __init__(self, name: str, end_event_type: Type[Event]) -> None:
        if not inspect.isclass(end_event_type):
            raise TypeError("end_event_type must be a class variable.")
        self.name = name
        self.end_event_type = end_event_type
        Frequency.registry[name] = self

    def __key(self) -> Tuple[str, Type[Event]]:
        return (self.name, self.end_event_type)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Frequency):
            return self.__key() == other.__key()  #pylint: disable=protected-access
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.__key())

    @classmethod
    def names(cls) -> Sequence[str]:
        """Returns the names of all defined frequencies."""
        return tuple(cls.registry.keys())

    @classmethod
    def freqs(cls) -> Sequence["Frequency"]:
        """Returns all defined frequencies."""
        return tuple(cls.registry.values())

    @classmethod
    def from_name(cls, freq_name: str) -> "Frequency":
        """Gets a frequency object by its name.

        Args:
          freq_name: The name of the frequency.

        Raises:
          ValueError: If no frequency exists with name `freq_name`.

        """
        try:
            return cls.registry[freq_name]
        except KeyError:
            raise ValueError(
                "No frequency with name {0} exists.".format(freq_name))


CycleFreq = Frequency(name="cycle", end_event_type=Cycle)
TrialFreq = Frequency(name="trial", end_event_type=EndTrial)
EpochFreq = Frequency(name="epoch", end_event_type=EndEpoch)
BatchFreq = Frequency(name="batch", end_event_type=EndBatch)


class PhaseType(Enum):
    PLUS = auto()
    MINUS = auto()
    NONE = auto()

    @classmethod
    def get_phase_type(cls, name: str) -> "PhaseType":
        if name == "plus":
            return PhaseType.PLUS
        elif name == "minus":
            return PhaseType.MINUS
        elif name == "none":
            return PhaseType.NONE

        raise ValueError("""Invalid: Phase type '{0}' is not one of
            'plus', 'minus', or 'none'""".format(name))


class Phase():
    """Defines network phases.

    Args:
        name: Name of phase.
        phase_type: Type of phase ('plus', 'minus', or 'none').

    Raises:
        ValueError: If phase_type not one of 'plus', 'minus', or 'none'.

    """
    # Stores a reference to each created frequency object, keyed by name
    registry: Dict[str, "Phase"] = {}

    name: str
    begin_event_type: Type[Event]
    end_event_type: Type[Event]

    def __init__(self, name: str, phase_type: str) -> None:
        self.name = name
        self.type = PhaseType.get_phase_type(phase_type)
        Phase.registry[name] = self

    def __key(self) -> Tuple[str, PhaseType]:
        return (self.name, self.type)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Phase):
            return self.__key() == other.__key()  #pylint: disable=protected-access
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.__key())

    def get_name(self) -> str:
        return self.name

    @classmethod
    def names(cls) -> Sequence[str]:
        """Returns the names of all defined phases."""
        return tuple(cls.registry.keys())

    @classmethod
    def phases(cls) -> Sequence["Phase"]:
        """Returns all defined phases."""
        return tuple(cls.registry.values())

    @classmethod
    def from_name(cls, phase_name: str) -> "Phase":
        """Gets a phase object by its name.

        Args:
          phase_name: The name of the phase.

        Raises:
          ValueError: If no phase exists with name `phase_name`.

        """
        try:
            return cls.registry[phase_name]
        except KeyError:
            raise ValueError(
                "No phase with name {0} exists.".format(phase_name))


NonePhase = Phase(name="none", phase_type="none")
PlusPhase = Phase(name="plus", phase_type="plus")
MinusPhase = Phase(name="minus", phase_type="minus")


class BeginPhase(Event):
    """The event that ends a phase."""

    def __init__(self, phase: Phase) -> None:
        self.phase = phase


class EndPhase(Event):
    """The event that begins a phase."""

    def __init__(self, phase: Phase) -> None:
        self.phase = phase


class EventListenerMixin(metaclass=abc.ABCMeta):
    """Defines an interface for handling network events.

    This must be implemented by every object in the network.

    """

    @abc.abstractmethod
    def handle(self, event: Event) -> None:
        """When invoked, does any processing triggered by the event."""
