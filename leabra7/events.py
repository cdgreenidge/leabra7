"""Components for the AST that a leabra7 network can run."""
import abc
import inspect
from typing import Dict
from typing import Sequence
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


class BeginPhase(Event):
    """The event that ends a phase."""

    def __init__(self, phase: str) -> None:
        self.phase = phase


class EndPhase(Event):
    """The event that begins a phase."""

    def __init__(self, phase: str) -> None:
        self.phase = phase


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
      freq_name: The name of the frequency for which to pause logging.

    Raises:
      ValueError: If no frequency exists with name `freq_name`.

    """

    def __init__(self, freq_name: str) -> None:
        self.freq = Frequency.from_name(freq_name)


class ResumeLogging(Event):
    """The event that resumes logging in the network.

    Args:
      freq_name: The name of the frequency for which to resume logging.

    Raises:
      ValueError: If no frequency exists with name `freq_name`.

    """

    def __init__(self, freq_name: str) -> None:
        self.freq = Frequency.from_name(freq_name)


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
      name: The name of the node.

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
      layer_name: The name of the layer to unclamp.
      name: The name of the node.

    """

    def __init__(self, layer_name: str) -> None:
        self.layer_name = layer_name


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

    @classmethod
    def names(cls) -> Sequence[str]:
        """Returns the names of all defined frequencies."""
        return tuple(cls.registry.keys())

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


class Phase():
    """Defines network phases.

    Args:
        begin_event_type: Event type that marks the beginning of a phase.
        end_event_type: Event type that marks the end of a phase.

    """
    # Stores a reference to each created frequency object, keyed by name
    registry: Dict[str, "Phase"] = {}

    name: str
    begin_event_type: Type[Event]
    end_event_type: Type[Event]

    def __init__(self, name: str) -> None:
        self.name = name
        self.begin_event = BeginPhase(name)
        self.end_event = EndPhase(name)
        Phase.registry[name] = self

    @classmethod
    def names(cls) -> Sequence[str]:
        """Returns the names of all defined phases."""
        return tuple(cls.registry.keys())

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

    @classmethod
    def phases(cls) -> Sequence["Phase"]:
        """Returns all the defines phases."""
        return tuple(cls.registry.values())


NonePhase = Phase(name="none")
PlusPhase = Phase(name="plus")
MinusPhase = Phase(name="minus")

ThetaTrough = Phase(name="theta_trough")
ThetaPeak = Phase(name="theta_peak")
ThetaPlus = Phase(name="theta_plus")


class EventListenerMixin(metaclass=abc.ABCMeta):
    """Defines an interface for handling network events.

    This must be implemented by every object in the network.

    """

    @abc.abstractmethod
    def handle(self, event: Event) -> None:
        """When invoked, does any processing triggered by the event."""
