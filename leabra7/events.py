"""Components for the AST that a leabra7 network can run."""
import abc
from typing import Sequence


class Event():
    """An atomic event is an event that the network can execute directly.

    It's ok to use isinstance() to check if what kind of Event you get,
    because we're basically mimicing an algebraic datatype.

    """
    pass


class Cycle(Event):
    """The event that cycles the network."""
    pass


class BeginPlusPhase(Event):
    """The event that begins the plus phase in a trial."""
    pass


class EndPlusPhase(Event):
    """The event that ends the plus phase in a trial."""
    pass


class BeginMinusPhase(Event):
    """The event that begins the minus phase in a trial."""
    pass


class EndMinusPhase(Event):
    """The event that ends the minus phase in a trial."""
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


class PauseCycleLog(Event):
    """The even that pauses cycle logging."""
    pass


class PauseTrialLog(Event):
    """The even that pauses trial logging."""
    pass


class PauseEpochLog(Event):
    """The even that pauses epoch logging."""
    pass


class PauseBatchLog(Event):
    """The even that pauses batch logging."""
    pass


class ResumeCycleLog(Event):
    """The even that resumes cycle logging."""
    pass


class ResumeTrialLog(Event):
    """The even that resumes trial logging."""
    pass


class ResumeEpochLog(Event):
    """The even that resumes epoch logging."""
    pass


class ResumeBatchLog(Event):
    """The even that resumes batch logging."""
    pass


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


class EventListenerMixin(metaclass=abc.ABCMeta):
    """Defines an interface for handling network events.

    This must be implemented by every object in the network.

    """

    @abc.abstractmethod
    def handle(self, event: Event) -> None:
        """When invoked, does any processing triggered by the event."""
