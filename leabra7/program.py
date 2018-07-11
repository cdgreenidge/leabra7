"""Components for the AST that a leabra7 network can run."""
import abc


class Node:
    pass


class AtomicEvent():
    """An atomic event is an event that the network can execute directly.

    It's ok to use isinstance() to check if what kind of AtomicEvent you get,
    because we're basically mimicing an algebraic datatype.

    """
    pass


class Cycle(Node, AtomicEvent):
    """The event that cycles the network."""
    pass


class BeginPlusPhase(Node, AtomicEvent):
    """The event that begins the plus phase in a trial."""
    pass


class EndPlusPhase(Node, AtomicEvent):
    """The event that ends the plus phase in a trial."""
    pass


class BeginMinusPhase(Node, AtomicEvent):
    """The event that begins the minus phase in a trial."""
    pass


class EndMinusPhase(Node, AtomicEvent):
    """The event that ends the minus phase in a trial."""
    pass


class ForceLayer(Node, AtomicEvent):
    """The event that forces a layer."""
    pass


class EventListenerMixin(metaclass=abc.ABCMeta):
    """Defines an interface for handling network events.

    This must be implemented by every object in the network.

    """

    @abc.abstractmethod
    def handle(self, event: AtomicEvent) -> None:
        """When invoked, does any processing triggered by the event."""
