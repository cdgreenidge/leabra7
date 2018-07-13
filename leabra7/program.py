"""Components for the AST that a leabra7 network can run."""
import abc
from typing import Any
from typing import Iterable
from typing import Sequence
import itertools
import uuid


class Node:
    """A node in a rooted, directed tree.

    Args:
      children: An iterable of the children of the node.
      name: The name of the node. Defaults to a unique auto-generated string.

    """

    def __init__(self, children: Sequence["Node"] = (),
                 name: str = None) -> None:
        try:
            iter(children)
        except TypeError:
            raise ValueError("children argument must be iterable.")

        if name is None:
            self.name = uuid.uuid4().hex
        else:
            self.name = name

        self.children = children

    def preorder(self) -> Iterable["Node"]:
        """Returns a DFS preordering of the node's children.

        Children are ordered left-to-right. We also assume that the graph
        is a tree (so cycles will cause an infinite loop).

        """
        stack = [self]
        while stack:
            n = stack.pop()
            stack.extend(reversed(n.children))
            yield n

    def atomic_stream(self) -> Iterable["AtomicEvent"]:
        """Returns a stream of the node's children that are atomic events.

        The atomic events are ordered using a L-R DFS preodering.

        """
        return (i for i in self.preorder() if isinstance(i, AtomicEvent))


class Program(Node):
    """A program is a tree of events that can be executed by the network."""
    pass


class Loop(Node):
    """A loop is simply a node that loops its children.

    Args:
      children: The node children.
      num_iter: The number of iterations to loop for.
      name: The name of the node.

    Raises:
      ValueError: if num_iter is less than 1.

    """

    def __init__(self,
                 children: Sequence["Node"] = (),
                 num_iter: int = 1,
                 name: str = None) -> None:
        if num_iter < 1:
            raise ValueError("num_iter must be >= 1.")

        looped = tuple(
            itertools.islice(
                itertools.cycle(children), num_iter * len(children)))
        super().__init__(children=looped, name=name)


class AtomicEvent(Node):
    """An atomic event is an event that the network can execute directly.

    Because the network can execute it directly, it cannot have child nodes.

    It's ok to use isinstance() to check if what kind of AtomicEvent you get,
    because we're basically mimicing an algebraic datatype.

    Raises:
      ValueError: if children is not an empty sequence.

    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self.children:
            raise ValueError("Atomic events cannot have children.")


class Cycle(AtomicEvent):
    """The event that cycles the network."""
    pass


class BeginPlusPhase(AtomicEvent):
    """The event that begins the plus phase in a trial."""
    pass


class EndPlusPhase(AtomicEvent):
    """The event that ends the plus phase in a trial."""
    pass


class BeginMinusPhase(AtomicEvent):
    """The event that begins the minus phase in a trial."""
    pass


class EndMinusPhase(AtomicEvent):
    """The event that ends the minus phase in a trial."""
    pass


class HardClamp(AtomicEvent):
    """The event that hard clamps a layer.

    Args:
      layer_name: The name of the layer to hard clamp.
      acts: An iterable of the activations to clamp the layer to. If there are
        fewer values than the number of units in the layer, it will be tiled.
      name: The name of the node.

    Raises:
      ValueError: If any value of acts is outside the range [0, 1].

    """

    def __init__(self,
                 layer_name: str,
                 acts: Sequence[float],
                 name: str = None) -> None:
        self.layer_name = layer_name

        if not all(0 <= i <= 1 for i in acts):
            raise ValueError("All values of acts must be in [0, 1].")
        self.acts = acts

        super().__init__(name=name)


class EventListenerMixin(metaclass=abc.ABCMeta):
    """Defines an interface for handling network events.

    This must be implemented by every object in the network.

    """

    @abc.abstractmethod
    def handle(self, event: AtomicEvent) -> None:
        """When invoked, does any processing triggered by the event."""
