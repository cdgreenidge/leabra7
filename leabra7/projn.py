"""A connection between layers."""
import itertools
import random
from typing import Iterable
from typing import List
from typing import TypeVar

from leabra7 import specs
from leabra7 import layer
from leabra7 import unit


class Conn:
    """A connection links two units/neurons.

    Args:
        name: The name of the connection.
        pre: The presynaptic, or sending, unit.
        post: The postsynaptic, or receiving, unit.
        spec: The connection specification. If none is provided, the default
            spec will be used.

    """

    def __init__(self,
                 name: str,
                 pre: unit.Unit,
                 post: unit.Unit,
                 spec: specs.ConnSpec = None) -> None:
        self.name = name
        self.pre = pre
        self.post = post

        if spec is None:
            self.spec = specs.ConnSpec()
        else:
            self.spec = spec

        self.wt = self.spec.dist.draw()


def make_full_conn_list(proj_name: str, pre_units: Iterable[unit.Unit],
                        post_units: Iterable[unit.Unit],
                        spec: specs.ConnSpec) -> List[Conn]:
    """Constructs the connections needed for a full projection.

    In a full projection, every unit in the sending layer is connected to
    every unit in the receiving layer.

    Args:
        proj_name: The name of the projection. Used to generate the connection
            names.
        pre_units: The sending layer's units.
        post_units: The receiving layer's units.
        spec: The spec to use for every connection.

    Returns:
        A full list of connections from the sending layer's units to the
            receiving layer's units.

    """

    def name(conn_number: int) -> str:
        """Generates a name for each connection."""
        return "{0}_conn{1}".format(proj_name, conn_number)

    connections = []
    num = 0
    for i in pre_units:
        for j in post_units:
            connections.append(Conn(name(num), i, j, spec))
            num += 1
    return connections


T = TypeVar('T')


def mask(xs: Iterable[T], xs_mask: Iterable[bool]) -> List[T]:
    """Filters an iterable using a boolean mask.

    Args:
        xs: The iterable to filter.
        xs_mask: The boolean mask. If it is shorter than xs, it will be tiled.
            If it is longer than xs, it will be truncated.

    Returns:
        A list containing the values of xs for which mask is true.

    """
    return [x for x, m in zip(xs, itertools.cycle(xs_mask)) if m]


class Projn:
    """A projection links two layers. It is a bundle of connections.

    Args:
        name: The name of the projection.
        pre: The sending layer.
        post: The receiving layer.
        spec: The projection specification. If none is provided, the default
            spec will be used.

    """

    def __init__(self,
                 name: str,
                 pre: layer.Layer,
                 post: layer.Layer,
                 spec: specs.ProjnSpec = None) -> None:
        self.name = name
        self.pre = pre
        self.post = post

        if spec is None:
            self.spec = specs.ProjnSpec()
        else:
            self.spec = spec

        conn_spec = specs.ConnSpec(dist=self.spec.dist)

        # Only create the projection between the units selected by the masks
        pre_units = mask(self.pre.units, self.spec.pre_mask)
        post_units = mask(self.post.units, self.spec.post_mask)

        self.conns = make_full_conn_list(name, pre_units, post_units,
                                         conn_spec)

        # Enforce sparsity
        # This could be done more functionally (better) but we're going to
        # throw all this code out soon anyway, in favor of connection groups.
        # This is closer to the code we'll have then
        n = len(self.conns)
        num_to_disable = int((1 - self.spec.sparsity) * n)
        for i in random.sample(range(n), num_to_disable):
            self.conns[i].wt = 0

    def flush(self) -> None:
        """Propagates sending layer activation to the recieving layer.

        Separating this step from the activation and firing of the sending
        layer makes it easier to compute the net input scaling factor.

        """
        for c in self.conns:
            scale_eff = 1.0  # Currently netin scaling is not implemented
            c.post.add_input(scale_eff * c.pre.act * c.wt)
