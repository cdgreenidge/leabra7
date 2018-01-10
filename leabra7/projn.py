"""A connection between layers."""
from typing import List

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

        self.wt = 0.5


def make_full_conn_list(proj_name: str, pre: layer.Layer, post: layer.Layer,
                        spec: specs.ConnSpec) -> List[Conn]:
    """Constructs the connections needed for a full projection.

    In a full projection, every unit in the sending layer is connected to
    every unit in the receiving layer.

    Args:
        proj_name: The name of the projection. Used to generate the connection
            names.
        pre: The sending layer.
        post: The receiving layer.
        spec: The spec to use for every connection.

    Returns:
        A full list of connections from the sending to the receiving layer.

    """

    def name(conn_number: int) -> str:
        """Generates a name for each connection."""
        return "{0}_conn{1}".format(proj_name, conn_number)

    connections = []
    num = 0
    for i in pre.units:
        for j in post.units:
            connections.append(Conn(name(num), i, j, spec))
            num += 1
    return connections


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

        self.conns = make_full_conn_list(name, pre, post, specs.ConnSpec())

    def flush(self) -> None:
        """Propagates sending layer activation to the recieving layer.

        Separating this step from the activation and firing of the sending
        layer makes it easier to compute the net input scaling factor.

        """
        for c in self.conns:
            scale_eff = 1.0  # Currently netin scaling is not implemented
            c.post.add_input(scale_eff * c.pre.act * c.wt)
