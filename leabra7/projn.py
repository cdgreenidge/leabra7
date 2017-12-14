"""A connection between layers."""
from typing import List

from leabra7 import specs
from leabra7 import layer
from leabra7 import unit


class Conn:
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
    def name(conn_number: int) -> str:
        return "{0}_conn{1}".format(proj_name, conn_number)

    connections = []
    num = 0
    for i in pre.units:
        for j in post.units:
            connections.append(Conn(name(num), i, j, spec))
            num += 1
    return connections


class Projn:
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
        for c in self.conns:
            scale_eff = 1.0  # Currently netin scaling is not implemented
            c.post.add_input(scale_eff * c.pre.act * c.wt)
