"""A network."""
from typing import Any  # noqa pylint: disable=W0611
from typing import Callable  # noqa pylint: disable=W0611
from typing import Dict  # noqa pylint: disable=W0611
from typing import List  # noqa pylint: disable=W0611
from typing import Tuple  # noqa pylint: disable=W0611

from leabra7 import layer
from leabra7 import projn
from leabra7 import specs


class Net:
    def __init__(self) -> None:
        self.objs = {}  # type: Dict[str, Any]
        self.layers = []  # type: List[layer.Layer]
        self.projns = []  # type: List[projn.Projn]

    def _register(self, obj: Any) -> None:
        self.objs[obj.name] = obj

    def new_layer(self, name: str, size: int,
                  spec: specs.LayerSpec = None) -> None:
        lr = layer.Layer(name, size, spec)
        self.layers.append(lr)
        self._register(lr)

    def new_projn(self,
                  name: str,
                  pre: str,
                  post: str,
                  spec: specs.ProjnSpec = None) -> None:
        pre_lr = self.objs[pre]
        post_lr = self.objs[post]
        pr = projn.Projn(name, pre_lr, post_lr, spec)
        self.projns.append(pr)
        self._register(pr)

    def cycle(self) -> None:
        for lr in self.layers:
            lr.activation_cycle()

        for pr in self.projns:
            pr.flush()
