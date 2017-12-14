"""A network."""
from typing import Any  # noqa pylint: disable=W0611
from typing import Callable  # noqa pylint: disable=W0611
from typing import Dict  # noqa pylint: disable=W0611
from typing import List  # noqa pylint: disable=W0611
from typing import Tuple  # noqa pylint: disable=W0611

from leabra7 import layer
from leabra7 import log
from leabra7 import projn
from leabra7 import specs


class Net:
    def __init__(self) -> None:
        self.objs = {}  # type: Dict[str, Any]
        self.layers = []  # type: List[layer.Layer]
        self.projns = []  # type: List[projn.Projn]
        self.cycle_hooks = [
        ]  # type: List[Callable[[], List[Tuple[str, Any]]]]
        self.buf = log.DataFrameBuffer()

    def new_layer(self, name: str, size: int,
                  spec: specs.LayerSpec = None) -> None:
        lr = layer.Layer(name, size, spec)
        self.layers.append(lr)
        self.objs[name] = lr

        logger = log.layer_reader(lr.spec.log_on_cycle, size)

        def hook():
            self.buf.append(logger(lr))

        self.cycle_hooks.append(hook)

    def new_projn(self,
                  name: str,
                  pre: str,
                  post: str,
                  spec: specs.ProjnSpec = None) -> None:
        pre_lr = self.objs[pre]
        post_lr = self.objs[post]
        pr = projn.Projn(name, pre_lr, post_lr, spec)
        self.projns.append(pr)
        self.objs[name] = pr

    def cycle(self) -> None:
        for lr in self.layers:
            lr.activation_cycle()

        for pr in self.projns:
            pr.flush()

        for f in self.cycle_hooks:
            f()

    def data(self) -> Any:
        return self.buf.to_df()
