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

        self.cycle_loggers = []  # type: List[log.Logger]

    def new_layer(self, name: str, size: int,
                  spec: specs.LayerSpec = None) -> None:
        lr = layer.Layer(name, size, spec)
        self.layers.append(lr)
        self.objs[lr.name] = lr

        if lr.spec.log_on_cycle != ():
            self.cycle_loggers.append(log.Logger(lr, lr.spec.log_on_cycle))

    def new_projn(self,
                  name: str,
                  pre: str,
                  post: str,
                  spec: specs.ProjnSpec = None) -> None:
        pre_lr = self.objs[pre]
        post_lr = self.objs[post]
        pr = projn.Projn(name, pre_lr, post_lr, spec)
        self.projns.append(pr)
        self.objs[pr.name] = pr

    def cycle(self) -> None:
        for lg in self.cycle_loggers:
            lg.record()

        for lr in self.layers:
            lr.activation_cycle()

        for pr in self.projns:
            pr.flush()

    def logs(self, freq: str, name: str) -> None:
        freq_names = {"cycle": self.cycle_loggers}
        try:
            freq_loggers = freq_names[freq]
        except KeyError:
            raise ValueError("{0} must be one of {1}.".format(
                freq, freq_names.keys()))

        try:
            logger = next(i for i in freq_loggers if i.target_name == name)
        except StopIteration:
            raise ValueError(
                "No logs recorded for object {0}, frequency {1}.".format(
                    name, freq))

        return logger.to_df()
