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
    """A leabra7 network. This is the main class."""

    def __init__(self) -> None:
        self.objs = {}  # type: Dict[str, Any]
        self.layers = []  # type: List[layer.Layer]
        self.projns = []  # type: List[projn.Projn]

        self.cycle_loggers = []  # type: List[log.Logger]

    def new_layer(self, name: str, size: int,
                  spec: specs.LayerSpec = None) -> None:
        """Adds a new layer to the network.

        Args:
            name: The name of the layer.
            size: How many units the layer should have.
            spec: The layer specification.

        Raises:
            spec.ValidationError: If the spec contains an invalid parameter
                value.

        """
        if spec is not None:
            spec.validate()
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
        """Adds a new projection to the network.

        Args:
            name: The name of the projection.
            pre: The name of the sending layer.
            post: The name of the receiving layer.
            spec: The projection specification.

        Raises:
            ValueError: If `pre` or `post` do not match any existing layer
                name.
            spec.ValidationError: If the spec contains an invalid parameter
                value.

        """
        if spec is not None:
            spec.validate()

        try:
            pre_lr = self.objs[pre]
        except KeyError:
            raise ValueError("No layer found with name {0}.".format(pre))

        try:
            post_lr = self.objs[post]
        except KeyError:
            raise ValueError("No layer found with name {0}.".format(post))

        pr = projn.Projn(name, pre_lr, post_lr, spec)
        self.projns.append(pr)
        self.objs[pr.name] = pr

    def cycle(self) -> None:
        """Cycles the network."""
        for lg in self.cycle_loggers:
            lg.record()

        for lr in self.layers:
            lr.activation_cycle()

        for pr in self.projns:
            pr.flush()

    def logs(self, freq: str, name: str) -> None:
        """Retrieves logs for an object in the network.

        Args:
            freq: The frequency at which the desired logs were recorded. One
                of `["cycle"]`.
            name: The name of the object for which the logs were recorded.

        Raises:
            ValueError: If the frequency name is invalid, or if no logs were
                recorded for the desired object.

        """
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
