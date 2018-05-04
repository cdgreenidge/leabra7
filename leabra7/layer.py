"""A layer, or group, of units."""
import itertools
import statistics
from typing import List
from typing import Iterable

import torch  # type: ignore

from leabra7 import log
from leabra7 import specs
from leabra7 import unit


def _parse_unit_attr(attr: str) -> str:
    """Removes the unit_ prefix from a unit attribute.

    For example, `_parse_unit_attr("unit_act")` will return `"act"`.

    Args:
        attr: The attribute to parse.

    Returns:
        `attr`, but with the `"unit_"` prefix removed.

    Raises:
        ValueError: If `attr` cannot be parsed.

    """
    parts = attr.split('_', maxsplit=1)
    valid_attr = len(parts) == 2 and parts[0] == "unit"
    if not valid_attr:
        raise ValueError("{0} is not a valid unit attribute.".format(attr))
    return parts[1]


class Layer(log.ObservableMixin):
    """A layer of units (neurons).

    Args:
        name: The name of the layer.
        size: The number of units in the layer.
        spec: The layer specification. If it is `None`, the default spec will
            be used.

    """

    def __init__(self, name: str, size: int,
                 spec: specs.LayerSpec = None) -> None:
        self.size = size

        if spec is None:
            self.spec = specs.LayerSpec()
        else:
            self.spec = spec

        self.units = unit.UnitGroup(size=size, spec=self.spec.unit_spec)

        # When adding any attribute or property to this class, update
        # layer.LayerSpec._valid_log_on_cycle

        # Feedback inhibition
        self.fbi = 0.0
        # Global inhibition
        self.gc_i = 0.0
        # Is the layer activation forced?
        self.forced = False

        super().__init__(name)

    @property
    def avg_act(self) -> float:
        """Returns the average activation of the layer's units."""
        return statistics.mean(self.units.act)

    @property
    def avg_net(self) -> float:
        """Returns the average net input of the layer's units."""
        return statistics.mean(self.units.net)

    def update_net(self) -> None:
        """Updates the net input of the layer's units."""
        self.units.update_net()

    def calc_fffb_inhibition(self) -> None:
        """Calculates feedforward-feedback inhibition for the layer."""
        # Feedforward inhibition
        ffi = self.spec.ff * max(self.avg_net - self.spec.ff0, 0)
        # Feedback inhibition
        self.fbi = self.spec.fb_dt * (self.spec.fb * self.avg_act - self.fbi)
        # Global inhibition
        self.gc_i = self.spec.gi * (ffi * self.fbi)

    def calc_kwta_inhibition(self) -> None:
        """Calculates k-winner-take-all inhibition for the layer."""
        top_m_units = self.units.top_k_net_indices(self.spec.k + 1)
        g_i_thr_m = self.units.g_i_thr(top_m_units[-1])
        g_i_thr_k = self.units.g_i_thr(top_m_units[-2])
        self.gc_i = g_i_thr_m + 0.5 * (g_i_thr_k - g_i_thr_m)

    def update_inhibition(self) -> None:
        """Updates the inhibition for the layer's units."""
        if self.spec.inhibition_type == "fffb":
            self.calc_fffb_inhibition()
        else:
            self.calc_kwta_inhibition()

        self.units.update_inhibition(torch.Tensor(self.size).fill_(self.gc_i))

    def update_membrane_potential(self) -> None:
        """Updates the membrane potential of the layer's units."""
        self.units.update_membrane_potential()

    def update_activation(self) -> None:
        """Updates the activation of the layer's units."""
        if self.forced:
            return
        self.units.update_activation()

    def activation_cycle(self) -> None:
        """Runs one complete activation cycle of the layer."""
        self.update_net()
        self.update_inhibition()
        self.update_membrane_potential()
        self.update_activation()

    def force(self, acts: Iterable[float]) -> None:
        """Forces the layer's activations.

        After forcing, the layer's activations will be set to the values
        contained in `acts` and will not change from cycle to cycle.

        Args:
            acts: An iterable containing the activations that the layer's
                units will be forced to. If its length is less than the number
                of units in the layer, it will be tiled. If its length is
                greater, the extra values will be ignored.

        """
        self.forced = True
        for i, act in zip(range(self.size), itertools.cycle(acts)):
            self.units.act[i] = act

    def observe(self, attr: str) -> List[log.Obs]:
        """Overrides `log.ObservableMixin.observe`."""
        # TODO: fix the logging system, which is kinda broken
        try:
            parsed = _parse_unit_attr(attr)
            return self.units.observe(parsed)
        except ValueError:
            pass

        valid = ("avg_act", "avg_net", "fbi")
        if attr not in valid:
            raise ValueError(
                "{0} is not a valid layer attribute.".format(attr))

        return [{attr: getattr(self, attr)}]
