"""A layer, or group, of units."""
import itertools
from typing import List
from typing import Iterable

import torch  # type: ignore

from leabra7 import log
from leabra7 import specs
from leabra7 import events
from leabra7 import unit
from leabra7 import utils


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


class Layer(log.ObservableMixin, events.EventListenerMixin):
    """A layer of units (neurons).

    Args:
        name: The name of the layer.
        size: The number of units in the layer.
        spec: The layer specification. If it is `None`, the default spec will
            be used.

    """

    def __init__(self, name: str, size: int,
                 spec: specs.LayerSpec = None) -> None:
        self._name = name
        self.size = size

        if spec is None:
            self._spec = specs.LayerSpec()
        else:
            self._spec = spec

        self.units = unit.UnitGroup(size=size, spec=self.spec.unit_spec)

        # Feedback inhibition
        self.fbi = 0.0
        # Global inhibition
        self.gc_i = 0.0
        # Is the layer activation clamped?
        self.clamped = False
        # Set k units for inhibition
        self.k = max(1, int(round(self.size * self.spec.kwta_pct)))

        # Desired clamping values
        self.act_ext = torch.Tensor(self.size).zero_()
        # Last plus phase activation
        self.acts_p = torch.Tensor(self.size).zero_()
        # Last minus phase activation
        self.acts_m = torch.Tensor(self.size).zero_()
        # Cosine similiarity between acts_p and acts_m, integrated over trials
        self.cos_diff_avg = 0.0

        # The following two buffers are filled every time self.add_input() is
        # called, and reset at the end of self.activation_cycle()

        # Net input (excitation) input buffer. For every cycle, we
        # store the layer inputs here. Once we have all the inputs, we
        # normalize by wt_scale_rel_sum and send to the unit group.
        self.input_buffer = torch.Tensor(self.size).zero_()

        # Sum of the wt_scale_rel parameters for each projection terminating in
        # this layer. We use this to normalize the inputs before propagating to
        # unit group
        self.wt_scale_rel_sum = 0.0

        # When adding any loggable attribute or property to these lists, update
        # specs.LayerSpec._valid_log_on_cycle (we represent in two places to
        # avoid a circular dependency)
        whole_attrs: List[str] = ["avg_act", "avg_net", "cos_diff_avg", "fbi"]
        parts_attrs: List[str] = [
            "unit_net", "unit_net_raw", "unit_gc_i", "unit_act", "unit_i_net",
            "unit_i_net_r", "unit_v_m", "unit_v_m_eq", "unit_adapt",
            "unit_spike"
        ]

        super().__init__(whole_attrs=whole_attrs, parts_attrs=parts_attrs)

    @property
    def avg_act(self) -> float:
        """Returns the average activation of the layer's units."""
        return float(torch.mean(self.units.act))

    @property
    def avg_net(self) -> float:
        """Returns the average net input of the layer's units."""
        return torch.mean(self.units.net)

    @property
    def name(self) -> str:
        """Overrides `ObservableMixin.name`."""
        return self._name

    @property
    def spec(self) -> specs.LayerSpec:
        """Overrides `ObservableMixin.spec`."""
        return self._spec

    def add_input(self, inpt: torch.Tensor, wt_scale_rel: float = 1.0) -> None:
        """Adds an input to the layer.

        Args:
          inpt: The vector of inputs to add to each unit in the layer. These
            should NOT be scaled by wt_scale_rel.
          wt_scale_rel: The wt_scale_rel parameter of the projection sending
            the inputs.

        """
        self.input_buffer += inpt * wt_scale_rel
        self.wt_scale_rel_sum += wt_scale_rel

    def update_net(self) -> None:
        """Updates the net input of the layer's units."""
        # self.wt_scale_rel_sum could be zero if there are no inbound
        # projections, or if the projections have not been flushed yet
        if self.wt_scale_rel_sum == 0:
            assert (self.input_buffer == 0).all()
        else:
            self.input_buffer /= self.wt_scale_rel_sum

        self.units.add_input(self.input_buffer)
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
        if self.k == self.size:
            self.gc_i = 0
            return
        top_m_units = self.units.top_k_net_indices(self.k + 1)
        g_i_thr_m = self.units.g_i_thr(top_m_units[-1])
        g_i_thr_k = self.units.g_i_thr(top_m_units[-2])
        self.gc_i = g_i_thr_m + self.spec.kwta_pt * (g_i_thr_k - g_i_thr_m)

    def calc_kwta_avg_inhibition(self) -> None:
        """Calculates k-winner-take-all average inhibition for the layer."""
        if self.k == self.size:
            self.gc_i = 0
            return
        g_i_thr, _ = torch.sort(self.units.group_g_i_thr(), descending=True)
        g_i_thr_k = torch.mean(g_i_thr[0:self.k])
        g_i_thr_n_k = torch.mean(g_i_thr[self.k:])
        self.gc_i = g_i_thr_n_k + self.spec.kwta_pt * (g_i_thr_k - g_i_thr_n_k)

    def update_inhibition(self) -> None:
        """Updates the inhibition for the layer's units."""
        if self.spec.inhibition_type == "fffb":
            self.calc_fffb_inhibition()
        elif self.spec.inhibition_type == "kwta":
            self.calc_kwta_inhibition()
        elif self.spec.inhibition_type == "kwta_avg":
            self.calc_kwta_avg_inhibition()

        self.units.update_inhibition(torch.Tensor(self.size).fill_(self.gc_i))

    def activation_cycle(self) -> None:
        """Runs one complete activation cycle of the layer."""
        if not self.clamped:
            self.update_net()
            self.update_inhibition()
            self.units.update_membrane_potential()
            self.units.update_activation()

        self.units.update_cycle_learning_averages()
        self.input_buffer.zero_()
        self.wt_scale_rel_sum = 0

    def update_trial_learning_averages(self) -> None:
        """Updates the learning averages computed at the end of each trial."""
        cos_diff = torch.nn.functional.cosine_similarity(
            self.acts_p, self.acts_m, dim=0)
        cos_diff = utils.clip_float(low=0.01, high=0.99, x=cos_diff)
        self.cos_diff_avg = self.spec.avg_dt * (cos_diff - self.cos_diff_avg)

        acts_p_avg_eff = self.acts_p.mean()
        self.units.update_trial_learning_averages(acts_p_avg_eff)

    @property
    def avg_s(self) -> torch.Tensor:
        """The short learning average for each unit."""
        return self.units.avg_s

    @property
    def avg_m(self) -> torch.Tensor:
        """The medium learning average for each unit."""
        return self.units.avg_m

    @property
    def avg_l(self) -> torch.Tensor:
        """The long learning average for each unit."""
        return self.units.avg_l

    def hard_clamp(self, act_ext: Iterable[float]) -> None:
        """Forces the layer's activations.

        After forcing, the layer's activations will be set to the values
        contained in `acts` and will not change from cycle to cycle.

        Args:
            act_ext: An iterable containing the activations that the layer's
                units will be clamped to. If its length is less than the number
                of units in the layer, it will be tiled. If its length is
                greater, the extra values will be ignored.

        """
        self.clamped = True
        trimmed = utils.clip_iterable(0.0, self.spec.clamp_max, act_ext)
        self.act_ext = torch.Tensor(
            list(itertools.islice(itertools.cycle(trimmed), self.size)))
        self.units.hard_clamp(self.act_ext)

    def unclamp(self) -> None:
        """Unclamps the layer."""
        self.clamped = False

    def observe_parts_attr(self, attr: str) -> log.PartsObs:
        if attr not in self.parts_attrs:
            raise ValueError("{0} is not a valid parts attr.".format(attr))
        parsed = _parse_unit_attr(attr)
        return self.units.observe(parsed)

    def handle(self, event: events.Event) -> None:
        if isinstance(event, events.HardClamp):
            if event.layer_name == self.name:
                self.hard_clamp(event.acts)
        elif isinstance(event, events.EndPlusPhase):
            self.acts_p.copy_(self.units.act)
            self.update_trial_learning_averages()
        elif isinstance(event, events.EndMinusPhase):
            self.acts_m.copy_(self.units.act)
        elif isinstance(event, events.Unclamp):
            if event.layer_name == self.name:
                self.unclamp()
