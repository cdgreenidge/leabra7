"""Classes that bundle simulation parameters."""
import abc
import math

from typing import Any
from typing import Iterable

from leabra7 import rand


class ValidationError(Exception):
    """Raised when an object fails validation checks."""
    pass


class Spec(metaclass=abc.ABCMeta):
    """Specs are record classes that hold parameter values.

    Every parameter in a spec has a default value, but you can override these
    default values using constructor keword arguments. For example,
    `UnitSpec()` will create a unit spec with all default values, but
    `UnitSpec(spk_thr=0.3)` will create a unit spec with all default values
    except for `spk_thr`, which is set to `0.3`.

    Specs can be modified at runtime, so before using a spec given
    to you by a user, always call the `validate()` method.

    Raises:
        ValueError: if you attempt to override a property that does not exist
            in this spec.

    """
    # Global integration time constant
    integ = 1.0

    def __init__(self, **kwargs: Any) -> None:
        for name, value in kwargs.items():
            if not hasattr(self, name):
                raise ValueError("{0} is not a valid parameter name for this "
                                 "spec.".format(name))
            setattr(self, name, value)

    def __eq__(self, other: object) -> bool:
        return self.__dict__ == other.__dict__

    # The following two assert methods could be pure functions, but this
    # way we have access to the attr name, which makes our error messages more
    # friendly

    def assert_in_range(self, attr: str, low: float, high: float) -> None:
        """Asserts that an attribute is in a closed interval.

        Args:
            attr: The attribute to check.
            low: The lower bound of the interval.
            high: The upper bound of the interval.

        Raises:
            ValidationError: If the attribute lies outside of [low, high].

        """
        if low > high:
            raise ValueError("low must be less than or equal to high.")
        if not low <= getattr(self, attr) <= high:
            msg = "{0} must be in the interval [{1}, {2}].".format(
                attr, low, high)
            raise ValidationError(msg)

    def assert_sane_float(self, attr: str) -> None:
        """Asserts that an attribute is not NaN, -Inf, or +Inf.

        Args:
            attr: The attribute to check.

        Raises:
            ValidationError: If the attribute is Nan.

        """
        value = getattr(self, attr)
        if math.isnan(value):
            raise ValidationError("Attribute {0} is NaN.".format(attr))
        elif value == float("-Inf"):
            raise ValidationError("Attribute {0} is -Inf.".format(attr))
        elif value == float("+Inf"):
            raise ValidationError("Attribute {0} is +Inf.".format(attr))

    @abc.abstractmethod
    def validate(self) -> None:
        """Checks if any parameter has an invalid value.

        Be sure to extend this method when subclassing.

        Raises:
            ValidationError: if any parameter has an invalid value.

        """
        self.assert_in_range("integ", 0, float("Inf"))


class UnitSpec(Spec):
    """Spec for unit objects."""
    # Excitation (net input) reversal potential
    e_rev_e = 1
    # Inhibitory reversal potential
    e_rev_i = 0.25
    # Leak reversal potential
    e_rev_l = 0.3
    # Leak current (this never updates, so it is a constant)
    gc_l = 0.1
    # Spiking threshold
    spk_thr = 0.5
    # Potential reset value after spike
    v_m_r = 0.3
    # Adaption current gain from potential
    vm_gain = 0.04
    # Adaption current gain from spiking
    spike_gain = 0.00805
    # Net input integration time constant
    net_dt = 1 / 1.4
    # Membrane potential integration time constant
    vm_dt = 1 / 3.3
    # Adaption current integration time constant
    adapt_dt = 1 / 144
    # Short term plasticity
    syn_tr = 1.0
    # Clamping activation limits
    act_min = 0.0
    act_max = 1.0
    # Clamping potential limits
    vm_min = 0.0
    vm_max = 2.0
    # Clamping gain
    act_gain = 100.0
    # Supershort learning average integration time constant
    ss_dt = 0.5
    # Short learning average integration time constant
    s_dt = 0.5
    # Medium learning average integration time constant
    m_dt = 0.1
    # Long learning average (decreasing) integration time constant
    l_dn_dt = 2.5
    # Long learning average (increasing) increment multiplier
    l_up_inc = 0.2

    def validate(self) -> None:
        """Extends `Spec.validate`."""
        super().validate()
        self.assert_sane_float("e_rev_e")
        self.assert_sane_float("e_rev_i")
        self.assert_sane_float("e_rev_l")
        self.assert_sane_float("gc_l")
        self.assert_sane_float("spk_thr")
        self.assert_sane_float("v_m_r")
        self.assert_sane_float("spike_gain")
        self.assert_in_range("net_dt", 0, float("Inf"))
        self.assert_in_range("vm_dt", 0, float("Inf"))
        self.assert_in_range("adapt_dt", 0, float("Inf"))
        self.assert_in_range("syn_tr", 0, 1.0)
        self.assert_in_range("act_min", 0, 1.0)
        self.assert_in_range("act_max", 0, 1.0)
        self.assert_sane_float("vm_min")
        self.assert_sane_float("vm_max")
        self.assert_in_range("act_gain", 0, float("Inf"))
        self.assert_in_range("ss_dt", 0, float("Inf"))
        self.assert_in_range("s_dt", 0, float("Inf"))
        self.assert_in_range("m_dt", 0, float("Inf"))
        self.assert_in_range("l_dn_dt", 0, float("Inf"))
        self.assert_in_range("l_up_inc", 0, float("Inf"))

        if self.v_m_r >= self.spk_thr:
            raise ValidationError(
                "v_m_r ({0}) cannot be >= spk_thr ({1}).".format(
                    self.v_m_r, self.spk_thr))

        if self.act_min >= self.act_max:
            raise ValidationError(
                "act_min ({0}) cannot be >= act_max ({1}).".format(
                    self.act_min, self.act_max))

        if self.vm_min >= self.vm_max:
            raise ValidationError(
                "vm_min ({0}) cannot be >= vm_max ({1}).".format(
                    self.vm_min, self.vm_max))

        if self.vm_min > self.v_m_r or self.vm_max < self.v_m_r:
            raise ValidationError(
                """v_m_r ({0}) must be in range from vm_min ({1})
                to vm_max ({2}).""".format(self.v_m_r, self.vm_min,
                                           self.vm_max))


class LayerSpec(Spec):
    """Spec for Layer objects."""
    # Can be either "fffb" for feedforward-feedback inhibition, or
    # "kwta" for k-winner-take-all inhibition
    inhibition_type = "fffb"
    # Proportion of winners for k-winner-take-all inhibition
    kwta_pct = 0.1
    # Proportion of kWTA step (scalar for update step size)
    kwta_pt = 0.5
    # Feedforward inhibition multiplier
    ff = 1.0
    # Feedforward inhibition offset
    ff0 = 0.1
    # Feedback inhibition multiplier
    fb = 1.0
    # Feedback inhibition integration time constant
    fb_dt = 1 / 1.4
    # Global (feedforward + feedback) inhibition multiplier
    gi = 1.8
    # cos_diff_avg integration time constant
    avg_dt = 0.01

    # Attrs to log every cycle
    log_on_cycle: Iterable[str] = ()
    # Attrs to log every trial
    log_on_trial: Iterable[str] = ()
    # Attrs to log every epoch
    log_on_epoch: Iterable[str] = ()
    # Attrs to log every batch
    log_on_batch: Iterable[str] = ()

    # Valid attributes to log on every cycle
    # When adding any loggable attribute or property to this list,
    # update layer._whole_attrs or layer._parts_attrs as appropriate
    # (we represent in two places to avoid a circular dependency)
    _valid_attrs_to_log = ("avg_act", "avg_net", "fbi", "unit_net_raw",
                           "unit_net", "unit_gc_i", "unit_act", "unit_i_net",
                           "unit_i_net_r", "unit_v_m", "unit_v_m_eq",
                           "unit_adapt", "unit_spike")

    # Layers need to know how to construct their units
    unit_spec = UnitSpec()

    def validate(self) -> None:
        """Extends `Spec.validate`."""
        super().validate()

        valid_inhibition_types = ["fffb", "kwta", "kwta_avg", "none"]
        if self.inhibition_type not in valid_inhibition_types:
            raise ValidationError(
                "Inhibition type {0} not one of [\"fffb\", \"kwta\", \
                \"kwta_avg\", \"none\"]".format(self.inhibition_type))

        self.assert_in_range("kwta_pct", 0.0, 1.0)
        self.assert_in_range("kwta_pt", 0.0, 1.0)
        self.assert_sane_float("ff")
        self.assert_sane_float("fb")
        self.assert_in_range("fb_dt", 0, float("Inf"))
        self.assert_sane_float("gi")
        self.unit_spec.validate()

        for attr in self.log_on_cycle:
            if attr not in self._valid_attrs_to_log:
                raise ValidationError("{0} is not a valid member of "
                                      "log_on_cycle.".format(attr))


class ProjnSpec(Spec):
    """Spec for `Projn` objects."""
    # The probability distribution from which the connection weights will be
    # drawn
    dist: rand.Distribution = rand.Scalar(0.5)
    # Selects which pre layer units will be included in the projection
    # If the length is less than the number of units in the pre_layer, it will
    # be tiled. If the length is more, it will be truncated.
    pre_mask: Iterable[bool] = (True, )
    # Selects which post layer units will be included in the projection
    # If the length is less than the number of units in the pre_layer, it will
    # be tiled. If the length is more, it will be truncated.
    post_mask: Iterable[bool] = (True, )
    # Sparsity of the connection (i.e. the percentage of active connections.)
    sparsity: float = 1.0
    # Set special type of projection
    projn_type = "full"
    # Absolute net input scaling weight
    wt_scale_abs: float = 1.0
    # Relative net input scaling weight (relative to other projections
    # terminating in the same layer)
    wt_scale_rel: float = 1.0
    # Learning rate
    lrate = 0.02
    # Gain for sigmoidal weight contrast enhancement
    sig_gain = 6
    # Offset for sigmoidal weight contrast enhancement
    sig_offset = 1

    def validate(self) -> None:  # pylint: disable=W0235
        """Extends `Spec.validate`."""
        if not isinstance(self.dist, rand.Distribution):
            raise ValidationError("{0} is not a valid "
                                  "distribution.".format(self.dist))
        self.assert_in_range("sparsity", low=0.0, high=1.0)

        valid_projn_types = ["one_to_one", "full"]
        if self.projn_type not in valid_projn_types:
            raise ValidationError(
                "Projn type {0} not one of [\"one_to_one\", \"full\"]".format(
                    self.projn_type))

        self.assert_in_range("wt_scale_abs", 0, float("Inf"))
        self.assert_in_range("wt_scale_rel", 0, float("Inf"))
        self.assert_in_range("lrate", 0, float("Inf"))
        self.assert_in_range("sig_gain", 0, float("Inf"))
        self.assert_sane_float("sig_offset")

        super().validate()
