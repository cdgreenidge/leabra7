"""Classes that bundle simulation parameters."""
import abc
import math
from typing import Any
from typing import Dict
from typing import Tuple  # noqa pylint: disable=W0611


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

    def __init__(self, **kwargs: Dict[str, Any]) -> None:
        for name, value in kwargs.items():
            if not hasattr(self, name):
                raise ValueError("{0} is not a valid parameter name for this "
                                 "spec.".format(name))
            setattr(self, name, value)

    def __eq__(self, other: object) -> bool:
        return self.__dict__ == other.__dict__

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

    def assert_not_nan(self, attr: str) -> None:
        """Asserts that an attribute is not NaN.

        Args:
            attr: The attribute to check.

        Raises:
            ValidationError: If the attribute is Nan.

        """
        if math.isnan(getattr(self, attr)):
            raise ValidationError("Attribute {0} is NaN.".format(attr))

    @abc.abstractmethod
    def validate(self) -> None:
        """Raises ValidationError if any parameter is out of range."""


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

    def validate(self) -> None:
        """Overrides `Spec.validate`."""
        raise NotImplementedError


class LayerSpec(Spec):
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

    # Attrs to log every cycle. Possible values are ("avg_act", "avg_net")
    log_on_cycle = ()  # type: Tuple[str, ...]

    # Layers need to know how to construct their units
    unit_spec = UnitSpec()

    def validate(self) -> None:
        """Overrides `Spec.validate`."""
        raise NotImplementedError


class ConnSpec(Spec):
    def validate(self) -> None:
        """Overrides `Spec.validate`."""
        raise NotImplementedError


class ProjnSpec(Spec):
    def validate(self) -> None:
        """Overrides `Spec.validate`."""
        raise NotImplementedError
