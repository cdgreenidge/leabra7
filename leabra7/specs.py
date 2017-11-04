"""Classes that bundle simulation parameters."""
from typing import Any
from typing import Dict


class Spec:
    # Global integration time constant
    integ = 1.0

    def __init__(self, **kwargs: Dict[str, Any]) -> None:
        for name, value in kwargs.items():
            getattr(self, name)  # Throw exception if it doesn't exist
            setattr(self, name, value)

    def __eq__(self, other: object) -> bool:
        return self.__dict__ == other.__dict__


class UnitSpec(Spec):
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

    # Layers need to know how to construct their units
    unit_spec = UnitSpec()
