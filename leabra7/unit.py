"""A single computational unit, similar to a neuron."""
from typing import Any
from typing import List
from typing import Tuple

import numpy as np  # type: ignore
import scipy.interpolate  # type: ignore

from leabra7 import specs

# The next few functions deal with the noisy x/(x + 1) activation
# function. The actual unit class is farther down


def gaussian(res: float, std: float) -> Any:
    """Returns a Gaussian PDF with mean 0 and variance std^2, in an array of
    resolution res.

    """
    if std < 1e-3:
        raise ValueError("Gaussian std cannot be less than 1e-3.")
    if res > 3.0 * std:
        raise ValueError("Gaussian res cannot be greater than 3.0 * std.")
    xlim = 3.0 * std
    xs = np.arange(-xlim, xlim + res, res)
    # Not sure why variance is half the normal size
    gauss = np.exp(-(xs * xs) / (std * std))
    # We need to normalize with a sum because our tails do not go to infinity
    return gauss / sum(gauss)


def xx1(res: float, xmin: float, xmax: float) -> Any:
    """Returns the xx1 function from xmin to xmax in an array with resolution
    res.

    """
    xs = np.arange(xmin, xmax, res)
    gain = 40
    u = gain * np.maximum(xs, 0.0)
    return u / (u + 1)


def nxx1_table() -> Any:
    """Returns an (x, y) tuple where x and y are arrays holding the x and f(x)
    values of the xx1 function convolved with Gaussian noise.

    """
    res = 0.001  # Resolution of x-axis
    std = 0.01  # Standard deviation of the Gaussian noise

    rng = 3 * std
    xmin = -2 * rng
    xmax = rng + 1.0 + res
    xs = np.arange(xmin, xmax, res)
    ys = xx1(res, xmin, xmax)
    conv = np.convolve(ys, gaussian(res, std), mode="same")

    xs_valid = np.arange(-rng, 1.0 + res, res)
    conv = conv[np.searchsorted(xs, xs_valid[0], side="left"):
                np.searchsorted(xs, xs_valid[-1], side="left") + 1]
    return (xs_valid, conv)


class Unit:
    nxx1_xs, nxx1_ys = nxx1_table()
    nxx1_interpolator = scipy.interpolate.interp1d(
        nxx1_xs, nxx1_ys, copy=False)

    def __init__(self, spec: specs.UnitSpec = None) -> None:
        if spec is None:
            self.spec = specs.UnitSpec()
        else:
            self.spec = spec

        # Net input (excitation) without time integration
        self.net_raw = 0.0
        # Net inpput (excitation) with time integration
        self.net = 0.0
        # Total (feedback + feedforward) inhibition
        self.gc_i = 0.0
        # Activation
        self.act = 0.0
        # Net current
        self.i_net = 0.0
        # Net current, rate-coded (driven by v_m_eq)
        self.i_net_r = 0.0
        # Membrane potential
        self.v_m = 0.0
        # Equilibrium membrane potential (does not reset on spike)
        self.v_m_eq = 0.0
        # Adaption current
        self.adapt = 0.0
        # Are we spiking? (0 or 1)
        self.spike = 0

    def add_input(self, inpt: float) -> None:
        self.net_raw += inpt

    def update_net(self) -> None:
        self.net += self.spec.integ * self.spec.net_dt * (
            self.net_raw - self.net)
        self.net_raw = 0.0

    def update_inhibition(self, gc_i: float) -> None:
        self.gc_i = gc_i

    def update_membrane_potential(self) -> None:
        # yapf: disable
        self.i_net = (self.net * (self.spec.e_rev_e - self.v_m) +
                      self.spec.gc_l * (self.spec.e_rev_l - self.v_m) +
                      self.gc_i * (self.spec.e_rev_i - self.v_m))
        self.v_m += self.spec.integ * self.spec.vm_dt * (
            self.i_net - self.adapt)

        self.i_net_r = (self.net * (self.spec.e_rev_e - self.v_m_eq) +
                        self.spec.gc_l * (self.spec.e_rev_l - self.v_m_eq) +
                        self.gc_i * (self.spec.e_rev_i - self.v_m_eq))
        self.v_m_eq += self.spec.integ * self.spec.vm_dt * (
            self.i_net_r - self.adapt)
        # yapf: enable

    def nxx1(self, x: float) -> float:
        if x < self.nxx1_xs[0]:
            return self.nxx1_ys[0]
        elif x > self.nxx1_xs[-1]:
            return self.nxx1_ys[-1]
        return self.nxx1_interpolator(x)

    def update_activation(self) -> None:
        # yapf: disable
        g_e_thr = (self.gc_i * (self.spec.e_rev_i - self.spec.spk_thr) *
                   self.spec.gc_l * (self.spec.e_rev_l - self.spec.spk_thr) -
                   self.adapt) / (self.spec.spk_thr - self.spec.e_rev_e)
        # yapf: enable

        if self.v_m > self.spec.spk_thr:
            self.spike = 1
            self.v_m = self.spec.v_m_r
        else:
            self.spike = 0

        if self.v_m_eq <= self.spec.spk_thr:
            new_act = self.nxx1(self.v_m_eq - self.spec.spk_thr)
        else:
            new_act = self.nxx1(self.net - g_e_thr)
        self.act += self.spec.integ * self.spec.vm_dt * (new_act - self.act)

        self.adapt += self.spec.integ * (
            self.spec.adapt_dt *
            (self.spec.vm_gain * (self.v_m - self.spec.e_rev_l) - self.adapt) +
            self.spike * self.spec.spike_gain)

    def observe(self, attr: str) -> List[Tuple[str, Any]]:
        simple_attrs = ("net_raw", "net", "gc_i", "act", "i_net", "i_net_r",
                        "v_m", "v_m_eq", "adapt", "spike")
        if attr in simple_attrs:
            return [("attr", getattr(self, attr))]
        else:
            raise ValueError("{0} is not a loggable attr.".format(attr))
