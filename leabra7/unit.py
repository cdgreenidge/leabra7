"""A single computational unit, similar to a neuron.

For details on the algorithm, see O'Reilly, R. C., Munakata, Y., Frank, M. J.,
Hazy, T. E., and Contributors (2012). Computational Cognitive Neuroscience.
Wiki Book, 1st Edition. URL: http://ccnbook.colorado.edu

"""
from typing import Any

import numpy as np  # type: ignore
import scipy.interpolate  # type: ignore
import torch  # type: ignore

from leabra7 import log
from leabra7 import specs

# The next few functions deal with the noisy x/(x + 1) activation
# function. The actual unit class is farther down


def gaussian(res: float, std: float) -> Any:
    """Returns a Gaussian PDF in a Numpy array.

    It has mean 0. Some of the math here is funky but it exactly matches the
    Emergent implementation.

    Args:
        res: The resolution of the array.
        std: The standard deviation of the PDF.

    Returns:
        A Numpy array containing the Gaussian PDF.

    """
    if std < 1e-3:
        raise ValueError("Gaussian std cannot be less than 1e-3.")
    if res > 3.0 * std:
        raise ValueError("Gaussian res cannot be greater than 3.0 * std.")
    xlim = 3.0 * std
    xs = np.arange(-xlim, xlim + res, res)
    # Not sure why variance is half the normal size in Emergent
    gauss = np.exp(-(xs * xs) / (std * std))
    # We need to normalize with a sum because our tails do not go to infinity
    return gauss / sum(gauss)


def xx1(res: float, xmin: float, xmax: float) -> Any:
    """Evaluates the xx1 from xmin to xmax.

    Args:
        res: The resolution of the array.
        xmin: The lower bound.
        xmax: The upper bound.

    Returns:
        A Numpy array containing the xx1 function evaluated from xmin to xmax.

    """
    xs = np.arange(xmin, xmax, res)
    gain = 40
    u = gain * np.maximum(xs, 0.0)
    return u / (u + 1)


def nxx1_table() -> Any:
    """Returns a lookup table for the noisy XX1 function.

    The standard XX1 function is convolved with Gaussian noise.

    Returns:
        An (x, y) tuple where x and y are Numpy arrays holding the x and f(x)
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
    conv = conv[np.searchsorted(xs, xs_valid[0]):
                np.searchsorted(xs, xs_valid[-1]) + 1]
    return xs_valid, conv


class UnitGroup:
    """A group of computational units (aka neurons.)

    In this class, we use assertions instead of exceptions for error checking
    because they can be removed for speed by running with the -O flag.

    Args:
        size: The number of neurons in the group.
        spec: The specification for the unit.

    """
    nxx1_xs, nxx1_ys = nxx1_table()
    nxx1_numpy = scipy.interpolate.interp1d(
        nxx1_xs,
        nxx1_ys,
        copy=False,
        fill_value=(nxx1_ys[0], nxx1_ys[-1]),
        bounds_error=False)
    loggable_attrs = ("net_raw", "net", "gc_i", "act", "i_net", "i_net_r",
                      "v_m", "v_m_eq", "adapt", "spike")

    def __init__(self, size: int, spec: specs.UnitSpec = None) -> None:
        if size <= 0:
            raise ValueError("size must be greater than zero.")
        self.size = size

        if spec is None:
            self.spec = specs.UnitSpec()
        else:
            self.spec = spec

        # When adding any attribute to this class, update
        # layer.LayerSpec._valid_log_on_cycle

        # Net input (excitation) without time integration
        self.net_raw = torch.Tensor(self.size).zero_()
        # Net inpput (excitation) with time integration
        self.net = torch.Tensor(self.size).zero_()
        # Total (feedback + feedforward) inhibition
        self.gc_i = torch.Tensor(self.size).zero_()
        # Activation
        self.act = torch.Tensor(self.size).zero_()
        # Net current
        self.i_net = torch.Tensor(self.size).zero_()
        # Net current, rate-coded (driven by v_m_eq)
        self.i_net_r = torch.Tensor(self.size).zero_()
        # Membrane potential
        self.v_m = torch.Tensor(self.size).zero_()
        # Equilibrium membrane potential (does not reset on spike)
        self.v_m_eq = torch.Tensor(self.size).zero_()
        # Adaption current
        self.adapt = torch.Tensor(self.size).zero_()
        # Are we spiking? (0 or 1)
        # In the future, this could be a ByteTensor
        self.spike = torch.Tensor(self.size).zero_()

        # Supershort learning average
        self.avg_ss = torch.Tensor(self.size).zero_()
        # Short learning average
        self.avg_s = torch.Tensor(self.size).zero_()
        # Medium learning average
        self.avg_m = torch.Tensor(self.size).zero_()
        # Long learning average
        self.avg_l = torch.Tensor(self.size).zero_()

    def g_i_thr(self, unit_idx: int) -> float:
        """The inhibition that will place a unit at its spike threshold.

        Args:
            unit_idx: The index of the unit.

        Returns:
            The inhibition that will place the unit denoted by
            unit_idx at its spike threshold.

        """
        return (((self.spec.e_rev_e - self.spec.spk_thr) * self.net[unit_idx] +
                 (self.spec.e_rev_l - self.spec.spk_thr) * self.spec.gc_l) /
                (self.spec.spk_thr - self.spec.e_rev_i))

    def group_g_i_thr(self) -> torch.Tensor:
        """The inhibition that will place each unit at its spike threshold.

        Returns:
            The inhibition that will place each unit denoted by
            unit_idx at its spike threshold.

        """
        return (((self.spec.e_rev_e - self.spec.spk_thr) * self.net +
                 (self.spec.e_rev_l - self.spec.spk_thr) * self.spec.gc_l) /
                (self.spec.spk_thr - self.spec.e_rev_i))

    def add_input(self, inpt: torch.Tensor) -> None:
        """Adds excitatory inputs to each unit.

        Args:
          inpt: A 1D tensor with one value for each unit.

        """
        assert inpt.size()[0] == self.size
        self.net_raw += inpt

    def update_net(self) -> None:
        """Calculates the input for the next cycle by integrating over time."""
        self.net += self.spec.integ * self.spec.net_dt * (
            self.net_raw - self.net)
        self.net_raw.zero_()

    def update_inhibition(self, gc_i: torch.Tensor) -> None:
        """Sets the inhibition for each unit. gc_i is a 1D Tensor."""
        assert gc_i.size()[0] == self.size
        self.gc_i = gc_i

    def update_membrane_potential(self) -> None:
        """Updates the membrane potential.

        This assumes we already have updated the net input and unit
        inhibition.

        """
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

    def nxx1(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluates the noisy X/(X + 1) function.

        This is used to approximate the rate-coded unit response to a given
        input. This is a function, since interp1d is callable.

        Args:
            x: The value at which to evaluate the noisy X/(X + 1)
               function. Can be any array-like type.

        Returns:
            The value of the noisy X/(X + 1) function at `x`.

        """
        return torch.Tensor(self.nxx1_numpy(x))

    def update_activation(self) -> None:
        """Updates the unit activation.

        This assumes we have already updated the unit membrane potential.

        """
        # yapf: disable
        g_e_thr = (self.gc_i * (self.spec.e_rev_i - self.spec.spk_thr) *
                   self.spec.gc_l * (self.spec.e_rev_l - self.spec.spk_thr) -
                   self.adapt) / (self.spec.spk_thr - self.spec.e_rev_e)
        # yapf: enable

        is_spiking = self.v_m > self.spec.spk_thr
        self.v_m[is_spiking] = self.spec.v_m_r
        self.spike = is_spiking.float()

        pre_spike = (self.v_m_eq < self.spec.spk_thr).float()
        post_spike = 1 - pre_spike
        act_driver = pre_spike * (self.v_m_eq - self.spec.spk_thr
                                  ) + post_spike * (self.net - g_e_thr)
        self.act += (self.spec.integ * self.spec.vm_dt *
                     (self.nxx1(act_driver) - self.act))

        self.adapt += self.spec.integ * (
            self.spec.adapt_dt * (self.spec.vm_gain *
                                  (self.v_m - self.spec.e_rev_l) - self.adapt)
            + self.spike * self.spec.spike_gain)

    def update_cycle_learning_averages(self) -> None:
        """Updates the learning averages computed at the end of each cycle."""
        self.avg_ss += self.spec.integ * self.spec.ss_dt * (
            self.act - self.avg_ss)
        self.avg_s += self.spec.integ * self.spec.s_dt * (
            self.avg_ss - self.avg_s)
        self.avg_m += self.spec.integ * self.spec.m_dt * (
            self.avg_s - self.avg_m)

    def update_trial_learning_averages(self, acts_p_avg: float) -> None:
        """Updates the learning averages computed at the end of each trial.

        Args:
          acts_p_avg: The average layer activation at the end of the plus
            phase.

        """
        self.avg_l = self.spec.l_dn_dt * acts_p_avg * (self.avg_m - self.avg_l)
        mask = self.avg_m > 0.1
        self.avg_l[mask] = self.avg_m[mask] * self.spec.l_up_inc

    def top_k_net_indices(self, k: int) -> torch.Tensor:
        """Returns the indices of the top k units, sorted by net input.

        Args:
            k: The number of top units to return.

        Returns:
            A 1D torch tensor containing the indices of the units with the top
            k net input values, sorted descending by net input.

        """
        _, indices = torch.topk(self.net, k, largest=True, sorted=True)
        return indices

    def observe(self, attr: str) -> log.PartsObs:
        """Observes an attribute.

        This is not quite the same as log.ObservableMixin.observe_parts_attr(),
        since this is a helper method for Layer.observe_parts_attr(), meant to
        isolate Layer from changes in the internal storage of UnitGorup.

        Args:
          attr: The attr to observe. Can be any specified in
            Layer.parts_attributes.

        Returns:
          A PartsObs containing the attribute observation for the UnitGroup.

        Raises:
          ValueError: if the attribute is unobservable.

        """
        if attr not in self.loggable_attrs:
            raise ValueError(
                "{0} is not an observable attribute.".format(attr))
        return {
            "unit": list(range(self.size)),
            attr: getattr(self, attr).numpy().tolist()
        }
