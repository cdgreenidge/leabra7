"""A single computational unit, similar to a neuron."""
from leabra7 import specs


class Unit:
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
