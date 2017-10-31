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

    def add_input(self, inpt: float) -> None:
        self.net_raw += inpt

    def update_net_input(self) -> None:
        self.net += self.spec.integ * self.spec.net_dt * (
            self.net_raw - self.net)

    def update_inhibition(self, gc_i: float) -> None:
        self.gc_i = gc_i
