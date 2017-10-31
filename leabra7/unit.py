"""A single computational unit, similar to a neuron."""
from leabra7 import specs


class Unit:
    def __init__(self, spec: specs.UnitSpec = None) -> None:
        if spec is None:
            self.spec = specs.UnitSpec()
        else:
            self.spec = spec

        self.net_raw = 0.0

    def add_input(self, inpt: float) -> None:
        self.net_raw += inpt
