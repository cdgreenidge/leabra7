"""A layer, or group, of units."""
import statistics

from leabra7 import specs
from leabra7 import unit


class Layer:
    def __init__(self, name: str, size: int,
                 spec: specs.LayerSpec = None) -> None:
        self.name = name
        self.size = size

        if spec is None:
            self.spec = specs.LayerSpec()
        else:
            self.spec = spec

        self.units = [unit.Unit(self.spec.unit_spec) for _ in range(size)]

        self.fbi = 0.0

    def avg_act(self) -> float:
        return statistics.mean(unit.act for unit in self.units)

    def avg_net(self) -> float:
        return statistics.mean(unit.net for unit in self.units)

    def update_net(self) -> None:
        for i in self.units:
            i.update_net()

    def update_inhibition(self) -> None:
        # Feedforward inhibition
        ffi = self.spec.ff * max(self.avg_net() - self.spec.ff0, 0)
        # Feedback inhibition
        self.fbi = self.spec.fb_dt * (self.spec.fb * self.avg_act() - self.fbi)
        gc_i = self.spec.gi * (ffi * self.fbi)

        for i in self.units:
            i.update_inhibition(gc_i)

    def update_membrane_potential(self) -> None:
        for i in self.units:
            i.update_membrane_potential()

    def update_activation(self) -> None:
        for i in self.units:
            i.update_activation()

    def activation_cycle(self) -> None:
        self.update_net()
        self.update_inhibition()
        self.update_membrane_potential()
        self.update_activation()
