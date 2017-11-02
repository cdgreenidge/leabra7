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

    def avg_act(self) -> float:
        return statistics.mean(unit.act for unit in self.units)

    def update_net_input(self) -> None:
        for u in self.units:
            u.update_net_input()
