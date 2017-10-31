"""Classes that bundle simulation parameters."""
from typing import Dict
from typing import Any


class Spec:
    # Global integration time constant
    integ = 1.0

    def __init__(self, **kwargs: Dict[str, Any]) -> None:
        for name, value in kwargs.items():
            getattr(self, name)  # Throw exception if it doesn't exist
            setattr(self, name, value)

    def __eq__(self, other: object) -> bool:
        return self.__dict__ == other.__dict__


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


class UnitSpec(Spec):
    # Net input integration time constant
    net_dt = 1 / 1.4
