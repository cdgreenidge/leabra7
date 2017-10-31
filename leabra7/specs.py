"""Classes that bundle simulation parameters."""
from typing import Dict
from typing import Any


class Spec:
    def __init__(self, **kwargs: Dict[str, Any]) -> None:
        for name, value in kwargs.items():
            getattr(self, name)  # Throw exception if it doesn't exist
            setattr(self, name, value)

    def __eq__(self, other: object) -> bool:
        return self.__dict__ == other.__dict__


class UnitSpec(Spec):
    pass
