"""Classes that bundle simulation parameters."""
from typing import Dict
from typing import Any


class Spec:
    def __init__(self, **kwargs: Dict[str, Any]) -> None:
        for name, value in kwargs.items():
            setattr(self, name, value)
