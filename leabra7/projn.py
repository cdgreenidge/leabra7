"""A connection between layers."""
import itertools
import math
from typing import TypeVar
from typing import Iterable
from typing import List

import torch  # type: ignore

from leabra7 import specs
from leabra7 import layer

T = TypeVar('T')


def tile(length: int, xs: Iterable[T]) -> List[T]:
    return list(itertools.islice(itertools.cycle(xs), length))


class Projn:
    """A projection links two layers. It is a bundle of connections.

    Args:
        name: The name of the projection.
        pre: The sending layer.
        post: The receiving layer.
        spec: The projection specification. If none is provided, the default
            spec will be used.

    """

    def __init__(self,
                 name: str,
                 pre: layer.Layer,
                 post: layer.Layer,
                 spec: specs.ProjnSpec = None) -> None:
        self.name = name
        self.pre = pre
        self.post = post

        if spec is None:
            self.spec = specs.ProjnSpec()
        else:
            self.spec = spec

        # Rows encode the postsynaptic units, and columns encode the
        # presynaptic units
        self.wts = torch.Tensor(self.post.size, self.pre.size).zero_()

        # Only create the projection between the units selected by the masks
        # Currently, only full connections are supported
        # TODO: Refactor mask expansion and creation into new methods + test
        expanded_pre_mask = tile(self.pre.size, self.spec.pre_mask)
        expanded_post_mask = tile(self.post.size, self.spec.post_mask)
        mask = torch.ger(
            torch.ByteTensor(expanded_post_mask),
            torch.ByteTensor(expanded_pre_mask))

        # Enforce sparsitya
        # TODO: Make this a separate method
        nonzero = mask.nonzero()
        num_nonzero = nonzero.shape[0]
        num_to_kill = math.floor((1 - self.spec.sparsity) * nonzero.shape[0])
        if num_to_kill > 0:
            to_kill = nonzero[torch.randperm(num_nonzero)[:num_to_kill], :]
            mask[to_kill[:, 0], to_kill[:, 1]] = 0
        num_nonzero = num_nonzero - num_to_kill

        # Fill the weight matrix with values
        rand_nums = torch.Tensor(num_nonzero)
        self.spec.dist.fill(rand_nums)
        self.wts[mask] = rand_nums

    def flush(self) -> None:
        """Propagates sending layer activation to the recieving layer.

        Separating this step from the activation and firing of the sending
        layer makes it easier to compute the net input scaling factor.

        """
        scale_eff = 1.0
        # TODO: stop violating law of Demeter here
        self.post.units.add_input(scale_eff * self.wts @ self.pre.units.act)
