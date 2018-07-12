"""A connection between layers."""
import itertools
import math
from typing import TypeVar
from typing import Iterable
from typing import List
from typing import Tuple

import torch  # type: ignore

from leabra7 import specs
from leabra7 import layer

T = TypeVar('T')


def tile(length: int, xs: Iterable[T]) -> List[T]:
    """Tiles an iterable.

    Args:
        length: The length to tile the iterable to.
        xs: The iterable to tile.

    Returns:
        An iterable of size `length`, containing elements from xs, tiled.

    """
    assert length > 0
    return list(itertools.islice(itertools.cycle(xs), length))


def expand_layer_mask_full(pre_mask: List[bool],
                           post_mask: List[bool]) -> torch.ByteTensor:
    """Expands layer masks into a weight matrix mask with full connectivity.

    Args:
        pre_mask: The mask for the pre layer specifying which pre layer
            units are included in the projection. Note that this mask will not
            be tiled, so it has as many elements as pre layer units.

        post_mask: The mask for the post layer specifying which post layer
            units are included in the projection. Has as many elements as post
            layer units.

    Returns:
        A mask for the full projection weight matrix indicating
        which elements of the matrix correspond to active connections
        in the full connectivity pattern.

    """
    # In the full connectivity case, it can be concisely calculated with an
    # outer product
    return torch.ger(torch.ByteTensor(post_mask), torch.ByteTensor(pre_mask))


def expand_layer_mask_one_to_one(pre_mask: List[bool],
                                 post_mask: List[bool]) -> torch.ByteTensor:
    """
    Expands layer masks into a weight matrix mask
    with one-to-one connectivity.

    Args:
        pre_mask: The mask for the pre layer specifying which pre layer
            units are included in the projection. Note that this mask will not
            be tiled, so it has as many elements as pre layer units.

        post_mask: The mask for the post layer specifying which post layer
            units are included in the projection. Has as many elements as post
            layer units.

    Returns:
        A mask for the one-to-one projection weight matrix indicating
        which elements of the matrix correspond to active connections
        in the full connectivity pattern.

    """
    if sum(pre_mask) != sum(post_mask):
        raise ValueError(
            """Mismatched one-to-one projection. Pre_mask units: {0}.
            Post_mask units: {1}.""".format(sum(pre_mask), sum(post_mask)))

    mask = torch.zeros(len(pre_mask), len(post_mask)).byte()
    i = j = 0
    while i < len(pre_mask):
        if pre_mask[i]:
            if post_mask[j]:
                mask[i, j] = 1
                i += 1
                j += 1
            else:
                j += 1
        else:
            i += 1
    return mask


def sparsify(sparsity: float,
             tensor: torch.ByteTensor) -> Tuple[torch.ByteTensor, int]:
    """
    Makes a boolean tensor sparse, by randomly setting True values to False.

    Args:
        sparsity: The percentage of `True` values from the original
            matrix to keep.
        tensor: The ByteTensor to sparsify.

    Returns:
        A tuple. The first element is a ByteTensor of the same shape
        as the original tensor, but with floor(1 - sparsity)% of its
        true values set to `False`. The second element is the number of True
        values in the sparsified Tensor.
    """
    assert 0 <= sparsity <= 1
    nonzero = tensor.nonzero()
    num_nonzero = nonzero.shape[0]
    num_to_keep = math.floor(sparsity * num_nonzero)
    to_keep = nonzero[torch.randperm(num_nonzero)[:num_to_keep]]
    sparse = torch.zeros_like(tensor)
    # All we want to do is set the elements of sparse given by the
    # indices in to_keep to True. The list comprehension below is pretty hacky,
    # but I can't find a cleaner way to do it in torch 0.3
    sparse[[to_keep[:, i] for i in range(to_keep.shape[1])]] = 1
    return (sparse, num_to_keep)


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

        # A matrix where each element is the weight of a connection.
        # Rows encode the postsynaptic units, and columns encode the
        # presynaptic units
        self.wts = torch.Tensor(self.post.size, self.pre.size).zero_()

        tiled_pre_mask = tile(self.pre.size, self.spec.pre_mask)
        tiled_post_mask = tile(self.post.size, self.spec.post_mask)

        if self.spec.projn_type == "one_to_one":
            mask = expand_layer_mask_one_to_one(tiled_pre_mask,
                                                tiled_post_mask)

        elif self.spec.projn_type == "none":
            # Only create the projection between the units
            # selected by the masks
            # Currently, only full connections are supported
            # TODO: Refactor mask expansion and creation into
            # new methods + test
            mask = expand_layer_mask_full(tiled_pre_mask, tiled_post_mask)

        # Enforce sparsity
        # TODO: Make this a separate method
        mask, num_nonzero = sparsify(self.spec.sparsity, mask)

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
