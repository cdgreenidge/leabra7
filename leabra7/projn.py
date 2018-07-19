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
from leabra7 import events

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
    i = 0
    j = 0
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


def xcal(x: torch.Tensor, thr: torch.Tensor) -> torch.Tensor:
    """Computes the XCAL learning function on a tensor (vectorized)

    See the Emergent docs for more info.

    Args:
      x: A 1D tensor of inputs to the function.
      thr: A 1D tensor of threshold values.

    Returns:
      A tensor with XCAL computed for each value.

    """
    d_thr = 0.0001
    d_rev = 0.1
    result = torch.zeros_like(x)

    mask = x > (thr * d_rev)
    result[mask] = x[mask] - thr[mask]
    result[~mask] = -x[~mask] * ((1 - d_rev) / d_rev)

    result[x < d_thr] = 0
    return result


def sig(gain: float, offset: float, x: torch.Tensor) -> torch.Tensor:
    """Computes element-wise sigmoid function.

    Args:
      gain: The sigmoid function gain.
      offset: The sigmoid function offset.

    Returns:
      The sigmoid function evaluated for each element in the tensor.

    """
    return 1 / (1 + (offset * (1 - x) / x)**gain)


class Projn(events.EventListenerMixin):
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
        # presynaptic units. These weights are sigmoidally contrast-enchanced,
        # and are used to send net input to other neurons.
        self.wts = torch.Tensor(self.post.size, self.pre.size).zero_()
        # These weights ("fast weights") are linear and not contrast enhanced
        self.fwts = torch.Tensor(self.post.size, self.pre.size).zero_()

        # Only create the projection between the units selected by the masks
        # Currently, only full connections are supported
        tiled_pre_mask = tile(self.pre.size, self.spec.pre_mask)
        tiled_post_mask = tile(self.post.size, self.spec.post_mask)

        if self.spec.projn_type == "one_to_one":
            mask = expand_layer_mask_one_to_one(tiled_pre_mask,
                                                tiled_post_mask)
        elif self.spec.projn_type == "full":
            mask = expand_layer_mask_full(tiled_pre_mask, tiled_post_mask)

        # Enforce sparsity
        self.mask, num_nonzero = sparsify(self.spec.sparsity, mask)

        # Fill the weight matrix with values
        rand_nums = torch.Tensor(num_nonzero)
        self.spec.dist.fill(rand_nums)
        self.wts[self.mask] = rand_nums

        self.fwts = self.wts

        # Record the number of incoming connections for each unit
        self.num_recv_conns = torch.sum(self.mask, dim=1).float()

    def netin_scale(self) -> torch.Tensor:
        """Computes the net input scaling factor for each receiving unit.

        See https://grey.colorado.edu/emergent/index.php/Leabra_Netin_Scaling
        for details.

        Returns:
          A tensor of size self.post.size containing in each element the netin
          scaling factor for that unit.

        """
        sem_extra = 2.0

        pre_act_avg = self.pre.avg_act
        pre_act_n = max(1, round(pre_act_avg * self.pre.units.size))
        post_act_n_avg = torch.max(
            torch.Tensor([1]), (pre_act_avg * self.num_recv_conns).round())
        post_act_n_max = torch.min(self.num_recv_conns,
                                   torch.Tensor([pre_act_n]))
        post_act_n_exp = torch.min(post_act_n_max, post_act_n_avg + sem_extra)

        scaling_factors = 1.0 / post_act_n_exp
        full_connectivity = pre_act_n == post_act_n_avg
        scaling_factors[full_connectivity] = 1.0 / pre_act_n

        return scaling_factors

    def flush(self) -> None:
        """Propagates sending layer activation to the recieving layer.

        Separating this step from the activation and firing of the sending
        layer makes it easier to compute the net input scaling factor.

        """
        wt_scale_act = self.netin_scale()
        self.post.add_input(
            self.spec.wt_scale_abs * wt_scale_act *
            (self.wts @ self.pre.units.act), self.spec.wt_scale_rel)

    def learn(self) -> None:
        """Updates weights with XCAL learning equation."""
        # Compute weight changes
        srs = torch.ger(self.post.avg_s, self.pre.avg_s)
        srm = torch.ger(self.post.avg_m, self.pre.avg_m)
        s_mix = 0.9
        sm_mix = s_mix * srs + (1 - s_mix) * srm
        thr_l_mix = 0.05
        lthr = thr_l_mix * self.post.cos_diff_avg * torch.ger(
            self.post.avg_m, self.pre.avg_l)
        mthr = (1 - thr_l_mix * self.post.cos_diff_avg) * srm
        dwts = self.spec.lrate * xcal(sm_mix, lthr + mthr)
        dwts[~self.mask] = 0

        # Apply weights
        mask = dwts > 0
        dwts[mask] *= 1 - self.fwts[mask]
        dwts[~mask] *= self.fwts[~mask]
        self.fwts += dwts
        self.wts = sig(self.spec.sig_gain, self.spec.sig_offset, self.fwts)
        return

    def handle(self, event: events.Event) -> None:
        """Overrides `event.EventListenerMixin.handle()`."""
        if isinstance(event, events.Learn):
            self.learn()
