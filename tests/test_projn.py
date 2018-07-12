"""Test projn.py"""
from hypothesis import given
import hypothesis.strategies as st
import torch  # type: ignore

from leabra7 import layer as lr
from leabra7 import projn as pr
from leabra7 import rand as rn
from leabra7 import specs as sp


def test_projn_has_a_name() -> None:
    pre = lr.Layer("lr1", size=1)
    post = lr.Layer("lr2", size=1)
    projn = pr.Projn("proj", pre, post)
    assert projn.name == "proj"


def test_projn_has_a_sending_layer() -> None:
    pre = lr.Layer("lr1", size=1)
    post = lr.Layer("lr2", size=1)
    projn = pr.Projn("proj", pre, post)
    assert projn.pre == pre


def test_projn_has_a_receiving_layer() -> None:
    pre = lr.Layer("lr1", size=1)
    post = lr.Layer("lr2", size=1)
    projn = pr.Projn("proj", pre, post)
    assert projn.post == post


def test_projn_can_specify_its_weight_distribution() -> None:
    pre = lr.Layer("lr1", size=3)
    post = lr.Layer("lr2", size=3)
    projn = pr.Projn("proj", pre, post, sp.ProjnSpec(dist=rn.Scalar(7)))
    assert (projn.wts == 7).all()


def test_projn_can_flush() -> None:
    pre = lr.Layer("lr1", size=1)
    post = lr.Layer("lr2", size=1)
    projn = pr.Projn("proj", pre, post)
    projn.flush()


def test_projn_can_mask_pre_layer_units() -> None:
    pre = lr.Layer("lr1", size=2)
    post = lr.Layer("lr2", size=2)
    mask = (True, False)
    spec = sp.ProjnSpec(pre_mask=mask, dist=rn.Scalar(1))
    projn = pr.Projn("proj", pre, post, spec)
    for i in range(post.size):
        for j in range(pre.size):
            if mask[j]:
                assert projn.wts[i, j] == 1
            else:
                assert projn.wts[i, j] == 0


def test_projn_pre_mask_tiles_if_it_is_too_short() -> None:
    pre = lr.Layer("lr1", size=4)
    post = lr.Layer("lr2", size=2)
    mask = (True, False)
    spec = sp.ProjnSpec(pre_mask=mask, dist=rn.Scalar(1))
    projn = pr.Projn("proj", pre, post, spec)
    for i in range(post.size):
        for j in range(pre.size):
            if mask[j % 2]:
                assert projn.wts[i, j] == 1
            else:
                assert projn.wts[i, j] == 0


def test_projn_pre_mask_truncates_if_it_is_too_long() -> None:
    pre = lr.Layer("lr1", size=1)
    post = lr.Layer("lr2", size=1)
    spec = sp.ProjnSpec(pre_mask=(True, False), dist=rn.Scalar(1))
    projn = pr.Projn("proj", pre, post, spec)
    assert projn.wts[0, 0] == 1
    assert projn.wts.shape == (1, 1)


def test_projn_can_mask_post_layer_units() -> None:
    pre = lr.Layer("lr1", size=2)
    post = lr.Layer("lr2", size=2)
    mask = (True, False)
    spec = sp.ProjnSpec(post_mask=mask, dist=rn.Scalar(1))
    projn = pr.Projn("proj", pre, post, spec)
    for i in range(post.size):
        for j in range(pre.size):
            if mask[i]:
                assert projn.wts[i, j] == 1
            else:
                assert projn.wts[i, j] == 0


def test_tile_can_tile_an_iterable() -> None:
    xs = [0, 1]
    assert pr.tile(4, xs) == [0, 1, 0, 1]


def test_expand_layer_mask_full_has_the_correct_connectivity_pattern() -> None:
    pre_mask = [True, False, True, True]
    post_mask = [True, True, True, False]
    # yapf: disable
    expected = torch.ByteTensor(
        [[True, False, True, True],
         [True, False, True, True],
         [True, False, True, True],
         [False, False, False, False]]
    )
    # yapf: enable
    assert (pr.expand_layer_mask_full(pre_mask, post_mask) == expected).all()


# TODO: turn this into a Hypothesis test
def test_sparsify_can_make_a_matrix_sparse() -> None:
    original = torch.ByteTensor([0, 1, 1, 0, 0, 0, 1, 0, 1])
    sparse, num_nonzero = pr.sparsify(0.75, original)
    assert sparse.sum() == num_nonzero
    assert num_nonzero == 3
    assert sparse.shape == original.shape


def test_projn_post_mask_tiles_if_it_is_too_short() -> None:
    pre = lr.Layer("lr1", size=2)
    post = lr.Layer("lr2", size=4)
    mask = (True, False)
    spec = sp.ProjnSpec(post_mask=mask, dist=rn.Scalar(1))
    projn = pr.Projn("proj", pre, post, spec)
    for i in range(post.size):
        for j in range(pre.size):
            if mask[i % 2]:
                assert projn.wts[i, j] == 1
            else:
                assert projn.wts[i, j] == 0


def test_projn_post_mask_truncates_if_it_is_too_long() -> None:
    pre = lr.Layer("lr1", size=1)
    post = lr.Layer("lr2", size=1)
    spec = sp.ProjnSpec(post_mask=(True, False), dist=rn.Scalar(1))
    projn = pr.Projn("proj", pre, post, spec)
    assert projn.wts[0, 0] == 1
    assert projn.wts.shape == (1, 1)


@given(
    x=st.integers(min_value=1, max_value=10),
    y=st.integers(min_value=1, max_value=10),
    z=st.integers(min_value=1, max_value=10),
    f=st.floats(min_value=0.0, max_value=1.0))
def test_projn_can_calculate_netin_scale_with_full_connectivity(x, y, z,
                                                                f) -> None:
    pre_a = lr.Layer("lr1", size=x)
    pre_b = lr.Layer("lr2", size=y)
    post = lr.Layer("lr3", size=z)

    pre_a.force(torch.ones(x) * f)
    pre_b.force(torch.ones(y) * f)

    projn_a = pr.Projn("proj1", pre_a, post)
    projn_b = pr.Projn("proj2", pre_b, post)

    projn_a_scale = projn_a.netin_scale()
    projn_b_scale = projn_b.netin_scale()

    if x > y:
        compare_tensor = projn_a_scale > projn_b_scale
    elif x < y:
        compare_tensor = projn_a_scale < projn_b_scale
    else:
        compare_tensor = projn_a_scale != projn_b_scale

    assert torch.sum(compare_tensor) == 0


@given(
    x=st.integers(min_value=1, max_value=10),
    z=st.integers(min_value=1, max_value=10),
    m=st.integers(min_value=1, max_value=3),
    n=st.integers(min_value=1, max_value=3),
    f=st.floats(min_value=0.0, max_value=1.0))
def test_projn_can_calculate_netin_scale_with_partial_connectivity(
        x, z, m, n, f) -> None:

    pre_a = lr.Layer("lr1", size=x)
    pre_b = lr.Layer("lr2", size=x)
    post = lr.Layer("lr3", size=z)

    spec = sp.ProjnSpec(post_mask=(True, ) * m + (False, ) * n)

    pre_a.force(torch.ones(x) * f)
    pre_b.force(torch.ones(x) * f)

    projn_a = pr.Projn("proj1", pre_a, post)
    projn_b = pr.Projn("proj2", pre_b, post, spec)

    projn_a_scale = projn_a.netin_scale()
    projn_b_scale = projn_b.netin_scale()

    assert torch.sum(projn_a_scale > projn_b_scale) == 0


def test_projns_can_be_sparse() -> None:
    pre = lr.Layer("lr1", size=2)
    post = lr.Layer("lr2", size=2)
    spec = sp.ProjnSpec(dist=rn.Scalar(1.0), sparsity=0.5)
    projn = pr.Projn("proj", pre, post, spec)
    num_on = projn.wts.sum()
    assert num_on == 2.0
