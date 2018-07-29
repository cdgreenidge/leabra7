"""Test projn.py"""
from hypothesis import given
import hypothesis.strategies as st
import pytest
import torch  # type: ignore

from leabra7 import events as ev
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
    projn = pr.Projn("proj", pre, post, spec=sp.ProjnSpec(dist=rn.Scalar(7)))
    assert (projn.wts == 7).all()


def test_projn_can_flush() -> None:
    pre = lr.Layer("lr1", size=1)
    post = lr.Layer("lr2", size=1)
    projn = pr.Projn("proj", pre, post)
    projn.flush()


def test_projn_can_inhibit_flush() -> None:
    pre = lr.Layer("lr1", size=1)
    post = lr.Layer("lr2", size=1)
    projn = pr.Projn("proj", pre, post)

    pre.hard_clamp(act_ext=[1])
    projn.inhibit()

    projn.flush()

    assert post.input_buffer == 0.0


def test_projn_can_uninhibit_flush() -> None:
    pre = lr.Layer("lr1", size=1)
    post = lr.Layer("lr2", size=1)
    projn = pr.Projn("proj", pre, post)

    pre.hard_clamp(act_ext=[1])

    projn.inhibit()
    projn.flush()
    projn.uninhibit()
    projn.flush()

    assert post.input_buffer == 0.5


def test_projn_inhibit_handling_event() -> None:
    pre = lr.Layer("lr1", size=1)
    post = lr.Layer("lr2", size=1)
    projn = pr.Projn("proj", pre, post)

    pre.hard_clamp(act_ext=[1])
    projn.handle(ev.InhibitProjns("proj"))

    projn.flush()

    assert post.input_buffer == 0.0


def test_projn_can_uninhibit_flush() -> None:
    pre = lr.Layer("lr1", size=1)
    post = lr.Layer("lr2", size=1)
    projn = pr.Projn("proj", pre, post)

    pre.hard_clamp(act_ext=[1])

    projn.handle(ev.InhibitProjns("proj"))
    projn.flush()
    projn.handle(ev.UninhibitProjns("proj"))
    projn.flush()

    assert post.input_buffer == 0.5


def test_projn_can_mask_pre_layer_units() -> None:
    pre = lr.Layer("lr1", size=2)
    post = lr.Layer("lr2", size=2)
    mask = (True, False)
    spec = sp.ProjnSpec(pre_mask=mask, dist=rn.Scalar(1))
    projn = pr.Projn("proj", pre, post, spec=spec)
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
    projn = pr.Projn("proj", pre, post, spec=spec)
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
    projn = pr.Projn("proj", pre, post, spec=spec)
    assert projn.wts[0, 0] == 1
    assert projn.wts.shape == (1, 1)


def test_projn_can_mask_post_layer_units() -> None:
    pre = lr.Layer("lr1", size=2)
    post = lr.Layer("lr2", size=2)
    mask = (True, False)
    spec = sp.ProjnSpec(post_mask=mask, dist=rn.Scalar(1))
    projn = pr.Projn("proj", pre, post, spec=spec)
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


def test_expand_layer_mask_one_to_one_tests_unit_count() -> None:
    pre_mask = [True, False, True, True]
    post_mask = [False, False, True, True]

    with pytest.raises(ValueError):
        pr.expand_layer_mask_one_to_one(pre_mask, post_mask)


def test_expand_layer_mask_one_to_one_has_the_correct_connectivity_pattern(
) -> None:
    pre_mask = [True, False, False, True]
    post_mask = [False, True, True, False]
    # yapf: disable
    expected = torch.ByteTensor(
        [[False, True, False, False],
         [False, False, False, False],
         [False, False, False, False],
         [False, False, True, False]]
    )
    # yapf: enable
    actual = pr.expand_layer_mask_one_to_one(pre_mask, post_mask)
    assert (actual == expected).all()


def test_projn_one_to_one_connectivity_pattern_is_correct() -> None:
    pre = lr.Layer("lr1", size=3)
    post = lr.Layer("lr2", size=3)
    projn = pr.Projn(
        "proj",
        pre,
        post,
        spec=sp.ProjnSpec(projn_type="one_to_one", dist=rn.Scalar(1.0)))
    assert (projn.wts == torch.eye(3)).all()


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
    projn = pr.Projn("proj", pre, post, spec=spec)
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
    projn = pr.Projn("proj", pre, post, spec=spec)
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

    pre_a.hard_clamp(torch.ones(x) * f)
    pre_b.hard_clamp(torch.ones(y) * f)

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

    pre_a.hard_clamp(torch.ones(x) * f)
    pre_b.hard_clamp(torch.ones(x) * f)

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


def test_projn_can_learn() -> None:
    pre = lr.Layer("lr1", size=2)
    post = lr.Layer("lr2", size=2)
    projn = pr.Projn("proj", pre, post)
    projn.learn()


def test_projn_can_handle_learn_events(mocker) -> None:
    pre = lr.Layer("lr1", size=2)
    post = lr.Layer("lr2", size=2)
    projn = pr.Projn("proj", pre, post)
    mocker.spy(projn, "learn")
    projn.handle(ev.Learn())
    projn.learn.assert_called_once()


def test_you_can_log_projection_weights() -> None:
    pre = lr.Layer("lr1", size=2)
    post = lr.Layer("lr2", size=2)
    projn = pr.Projn(
        "proj",
        pre,
        post,
        spec=sp.ProjnSpec(projn_type="one_to_one", dist=rn.Scalar(0.5)))
    expected = {"pre_unit": [0, 1], "post_unit": [0, 1], "conn_wt": [0.5, 0.5]}
    assert projn.observe_parts_attr("conn_wt") == expected


def test_you_can_log_projection_fast_weights() -> None:
    pre = lr.Layer("lr1", size=2)
    post = lr.Layer("lr2", size=2)
    projn = pr.Projn(
        "proj",
        pre,
        post,
        spec=sp.ProjnSpec(projn_type="one_to_one", dist=rn.Scalar(0.5)))
    expected = {
        "pre_unit": [0, 1],
        "post_unit": [0, 1],
        "conn_fwt": [0.5, 0.5]
    }
    assert projn.observe_parts_attr("conn_fwt") == expected


def test_observing_invalid_parts_attr_raises_value_error() -> None:
    pre = lr.Layer("lr1", size=2)
    post = lr.Layer("lr2", size=2)
    projn = pr.Projn("proj", pre, post)
    with pytest.raises(ValueError):
        projn.observe_parts_attr("whales")
