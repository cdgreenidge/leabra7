"""Test projn.py"""
from leabra7 import unit as un
from leabra7 import layer as lr
from leabra7 import projn as pr
from leabra7 import rand as rn
from leabra7 import specs as sp


def test_conn_init_uses_the_spec_you_pass_it() -> None:
    spec = sp.ConnSpec()
    pre = un.Unit()
    post = un.Unit()
    conn = pr.Conn("proj1", pre, post, spec=spec)
    assert conn.spec is spec


def test_conn_has_a_name() -> None:
    pre = un.Unit()
    post = un.Unit()
    conn = pr.Conn("con1", pre, post)
    assert conn.name == "con1"


def test_projn_has_a_sending_unit() -> None:
    pre = un.Unit()
    post = un.Unit()
    conn = pr.Conn("con1", pre, post)
    assert conn.pre == pre


def test_conn_has_a_receiving_unit() -> None:
    pre = un.Unit()
    post = un.Unit()
    conn = pr.Conn("con1", pre, post)
    assert conn.post == post


def test_conn_has_a_weight() -> None:
    pre = un.Unit()
    post = un.Unit()
    conn = pr.Conn("con1", pre, post, sp.ConnSpec(dist=rn.Scalar(0.3)))
    assert conn.wt == 0.3


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


def test_projns_can_be_sparse() -> None:
    pre = lr.Layer("lr1", size=2)
    post = lr.Layer("lr2", size=2)
    spec = sp.ProjnSpec(dist=rn.Scalar(1.0), sparsity=0.5)
    projn = pr.Projn("proj", pre, post, spec)
    num_on = projn.wts.sum()
    assert num_on == 2.0
