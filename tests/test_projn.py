"""Test projn.py"""
from leabra7 import unit as un
from leabra7 import layer as lr
from leabra7 import projn as pr
from leabra7 import random as rn
from leabra7 import specs as sp


def test_conn_init_uses_the_spec_you_pass_it():
    spec = sp.ConnSpec()
    pre = un.Unit()
    post = un.Unit()
    conn = pr.Conn("proj1", pre, post, spec=spec)
    assert conn.spec is spec


def test_conn_has_a_name():
    pre = un.Unit()
    post = un.Unit()
    conn = pr.Conn("con1", pre, post)
    assert conn.name == "con1"


def test_projn_has_a_sending_unit():
    pre = un.Unit()
    post = un.Unit()
    conn = pr.Conn("con1", pre, post)
    assert conn.pre == pre


def test_conn_has_a_receiving_unit():
    pre = un.Unit()
    post = un.Unit()
    conn = pr.Conn("con1", pre, post)
    assert conn.post == post


def test_conn_has_a_weight():
    pre = un.Unit()
    post = un.Unit()
    conn = pr.Conn("con1", pre, post, sp.ConnSpec(dist=rn.Scalar(0.3)))
    assert conn.wt == 0.3


def test_make_full_conn_list_returns_a_full_connection_list():
    pre = lr.Layer(name="pre", size=3)
    post = lr.Layer(name="post", size=3)
    conns = pr.make_full_conn_list("proj", pre.units, post.units,
                                   sp.ConnSpec())
    units = [(u, v) for u in pre.units for v in post.units]
    assert [c.pre for c in conns] == [u for u, _ in units]
    assert [c.post for c in conns] == [v for _, v in units]


def test_mask_masks_an_iterable():
    xs = [1, 2, 3, 4]
    xs_mask = [True, False, True, False]
    assert pr.mask(xs, xs_mask) == [1, 3]


def test_projn_has_a_name():
    pre = lr.Layer("lr1", size=1)
    post = lr.Layer("lr2", size=1)
    projn = pr.Projn("proj", pre, post)
    assert projn.name == "proj"


def test_projn_has_a_sending_layer():
    pre = lr.Layer("lr1", size=1)
    post = lr.Layer("lr2", size=1)
    projn = pr.Projn("proj", pre, post)
    assert projn.pre == pre


def test_projn_has_a_receiving_layer():
    pre = lr.Layer("lr1", size=1)
    post = lr.Layer("lr2", size=1)
    projn = pr.Projn("proj", pre, post)
    assert projn.post == post


def test_projn_can_specify_its_weight_distribution():
    pre = lr.Layer("lr1", size=3)
    post = lr.Layer("lr2", size=3)
    projn = pr.Projn("proj", pre, post, sp.ProjnSpec(dist=rn.Scalar(7)))
    for conn in projn.conns:
        assert conn.wt == 7


def test_projn_can_flush():
    pre = lr.Layer("lr1", size=1)
    post = lr.Layer("lr2", size=1)
    projn = pr.Projn("proj", pre, post)
    projn.flush()


def test_projn_can_mask_pre_layer_units():
    pre = lr.Layer("lr1", size=2)
    post = lr.Layer("lr2", size=2)
    spec = sp.ProjnSpec(pre_mask=(True, False))
    projn = pr.Projn("proj", pre, post, spec)
    for i in projn.conns:
        assert i.pre == pre.units[0]


def test_projn_pre_mask_tiles_if_it_is_too_short():
    pre = lr.Layer("lr1", size=4)
    post = lr.Layer("lr2", size=1)
    spec = sp.ProjnSpec(pre_mask=(True, False))
    projn = pr.Projn("proj", pre, post, spec)
    assert projn.conns[0].pre == pre.units[0]
    assert projn.conns[1].pre == pre.units[2]


def test_projn_pre_mask_truncates_if_it_is_too_long():
    pre = lr.Layer("lr1", size=1)
    post = lr.Layer("lr2", size=1)
    spec = sp.ProjnSpec(pre_mask=(True, False))
    projn = pr.Projn("proj", pre, post, spec)
    assert projn.conns[0].pre == pre.units[0]
    assert len(projn.conns) == 1


def test_projn_can_mask_post_layer_units():
    pre = lr.Layer("lr1", size=2)
    post = lr.Layer("lr2", size=2)
    spec = sp.ProjnSpec(post_mask=(False, True))
    projn = pr.Projn("proj", pre, post, spec)
    for i in projn.conns:
        assert i.post == post.units[1]


def test_projn_post_mask_tiles_if_it_is_too_short():
    pre = lr.Layer("lr1", size=1)
    post = lr.Layer("lr2", size=4)
    spec = sp.ProjnSpec(post_mask=(True, False))
    projn = pr.Projn("proj", pre, post, spec)
    assert projn.conns[0].post == post.units[0]
    assert projn.conns[1].post == post.units[2]


def test_projn_post_mask_truncates_if_it_is_too_long():
    pre = lr.Layer("lr1", size=1)
    post = lr.Layer("lr2", size=1)
    spec = sp.ProjnSpec(post_mask=(True, False))
    projn = pr.Projn("proj", pre, post, spec)
    assert projn.conns[0].post == post.units[0]
    assert len(projn.conns) == 1
