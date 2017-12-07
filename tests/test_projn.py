"""Test projn.py"""
from leabra7 import unit as un
from leabra7 import layer as lr
from leabra7 import projn as pr
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
    conn = pr.Conn("con1", pre, post)
    assert conn.wt > 0


def test_make_full_conn_list_returns_a_full_connection_list():
    pre = lr.Layer(name="pre", size=3)
    post = lr.Layer(name="post", size=3)
    conns = pr.make_full_conn_list("proj", pre, post, sp.ConnSpec())
    units = [(u, v) for u in pre.units for v in post.units]
    assert [c.pre for c in conns] == [u for u, _ in units]
    assert [c.post for c in conns] == [v for _, v in units]


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


def test_projn_can_flush():
    pre = lr.Layer("lr1", size=1)
    post = lr.Layer("lr2", size=1)
    projn = pr.Projn("proj", pre, post)
    projn.flush()
