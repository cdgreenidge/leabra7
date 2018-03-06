"""Integration tests k-winner-take-all activation."""
from leabra7 import net
from leabra7 import specs as sp
from leabra7 import random as rn


def test_kwta_suppresses_all_but_k_units():
    n = net.Net()

    n.new_layer(name="lr1", size=1)

    lr2_spec = sp.LayerSpec(
        inhibition_type="kwta",
        k=2,
        log_on_cycle=("unit_act", ),
        unit_spec=sp.UnitSpec(adapt_dt=0, spike_gain=0))
    n.new_layer(name="lr2", size=3, spec=lr2_spec)

    pr1_spec = sp.ProjnSpec(dist=rn.Scalar(0.3))
    n.new_projn("proj1", "lr1", "lr2", pr1_spec)

    pr2_spec = sp.ProjnSpec(dist=rn.Scalar(0.5), post_mask=[0, 1, 1])
    n.new_projn("proj2", "lr1", "lr2", pr2_spec)

    n.force_layer("lr1", [1])
    for i in range(100):
        n.cycle()

    acts = n.logs("cycle", "lr2").tail(1).as_matrix()
    assert (acts > 0.8).sum() == 2
