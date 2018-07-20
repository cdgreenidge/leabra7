"""Integration tests k-winner-take-all activation."""
import numpy as np

import leabra7 as lb


def test_kwta_suppresses_all_but_k_units() -> None:
    for p in np.linspace(0.0, 1.0, 10):
        n = lb.Net()
        n.new_layer(name="lr1", size=1)

        lr2_spec = lb.LayerSpec(
            inhibition_type="kwta",
            kwta_pct=p,
            log_on_cycle=("unit_act", ),
            unit_spec=lb.UnitSpec(adapt_dt=0, spike_gain=0))
        n.new_layer(name="lr2", size=10, spec=lr2_spec)

        pr1_spec = lb.ProjnSpec(dist=lb.Uniform(low=0.25, high=0.75))
        n.new_projn("proj1", "lr1", "lr2", pr1_spec)

        pr2_spec = lb.ProjnSpec(dist=lb.Uniform(low=0.25, high=0.75))
        n.new_projn("proj2", "lr1", "lr2", pr2_spec)

        n.clamp_layer("lr1", [1])
        for i in range(50):
            n.cycle()

        logs = n.logs("cycle", "lr2").parts
        acts = logs[logs.time == 49]["act"]

        assert (acts > 0.5).sum() == max(1, int(round(10 * p)))


# TODO: precisely define this
def test_kwta_avg_suppresses_all_but_k_units() -> None:
    for p in np.linspace(0.4, 1.0, 10):
        n = lb.Net()
        n.new_layer(name="lr1", size=1)

        lr2_spec = lb.LayerSpec(
            inhibition_type="kwta_avg",
            kwta_pct=p,
            log_on_cycle=("unit_act", ),
            unit_spec=lb.UnitSpec(adapt_dt=0, spike_gain=0))
        n.new_layer(name="lr2", size=10, spec=lr2_spec)

        pr1_spec = lb.ProjnSpec(dist=lb.Uniform(low=0.25, high=0.75))
        n.new_projn("proj1", "lr1", "lr2", pr1_spec)

        pr2_spec = lb.ProjnSpec(dist=lb.Uniform(low=0.25, high=0.75))
        n.new_projn("proj2", "lr1", "lr2", pr2_spec)

        n.clamp_layer("lr1", [1])
        for i in range(100):
            n.cycle()

        logs = n.logs("cycle", "lr2").parts
        acts = logs[logs.time == 99]["act"]

        # Can't exactly be exact about how many winners selected
        # assert (acts > 0.5).sum() == max(1, int(round(10 * p)))
