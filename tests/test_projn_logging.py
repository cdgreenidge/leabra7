"""Integration tests projection logging."""
from typing import Iterable

import pandas as pd

import leabra7 as lb


def trial(network: lb.Net, input_pattern: Iterable[float],
          output_pattern: Iterable[float]) -> None:
    """Runs a trial."""
    network.clamp_layer("input", input_pattern)
    network.minus_phase_cycle(num_cycles=50)
    network.clamp_layer("output", output_pattern)
    network.plus_phase_cycle(num_cycles=25)
    network.unclamp_layer("input")
    network.unclamp_layer("output")


def test_you_can_log_projection_weights() -> None:
    network = lb.Net()
    network.new_layer("input", size=2)
    network.new_layer("output", size=2)
    projn_spec = lb.ProjnSpec(log_on_trial=["conn_wt"])
    network.new_projn(
        "input_to_output", pre="input", post="output", spec=projn_spec)

    trial(network, (1, 0), (0, 1))

    _, part_logs = network.logs(freq="trial", name="input_to_output")

    expected = pd.DataFrame({
        "pre_unit": (0, 1, 0, 1),
        "conn_wt": (0.5, 0.5, 0.5, 0.5),
        "post_unit": (0, 0, 1, 1),
        "time": (0, 0, 0, 0)
    })

    pd.util.testing.assert_frame_equal(part_logs, expected, check_like=True)
