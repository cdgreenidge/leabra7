"""Tests whether a simple network can learn with xcal."""
from typing import List
from typing import Iterable

from leabra7 import specs
from leabra7 import net


def trial(network: net.Net, input_pattern: Iterable[float],
          output_pattern: Iterable[float]) -> None:
    """Runs a trial."""
    network.clamp_layer("input", input_pattern)
    network.minus_phase_cycle(num_cycles=50)
    network.clamp_layer("output", output_pattern)
    network.plus_phase_cycle(num_cycles=25)
    network.unclamp_layer("input")
    network.unclamp_layer("output")
    network.learn()


def epoch(network: net.Net, input_patterns: Iterable[Iterable[float]],
          output_patterns: Iterable[Iterable[float]]) -> None:
    """Runs an epoch."""
    for in_pattern, out_pattern in zip(input_patterns, output_patterns):
        for _ in range(10):
            trial(network, in_pattern, out_pattern)


def batch(network: net.Net, input_patterns: Iterable[Iterable[float]],
          output_patterns: Iterable[Iterable[float]]) -> None:
    """Runs a training batch."""
    num_epochs = 2
    for _ in range(num_epochs):
        epoch(network, input_patterns, output_patterns)


def output(network: net.Net, pattern: Iterable[float]) -> List[float]:
    network.clamp_layer("input", pattern)
    for _ in range(50):
        network.cycle()
    # We skip logging for speed
    out = network.layers["output"].units.act.numpy()
    out[out > 0.7] = 1
    out[out < 0.1] = 0
    return list(out)


def test_a_simple_network_can_learn_simple_things() -> None:
    network = net.Net()
    network.new_layer("input", size=2)
    network.new_layer("output", size=2)
    projn_spec = specs.ProjnSpec(lrate=0.2)
    network.new_projn(
        "input_to_output", pre="input", post="output", spec=projn_spec)

    input_patterns = [[1, 0], [0, 1]]
    output_patterns = [[0, 1], [1, 0]]
    batch(network, input_patterns, output_patterns)

    actual = [output(network, i) for i in input_patterns]

    for act, exp in zip(actual, output_patterns):
        assert act == exp
