"""4c: the vectorized layer gives the same outputs and derivatives as the loop version."""
import numpy as np
import pytest
from ki_ba.activation_function import ReLU, SigmoidLog

from tests.helpers import build_mlp

TOPOLOGIES = [[2, 4, 2], [5, 7, 3], [10, 8, 6, 4]]


@pytest.mark.parametrize("topology", TOPOLOGIES, ids=lambda t: "-".join(map(str, t)))
@pytest.mark.parametrize("activation_func", [SigmoidLog(), ReLU()], ids=["sigmoid_log", "relu"])
def test_loop_equals_matrix(topology, activation_func):
    rng = np.random.default_rng(3)
    inputs = rng.normal(size=(6, topology[0]))
    targets = np.eye(topology[-1])[rng.integers(topology[-1], size=6)]

    # same seed -> same start weights in both nets
    loop_mlp = build_mlp(topology, activation_func, "loop", seed=7)
    matrix_mlp = build_mlp(topology, activation_func, "matrix", seed=7)

    # TODO 4c: forward and backward with both nets, then compare outputs and every
    # gradient matrix with np.allclose. Why allclose instead of ==?
    pytest.skip("TODO 4c: compare loop and matrix")
