"""3c: gradient check - backpropagation vs. difference quotient.

  numerical = (E(w_ij + h) - E(w_ij - h)) / (2h)   with h = 1e-6
  relative deviation should be below 1e-6.
"""
import numpy as np
import pytest
from ki_ba.activation_function import ReLU, SigmoidLog

from tests.helpers import build_mlp

H = 1e-6
MAX_RELATIVE_ERROR = 1e-6
TOPOLOGY = [3, 4, 3, 2]  # small net: every weight is checked
INSTANCE_COUNT = 5


def numerical_gradient(mlp, inputs, targets, layer_index, i, j, h=H):
    """dE/dw_ij of layer `layer_index` via the central difference quotient."""
    # TODO 3c: change one weight by +h and -h (get_weights / set_weights), compute E
    # each time (forward_propagation + calculate_loss), restore the weight afterwards.
    # Careful: does get_weights return a copy or the array the layer computes with?
    raise NotImplementedError


def relative_error(backprop, numerical):
    # TODO 3c: which formula? What if both values are (almost) 0?
    raise NotImplementedError


@pytest.fixture
def random_data():
    rng = np.random.default_rng(0)
    inputs = rng.normal(size=(INSTANCE_COUNT, TOPOLOGY[0]))
    targets = np.eye(TOPOLOGY[-1])[rng.integers(TOPOLOGY[-1], size=INSTANCE_COUNT)]
    return inputs, targets


@pytest.mark.parametrize("calculation", ["loop", "matrix"])
@pytest.mark.parametrize("activation_func", [SigmoidLog(), ReLU()], ids=["sigmoid_log", "relu"])
def test_gradient_check(calculation, activation_func, random_data):
    inputs, targets = random_data
    mlp = build_mlp(TOPOLOGY, activation_func, calculation, seed=1)
    # random thresholds too, the default initialization sets them to 0
    rng = np.random.default_rng(2)
    mlp.set_weights([rng.normal(size=w.shape) for w in mlp.get_weights()])

    mlp.forward_propagation(inputs)
    gradients = mlp.backward_propagation(targets)

    # TODO 3c: for every layer and every (i, j): compare gradients[layer][i, j] with
    # numerical_gradient(...) and collect the deviations above MAX_RELATIVE_ERROR.
    # - Is E in calculate_loss a sum or a mean over the instances, and is your
    #   gradient scaled the same way?
    # - ReLU is not differentiable at 0: what happens if a net is within h of 0?
    pytest.skip("TODO 3c: compare every derivative")
