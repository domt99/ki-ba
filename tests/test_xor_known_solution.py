"""3b: test with a known solution (slides "XOR-MLP: vorwärts" and "XOR-MLP: rückwärts").

Fill in the slide values below. Each weight matrix has shape (neuron_count_prev + 1,
neuron_count): row 0 = thresholds theta (fixed input -1), rows 1.. = weights from the
previous layer. Until a value is filled in, the tests that need it are skipped.
Runs with loop and matrix, so (4c) "all tests from (iii) still pass" is covered too.
"""
import numpy as np
import pytest
from ki_ba.activation_function import ReLU
from ki_ba.optimizer import GradientDescent

from tests.helpers import build_mlp

INPUT = np.array([[1.0, 0.0]])
TARGET = np.array([[0.0, 1.0]])  # x = (1, 0) is class 1
LEARNING_RATE = 0.5

# TODO 3b: values from slide "XOR-MLP: vorwärts"
WEIGHTS_HIDDEN = None   # shape (3, 2): [[theta_3, theta_4], [w_13, w_14], [w_23, w_24]]
WEIGHTS_OUTPUT = None   # shape (3, 2): [[theta_5, theta_6], [w_35, w_36], [w_45, w_46]]
EXPECTED_HIDDEN = None  # (o_3, o_4)
EXPECTED_OUTPUT = None  # (o_5, o_6) before the step

# TODO 3b: values from slide "XOR-MLP: rückwärts"
EXPECTED_GRADIENT_HIDDEN = None  # dE/dw, same shape as WEIGHTS_HIDDEN
EXPECTED_GRADIENT_OUTPUT = None  # dE/dw, same shape as WEIGHTS_OUTPUT

# from the task sheet: after one step with eta = 0.5
EXPECTED_OUTPUT_AFTER_STEP = (0.08, 0.92)
# TODO 3b: the slides round to 2 decimals - which tolerance fits?
TOLERANCE = None


def require(*values):
    if any(value is None for value in values):
        pytest.skip("TODO 3b: fill in the slide values at the top of this file")


@pytest.fixture(params=["loop", "matrix"])
def xor_mlp(request):
    require(WEIGHTS_HIDDEN, WEIGHTS_OUTPUT)
    mlp = build_mlp([2, 2, 2], ReLU(), request.param)
    mlp.set_weights([np.array(WEIGHTS_HIDDEN, float), np.array(WEIGHTS_OUTPUT, float)])
    return mlp


def test_forward(xor_mlp):
    require(EXPECTED_HIDDEN, EXPECTED_OUTPUT, TOLERANCE)
    output = xor_mlp.forward_propagation(INPUT)
    # TODO 3b: where do you get o_3, o_4 from after the forward step?
    # Compare hidden and output values with the slide (np.allclose with atol=TOLERANCE).
    pytest.skip("TODO 3b: compare forward values")


def test_backward(xor_mlp):
    require(EXPECTED_GRADIENT_HIDDEN, EXPECTED_GRADIENT_OUTPUT, TOLERANCE)
    xor_mlp.forward_propagation(INPUT)
    gradients = xor_mlp.backward_propagation(TARGET)
    # TODO 3b: compare gradients[0] / gradients[1] with the slide.
    # Does the slide show dE/dw or the weight change -eta * dE/dw?
    pytest.skip("TODO 3b: compare gradients")


def test_one_step(xor_mlp):
    require(TOLERANCE)
    optimizer = GradientDescent(learning_rate=LEARNING_RATE)
    # TODO 3b: one training step - which calls in which order?
    # (forward, backward, optimizer, set weights), then forward again and compare
    # with EXPECTED_OUTPUT_AFTER_STEP.
    pytest.skip("TODO 3b: one gradient descent step")
