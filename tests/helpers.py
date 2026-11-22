import numpy as np
from ki_ba.calculation import CalculateLoop, CalculateMatrix
from ki_ba.layer import HiddenLayer, InputLayer, OutputLayer
from ki_ba.mlp import MLP

CALCULATIONS = {"loop": CalculateLoop, "matrix": CalculateMatrix}


def build_mlp(topology, activation_func, calculation, seed=42):
    """Build an MLP directly from layers (no .env needed).

    topology: e.g. [2, 2, 2]; activation_func: one instance for all hidden layers;
    calculation: "loop" or "matrix". Weight matrices have shape
    (neuron_count_prev + 1, neuron_count), row 0 holds the thresholds (input -1).
    """
    weight_rng = np.random.default_rng(seed)
    calculation = CALCULATIONS[calculation]()
    input_layer = InputLayer(topology[0])
    hidden_layers = [
        HiddenLayer(
            calculation=calculation,
            neuron_count_prev=topology[i - 1],
            neuron_count=topology[i],
            activation_func=activation_func,
            weight_rng=weight_rng,
        )
        for i in range(1, len(topology) - 1)
    ]
    output_layer = OutputLayer(
        calculation=calculation,
        neuron_count_prev=topology[-2],
        neuron_count=topology[-1],
        weight_rng=weight_rng,
    )
    return MLP(input_layer, hidden_layers, output_layer)
