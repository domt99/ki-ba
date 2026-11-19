from abc import ABC, abstractmethod
import numpy as np


class Layer(ABC):
    @abstractmethod
    def forward_step(self, neuron_values_prev):
        pass


class InputLayer(Layer):
    def __init__(self, neuron_count):
        self.neuron_count = neuron_count

    def forward_step(self, neuron_values_prev):
        if neuron_values_prev.shape[1] != self.neuron_count:
            raise ValueError("Input Matrix doesn't have the correct dimension")
        return neuron_values_prev


class DenseLayer(Layer):
    def __init__(self, calculation, neuron_count_prev, neuron_count, weight_rng):
        self.calculation = calculation
        self.neuron_count_prev = neuron_count_prev
        self.neuron_count = neuron_count

        self.neuron_values_ext = None
        
        
        self.weights = self._initialize_weights(weight_rng)

    def forward_step(self, neuron_values_prev):
        self.neuron_values_ext = np.hstack([-np.ones((len(neuron_values_prev), 1)), neuron_values_prev])
        net = self.calculation.calculate_net(self.neuron_values_ext, self.weights)
        return self.activate(net)

    def backward_step(self, backprop_error):
        delta = self.calculate_delta(backprop_error)
        delta_prev_layer = self.calculation.calculate_backprop_error(delta, self.weights)
        gradients = self.calculation.calculate_gradient(self.neuron_values_ext, delta)
        return gradients, delta_prev_layer

    def get_weights(self):
        return self.weights

    def set_weights(self, new_weights):
        if new_weights.shape != self.weights.shape:
            raise TypeError("New weights have wrong dimension")
        self.weights = new_weights

    def _initialize_weights(self, weight_rng):
        std = np.sqrt(2 / self.neuron_count_prev)
        weights = weight_rng.normal(loc=0.0, scale=std, size=(self.neuron_count_prev, self.neuron_count))
        weights_zeros = np.zeros((1, self.neuron_count))
        return np.vstack([weights_zeros, weights])

    @abstractmethod
    def activate(self, net):
        pass

    @abstractmethod
    def calculate_delta(self, backprop_error):
        pass


class HiddenLayer(DenseLayer):
    def __init__(self, calculation, neuron_count_prev, neuron_count, activation_func, weight_rng):
        ...

    def activate(self, net):
        ...

    def calculate_delta(self, backprop_error):
        ...


class OutputLayer(DenseLayer):
    def __init__(self, calculation, neuron_count_prev, neuron_count, weight_rng):
        ...

    def activate(self, net):
        ...

    def calculate_delta(self, target_value):
        ...

    def _softmax(self, net):
        ...

    def categorical_cross_entropy(self, prediction, target_value):
        ...