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
        if self.neuron_values_ext is None:
            raise RuntimeError("forward_step must be called before backward_step")
        delta = self.calculate_delta(backprop_error)
        delta_prev_layer = self.calculation.calculate_backprop_error(delta, self.weights)
        gradients = self.calculation.calculate_gradient(self.neuron_values_ext, delta)
        return gradients, delta_prev_layer

    def get_weights(self):
        return self.weights

    def count_parameters(self):
        # weights include the bias row, see _initialize_weights
        return self.weights.size

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
        super().__init__(calculation, neuron_count_prev, neuron_count, weight_rng)
        self.net = None
        self.activation_func = activation_func

    def activate(self, net):
        self.net = net
        return self.activation_func.activation(net)

    def calculate_delta(self, backprop_error):
        return backprop_error * self.activation_func.activation_derivative(self.net)


class OutputLayer(DenseLayer):
    def __init__(self, calculation, neuron_count_prev, neuron_count, weight_rng):
        super().__init__(calculation, neuron_count_prev, neuron_count, weight_rng)
        self.prediction = None

    def activate(self, net):
        self.prediction = self._softmax(net) 
        return self.prediction

    def calculate_delta(self, target_value):
        return self.prediction - target_value

    def _softmax(self, net):
        shifted = net - net.max(axis=1, keepdims=True)
        e = np.exp(shifted)
        return e / e.sum(axis=1, keepdims=True)

    def categorical_cross_entropy(self, prediction, target_value):
        prediction_clipped = np.clip(prediction, 1e-8, None)
        return np.sum(-np.log(prediction_clipped) * target_value) / len(prediction)