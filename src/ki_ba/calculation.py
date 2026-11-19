from abc import ABC, abstractmethod
import numpy as np

class Calculation(ABC):
    @abstractmethod
    def calculate_net(self, extended_input, current_weights):
        pass

    @abstractmethod
    def calculate_gradient(self, extended_input, delta):
        pass

    @abstractmethod
    def calculate_backprop_error(self, delta, weights):
        pass

class CalculateLoop(Calculation):
    def calculate_net(self, extended_input, current_weights):
        net = np.zeros((len(extended_input), len(current_weights[0])))
        for i in range(0, len(extended_input)):
            for k in range(0, len(current_weights[0])):
                total = 0
                for j in range(0, len(extended_input[i])):
                    total = total + extended_input[i, j] * current_weights[j, k]
                net[i, k] = total
        return net

    def calculate_gradient(self, extended_input, delta):
        batch_size = len(extended_input)
        extended_input_transposed = extended_input.T
        gradient = np.zeros((len(extended_input_transposed), len(delta[0])))

        for i in range(0, len(extended_input_transposed)):
            for k in range(0, len(delta[0])):
                total = 0
                for j in range(0, len(extended_input_transposed[i])):
                    total = total + extended_input_transposed[i, j] * delta[j, k]
                gradient[i, k] = total / batch_size
        return gradient

    def calculate_backprop_error(self, delta, weights):
        weights_transposed = weights[1:].T
        backprop_error = np.zeros((len(delta), len(weights_transposed[0])))

        for i in range(0, len(delta)):
            for k in range(0, len(weights_transposed[0])):
                total = 0
                for j in range(0, len(delta[i])):
                    total = total + delta[i, j] * weights_transposed[j, k]
                backprop_error[i, k] = total
        return backprop_error
    
class CalculateMatrix(Calculation):
    def calculate_net(self, extended_input, current_weights):
        return extended_input @ current_weights

    def calculate_gradient(self, extended_input, delta):
        return (extended_input.T @ delta) / len(extended_input)

    def calculate_backprop_error(self, delta, weights):
        return delta @ weights[1:].T