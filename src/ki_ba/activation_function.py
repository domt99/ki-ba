from abc import ABC, abstractmethod
import numpy as np

class ActivationFunction(ABC):
    @abstractmethod
    def activation(self, net):
        pass

    @abstractmethod
    def activation_derivative(self, net):
        pass

class ReLU(ActivationFunction):
    def activation(self, net):
        return np.where(net > 0, net, 0)

    def activation_derivative(self, net):
        return np.where(net > 0, 1, 0)

class SigmoidLog(ActivationFunction):
    def activation(self, net):
        #net = 1 / (1 + np.e**-net)
        net = 0.5 * (1 + np.tanh(0.5 * net))
        return net

    def activation_derivative(self, net):
        net_activation = self.activation(net)
        net = net_activation * (1 - net_activation)
        return net

class SigmoidTanh(ActivationFunction):
    def activation(self, net):
        net = np.tanh(net)
        return net

    def activation_derivative(self, net):
        net_activation = self.activation(net)
        net = 1 - (net_activation**2)
        return net

class Identity(ActivationFunction):
    def activation(self, net):
        return net

    def activation_derivative(self, net):
        return np.ones_like(net)
