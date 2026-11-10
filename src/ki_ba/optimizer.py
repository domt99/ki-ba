from abc import ABC, abstractmethod
import numpy as np

class Optimizer(ABC):
    @abstractmethod
    def optimize(self, gradients, weights_old):
        pass


class Rprop(Optimizer):
    def __init__(self, n_pos = 1.2, n_neg = 0.5, delta_null = 0.5, delta_min = 0, delta_max = 50):
        self.n_pos = n_pos
        self.n_neg = n_neg
        self.delta_null = delta_null
        self.delta_min = delta_min
        self.delta_max = delta_max
        self.delta = None
        self.gradients_old = None

    def optimize(self, gradients, weights_old):

        if self.delta is None and self.gradients_old is None:

            delta = []
            gradients_old = []

            for G in gradients:
                delta.append(np.full_like(G, self.delta_null))
                gradients_old.append(np.zeros_like(G))

            self.delta = delta
            self.gradients_old = gradients_old

        weights_new = []

        for i in range(0, len(gradients)):

            gradient_product = self.gradients_old[i] * gradients[i]

            self.delta[i] = np.where( gradient_product > 0,
                np.minimum(self.delta[i] * self.n_pos, self.delta_max),
                self.delta[i]
            )
            self.delta[i] = np.where( gradient_product < 0,
                np.maximum(self.delta[i] * self.n_neg, self.delta_min),
                self.delta[i]
            )

            weights_new.append(weights_old[i] - np.sign(gradients[i]) * self.delta[i])
        
        self.gradients_old = gradients

        return weights_new

class GradientDescent(Optimizer):
    def __init__(self, learning_rate = 0.1):
        self.learning_rate = learning_rate

    def optimize(self, gradients, weights_old):

        weights_new = []

        for i in range(0, len(gradients)):
            weights_new.append(weights_old[i] - self.learning_rate * gradients[i])

        return weights_new

class Adam(Optimizer):
    def __init__(self, alpha=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
        self.alpha = alpha
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def optimize(self, gradients, weights_old):

        weights_new = []
        m_new = []
        v_new = []

        self.t = self.t + 1

        if self.m is None and self.v is None:

            m_init = []
            v_init = []

            for W in weights_old:
                m_init.append(np.zeros_like(W))
                v_init.append(np.zeros_like(W))

            self.m = m_init
            self.v = v_init

        for i in range(0, len(self.m)):

            m_curr = self.beta_1 * self.m[i] + (1 - self.beta_1) * gradients[i]
            v_curr = self.beta_2 * self.v[i] + (1 - self.beta_2) * gradients[i]**2
            
            m_new.append(m_curr)
            v_new.append(v_curr)

            m_corrected = m_curr / (1 - self.beta_1 ** self.t)
            v_corrected = v_curr / (1 - self.beta_2 ** self.t)

            weights_new.append(weights_old[i] - self.alpha * m_corrected / (np.sqrt(v_corrected) + self.epsilon))
            
        self.m = m_new
        self.v = v_new

        return weights_new
