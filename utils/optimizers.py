import tensorflow.compat.v1 as cp, numpy as np, cupy as cp


epsilon = 10e-5


class Adam:
    def __init__(self, momentum = 0.9, beta_constant = 0.99):
        self.momentum_constant = momentum
        self.adam_constant = beta_constant

    def apply_gradient(self, values, gradient, descent_values, learning_rate):
        if not (descent_values is None):
            momentum, squared_momentum = descent_values

        else:
            squared_momentum = 0
            momentum = 0

        new_gradient_momentum = (self.momentum_constant * momentum) + (1 - self.momentum_constant) * gradient
        new_squared_momentum = (self.adam_constant * squared_momentum) + (1 - self.adam_constant) * (gradient ** 2)

        new_values = values - learning_rate * (new_gradient_momentum / (cp.sqrt(new_squared_momentum + epsilon)))
        new_descent_values = [new_gradient_momentum, new_squared_momentum]

        return new_values, new_descent_values
                
class RMSProp:
    def __init__(self, beta_constant = 0.99):
        self.adam_constant = beta_constant

    def apply_gradient(self, values, gradient, descent_values, learning_rate):
        if not (descent_values is None):
            squared_momentum = descent_values

        else:
            squared_momentum = 0

        new_squared_momentum = (self.beta_constant * squared_momentum) + (1 - self.beta_constant) * (gradient ** 2)
        new_squared_momentum = new_squared_momentum / (1 - self.beta_constant ** generation)

        new_values = values - learning_rate * (gradient / (cp.sqrt(new_squared_momentum + epsilon)))
        new_descent_values = new_squared_momentum

        return new_values, new_descent_values

class Momentum:
    def __init__(self, momentum = 0.9):
        self.momentum_constant = momentum

    def apply_gradient(self, values, gradient, descent_values, learning_rate):

        if not (descent_values is None):
            momentum = descent_values

        else:
            momentum = 0

        change = ( gradient ) + ( momentum * self.momentum_constant )

        new_values = values - change
        new_descent_values = change

        return new_values, new_descent_values

class SGD:
    def __init__(self):
        pass

    def apply_gradient(self, values, gradient, descent_values, learning_rate):
        gradient = gradient.copy()
        gradient *= learning_rate
        new_values = values - gradient

        return new_values, None