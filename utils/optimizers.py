import tensorflow.compat.v1 as tf, numpy as np, cupy as cp


class Adam:
    def __init__(self, momentum = 0.9, beta_constant = 0.99, weight_decay = 0, epsilon = 1e-4):
        self.momentum_constant = momentum
        self.beta_constant = beta_constant
        self.weight_decay = weight_decay
        self.cache_shape = [0, 0]
        self.epsilon = epsilon

    def apply_gradient(self, values, gradient, descent_values, learning_rate, iteration):
        momentum = descent_values[0]
        squared_momentum = descent_values[1]

        if momentum is None:
            momentum = tf.zeros_like(values)

        if squared_momentum is None:
            squared_momentum = tf.zeros_like(values)

        new_gradient_momentum = (self.momentum_constant * momentum) + (1 - self.momentum_constant) * gradient
        new_squared_momentum = (self.beta_constant * squared_momentum) + (1 - self.beta_constant) * (gradient ** 2)

        _new_gradient_momentum = new_gradient_momentum / (1 - self.momentum_constant ** (iteration + 1))
        _new_squared_momentum = new_squared_momentum / (1 - self.beta_constant ** (iteration + 1))

        new_descent_values = [new_gradient_momentum, new_squared_momentum]

        update = learning_rate * (_new_gradient_momentum / tf.sqrt(_new_squared_momentum + self.epsilon))
        update += values * self.weight_decay * learning_rate

        values.assign_sub(update)

        return new_descent_values
                
class RMSProp:
    def __init__(self, beta_constant = 0.9, weight_decay = 0, epsilon = 1e-4):
        self.beta_constant = beta_constant
        self.weight_decay = weight_decay
        self.cache_shape = tf.constant(0)
        self.epsilon = epsilon

    def apply_gradient(self, values, gradient, descent_values, learning_rate, iteration):
        if descent_values is None:
            squared_momentum = tf.zeros_like(values)

        else:
            squared_momentum = descent_values

        new_squared_momentum = (self.beta_constant * squared_momentum) + (1 - self.beta_constant) * (gradient ** 2)
        new_descent_values = new_squared_momentum

        update = learning_rate * (gradient / tf.sqrt(new_squared_momentum + self.epsilon))
        update += values * self.weight_decay * learning_rate

        values.assign_sub(update)

        return new_descent_values

class Momentum:
    def __init__(self, momentum = 0.9, weight_decay = 0):
        self.momentum_constant = momentum
        self.weight_decay = weight_decay
        self.cache_shape = tf.constant(0)

    def apply_gradient(self, values, gradient, descent_values, learning_rate, iteration):

        if descent_values is None:
            momentum = tf.zeros_like(values)
        else:
            momentum = descent_values

        change = ( gradient ) + ( momentum * self.momentum_constant )

        new_descent_values = change

        update = change * learning_rate
        update += values * self.weight_decay * learning_rate

        values.assign_sub(update)

        return new_descent_values

class SGD:
    def __init__(self, weight_decay = 0):
        self.weight_decay = weight_decay
        self.cache_shape = tf.constant(0)

    def apply_gradient(self, values, gradient, descent_values, learning_rate, iteration):
        update = values - (gradient * learning_rate)
        update += values * self.weight_decay * learning_rate

        values.assign_sub(update)

        return None