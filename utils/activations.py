from functools import partial
import tensorflow.compat.v1 as tf
import numpy as np, cupy as cp, time
from utils.functions import Processing
from PIL import Image, ImageTk, ImageDraw
from tensorflow.experimental.dlpack import from_dlpack, to_dlpack

class Sigmoid:
    def __init__(self):
        pass
        
    @staticmethod
    def forward(x):
        return 1 / ( 1 + tf.exp(-x) )
    
    @staticmethod
    def backward(x):
        return x * (1 - x)
        
class Tanh:
    def __init__(self):
        pass
        
    @staticmethod
    def forward(x):
        return tf.tanh(x)

    @staticmethod
    def backward(x):
        return (1 - x ** 2)

class Relu:
    def __init__(self):
        pass

    @staticmethod
    def forward(x):
        return tf.where(x > 0, x, tf.zeros_like(x))

    @staticmethod
    def backward(x):
        return tf.where(x > 0, tf.ones_like(x), tf.zeros_like(x))

class Selu:
    def __init__(self):
        self.alpha = 1.67326
        self.scale = 1.0507

    def forward(self, x):
        return tf.where(x <= 0, self.scale * self.alpha * tf.exp(x), self.scale)

    def backward(self, x):
        return tf.where(x <= 0, self.scale * self.alpha * tf.exp(x), self.scale)

class Elu:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def forward(self, x):
        return tf.where(x > 0, x, self.alpha * (tf.exp(x) - 1))

    def backward(self, x):
        return tf.where(x > 0, tf.ones_like(x), self.alpha * tf.exp(x))

class Gelu:
    def __init__(self):
        pass

    @staticmethod
    def forward(x):
        return 0.5 * x * (1 + tf.tanh(tf.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))

    @staticmethod
    def backward(x):
        tanh_term = tf.tanh(tf.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))
        sech2_term = 1 - tanh_term ** 2
        return 0.5 * (1 + tanh_term + x * sech2_term * tf.sqrt(2 / np.pi) * (1 + 3 * 0.044715 * tf.pow(x, 2)))


class Silu:
    def __init__(self):
        pass

    @staticmethod
    def forward(x):
        return x * Sigmoid.forward(x)

    @staticmethod
    def backward(x):
        sigmoid = Sigmoid.forward(x)
        return sigmoid * (1 + x * (1 - sigmoid))

class Mish:
    def __init__(self):
        pass

    @staticmethod
    def forward(x):
        return x * tf.tanh(tf.math.softplus(x))

    @staticmethod
    def backward(x):
        # Numerically stable derivative of Mish
        sp = tf.math.softplus(x)
        sp_tanh = tf.tanh(sp)
        grad_sp = 1 - sp_tanh ** 2  # Derivative of tanh(softplus(x))

        return sp_tanh + x * grad_sp / (1 + tf.exp(-x))

class LRelu:
    def __init__(self, negative_slope=0.1):
        self.negative_slope = negative_slope
        
    def forward(self, x):
        return tf.where(x > 0, x, self.negative_slope * x)

    def backward(self, x):
        return tf.where(x > 0, tf.ones_like(x), tf.ones_like(x) * self.negative_slope)

class Softmax:
    def __init__(self):
        pass

    @staticmethod
    def forward(x):
        e_x = tf.exp(x)
        return e_x / tf.reduce_sum(e_x, axis=1, keepdims=True)

    @staticmethod
    def backward(x):
        e_x = tf.exp(x)
        return x * (1 - x)
        

class YoloActivation:
    def __init__(self):
        pass

    def forward(self, x):
        x_reshaped = cp.array(x).reshape((-1, 5))

        x_reshaped[..., [0, 1, 2]] = Sigmoid.forward(Processing.to_tensorflow(x_reshaped[..., [0, 1, 2]]))
        x_reshaped = x_reshaped.reshape(x.shape)

        return Processing.to_tensorflow(x_reshaped)

    def backward(self, x):
        x_reshaped = cp.array(x).reshape((-1, 5))
        x_deriv = cp.ones(x_reshaped.shape, dtype=x_reshaped.dtype)

        x_deriv[..., [0, 1, 2]] = Sigmoid.backward(Processing.to_tensorflow(x_reshaped[..., [0, 1, 2]]))
        x_deriv = x_deriv.reshape(x.shape)

        return Processing.to_tensorflow(x_deriv)