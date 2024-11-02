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
        return 1 * (x > 0)

    @staticmethod
    def backward(x):
        return x * (x > 0)

class Selu:
    def __init__(self):
        self.alpha = 1.67326
        self.scale = 1.0507

    def forward(self, x):
        return tf.where(x <= 0, self.scale * self.alpha * tf.exp(x), self.scale)

    def backward(self, x):
        return tf.where(x <= 0, self.scale * self.alpha * (tf.exp(x) - 1), self.scale * x)

class Silu:
    def __init__(self):
        pass

    @staticmethod
    def forward(x):
        sigmoid = Sigmoid.forward(x)
        return x * sigmoid

    @staticmethod
    def backward(x):
        sigmoid = Sigmoid.forward(x)
        return sigmoid * (1 + x * (1 - sigmoid))

class LRelu:
    def __init__(self, negative_slope=0.1):
        self.negative_slope = negative_slope
        
    def forward(self, x):
        dtype = x.dtype
        x = cp.array(x)
        return tf.cast(Processing.to_tensorflow(cp.where(x > 0, x, self.negative_slope * x)), dtype)

    def backward(self, x):
        dtype = x.dtype
        x = cp.array(x)
        return tf.cast(Processing.to_tensorflow(cp.where(x > 0, 1, self.negative_slope)), dtype)

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
    def __init__(self, grid_size=13, anchors=4):
        self.grid_size = grid_size
        self.anchors = anchors

    def forward(self, x):
        x_reshaped = cp.array(x).reshape((-1, self.grid_size ** 2, self.anchors, 5))

        x_reshaped[..., [0, 1, 2]] = Sigmoid.forward(Processing.to_tensorflow(x_reshaped[..., [0, 1, 2]]))
        x_reshaped[..., [3, 4]] = cp.exp(x_reshaped[..., [3, 4]])
        x_reshaped = x_reshaped.reshape(x.shape)

        return Processing.to_tensorflow(x_reshaped)

    def backward(self, x):
        x_reshaped = cp.array(x).reshape((-1, self.grid_size ** 2, self.anchors, 5))
        x_deriv = cp.ones(x_reshaped.shape, dtype=x_reshaped.dtype)

        x_deriv[..., [0, 1, 2]] = Sigmoid.backward(Processing.to_tensorflow(x_reshaped[..., [0, 1, 2]]))
        x_deriv[..., [3, 4]] = cp.exp(x_reshaped[..., [3, 4]])
        x_deriv = x_deriv.reshape(x.shape)

        return Processing.to_tensorflow(x_deriv)