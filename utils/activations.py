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
        return 1 / (1 + tf.exp(-x))
    
    @staticmethod
    def backward(x):
        y = Sigmoid.forward(x)
        return y * (1 - y)
        
class Tanh:
    def __init__(self):
        pass
        
    @staticmethod
    def forward(x):
        return tf.tanh(x)

    @staticmethod
    def backward(x):
        return 1 - tf.tanh(x)**2

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
        return tf.where(x > 0, self.scale * x, self.scale * self.alpha * (tf.exp(x) - 1))

    def backward(self, x):
        return tf.where(x > 0, self.scale, self.scale * self.alpha * tf.exp(x))

class Elu:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    
    def forward(self, x):
        return tf.where(x > 0, x, self.alpha * (tf.exp(x) - 1))

    def backward(self, x):
        y = self.forward(x)
        return tf.where(x > 0, tf.ones_like(x), y + self.alpha)

class Gelu:
    def __init__(self):
        pass

    @staticmethod
    def forward(x):
        return 0.5 * x * (1 + tf.tanh(tf.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))

    @staticmethod
    def backward(x):
        gelu_output = Gelu.forward(x)
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
        y = Silu.forward(x)
        return y + Sigmoid.forward(x) * (1 - y)

class Mish:
    def __init__(self):
        pass

    @staticmethod
    def forward(x):
        return x * tf.tanh(tf.math.softplus(x))

    @staticmethod
    def backward(x):
        sp = tf.math.softplus(x)
        sp_tanh = tf.tanh(sp)
        grad_sp = 1 - sp_tanh ** 2
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
        e_x = tf.exp(x - tf.reduce_max(x, axis=1, keepdims=True))
        return e_x / tf.reduce_sum(e_x, axis=1, keepdims=True)

    @staticmethod
    def backward(x):
        y = Softmax.forward(x)
        return y * (1 - y)

class YoloActivation:
    def __init__(self, classes, dtype=np.float32):
        self.classes = tf.constant(classes, dtype=dtype)
        self.dtype = dtype

    def forward(self, x):
        x_reshaped = tf.reshape(x, (-1, 5+self.classes))
        return tf.reshape(tf.concat((
            Sigmoid.forward(x_reshaped[..., :3]),
            Sigmoid.forward(x_reshaped[..., 3:5]),
            Softmax.forward(x_reshaped[..., 5:])
        ), axis=-1), x.shape)

    def backward(self, x):
        x_reshaped = tf.reshape(x, (-1, 5+self.classes))
        return tf.reshape(tf.concat((
            Sigmoid.backward(x_reshaped[..., :3]),
            Sigmoid.backward(x_reshaped[..., 3:5]),
            Softmax.backward(x_reshaped[..., 5:])
        ), axis=-1), x.shape)