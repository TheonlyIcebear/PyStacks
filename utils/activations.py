from functools import partial
import tensorflow.compat.v1 as tf
import numpy as np, cupy as cp, time
from utils.layers import Layer
from utils.functions import Processing
from PIL import Image, ImageTk, ImageDraw
from tensorflow.experimental.dlpack import from_dlpack, to_dlpack

class Sigmoid(Layer):
    def forward(self, x, training=False):
        y = 1 / (1 + tf.exp(-x))
        if training:
            self.y = y
        return y
    
    def backward(self, output_gradient):
        y = self.y
        del self.y
        return output_gradient * y * (1 - y), []

class Tanh(Layer):
    def forward(self, x, training=False):
        y = tf.tanh(x)
        if training:
            self.y = y
        return y

    def backward(self, output_gradient):
        y = self.y
        del self.y
        return output_gradient * (1 - tf.square(y)), []

class Relu(Layer):
    def forward(self, x, training=False):
        return tf.maximum(0., x)

    def backward(self, output_gradient):
        return tf.cast(output_gradient != 0, output_gradient.dtype) * output_gradient, []

class Selu(Layer):
    def __init__(self):
        self.alpha = 1.67326
        self.scale = 1.0507

    def forward(self, x, training=False):
        y = self.scale * tf.where(x > 0, x, self.alpha * (tf.exp(x) - 1))
        if training:
            self.y = y
        return y

    def backward(self, output_gradient):
        y = self.y
        del self.y
        return output_gradient * tf.where(y > 0, self.scale, self.scale * self.alpha), []

class Elu(Layer):
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def forward(self, x, training=False):
        y = tf.where(x > 0, x, self.alpha * (tf.exp(x) - 1))
        if training:
            self.y = y
        return y

    def backward(self, output_gradient):
        y = self.y
        del self.y
        return output_gradient * tf.where(y > 0, 1., self.alpha + y/self.alpha), []

class Gelu(Layer):
    def forward(self, x, training=False):
        y = 0.5 * x * (1 + tf.tanh(tf.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))
        if training:
            self.x = x
        return y

    def backward(self, output_gradient):
        x = self.x
        del self.x
        term = tf.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))
        tanh_term = tf.tanh(term)
        return output_gradient * 0.5 * (1 + tanh_term + x * (1 - tf.square(tanh_term)) * tf.sqrt(2 / np.pi) * (1 + 3 * 0.044715 * tf.square(x))), []

class Silu(Layer):
    def forward(self, x, training=False):
        y = x * tf.sigmoid(x)
        if training:
            self.x = x
        return y

    def backward(self, output_gradient):
        x = self.x
        del self.x
        sigmoid = tf.sigmoid(x)
        return output_gradient * sigmoid * (1 + x * (1 - sigmoid)), []

class Mish(Layer):
    def forward(self, x, training=False):
        y = x * tf.tanh(tf.math.softplus(x))
        if training:
            self.x = x
        return y

    def backward(self, output_gradient):
        x = self.x
        del self.x
        sp = tf.math.softplus(x)
        sp_tanh = tf.tanh(sp)
        return output_gradient * (sp_tanh + x * (1 - tf.square(sp_tanh)) / (1 + tf.exp(-x))), []

class LRelu(Layer):
    def __init__(self, negative_slope=0.1):
        self.negative_slope = negative_slope
        
    def forward(self, x, training=False):
        return tf.where(x > 0, x, self.negative_slope * x)

    def backward(self, output_gradient):
        return tf.where(output_gradient > 0, output_gradient, self.negative_slope * output_gradient), []

class Softmax(Layer):
    def forward(self, x, training=False):
        e_x = tf.exp(x - tf.reduce_max(x, axis=-1, keepdims=True))
        y = e_x / tf.reduce_sum(e_x, axis=-1, keepdims=True)
        if training:
            self.y = y
        return y

    def backward(self, output_gradient):
        y = self.y
        del self.y
        return output_gradient * y * (1 - y), []

class YoloActivation(Layer):
    def forward(self, x, training=False):
        x_reshaped = tf.reshape(x, (-1, 5))
        y = tf.reshape(tf.concat((
            tf.sigmoid(x_reshaped[..., :3]),
            x_reshaped[..., 3:]
        ), axis=-1), x.shape)
        
        if training:
            self.y = y
        return y

    def backward(self, output_gradient):
        y = self.y
        del self.y
        y_reshaped = tf.reshape(y, (-1, 5))
        output_gradient_reshaped = tf.reshape(output_gradient, (-1, 5))
        
        sigmoid_input_gradient = output_gradient_reshaped[..., :3] * y_reshaped[..., :3] * (1 - y_reshaped[..., :3])
        identity_input_gradient = output_gradient_reshaped[..., 3:]
        
        return tf.reshape(tf.concat((sigmoid_input_gradient, identity_input_gradient), axis=-1), output_gradient.shape), []