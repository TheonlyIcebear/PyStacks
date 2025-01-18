import tensorflow.compat.v1 as tf, skimage, scipy, numpy as np, cupy as cp, time, os
from tensorflow.experimental.dlpack import from_dlpack, to_dlpack
from utils.functions import Processing
from utils.initializers import *
from utils.loss import *

class Layer:
    def __init__(self):
        pass

    @property
    def size(self):
        return 0

    def _size_of(self, variable):
        return tf.size(variable) * variable.dtype.size

    def initialize(self, input_shape, initialize_weights=True, dtype=np.float64):
        self.output_shape = input_shape
        self.dtype = dtype

    def forward(self, input_activations, training=True):
        return input_activations

    def backward(self, output_gradient):
        return output_gradient, []

    def update(self, optimizer, gradient, descent_values, learning_rate, iteration):
        pass

    def save(self):
        return [], None

    def load(self, data, dtype):
        pass

class Input(Layer):
    def __init__(self, input_shape):
        self.output_shape = np.array(input_shape)

    def forward(self, input_activations, training=True):
        output_activations = tf.constant(input_activations, dtype=self.dtype)

        return output_activations

    def initialize(self, input_shape, initialize_weights=True, dtype=np.float64):
        self.dtype = dtype
        self.output_shape = input_shape

    def save(self):
        return [self.output_shape], None

    def load(self, data, dtype):
        self.dtype = dtype

class Flatten(Layer):
    def __init__(self):
        pass

    def forward(self, input_activations, training=True):
        output_activations = tf.reshape(input_activations, (input_activations.shape[0], -1))

        self.input_shape = input_activations.shape
        return output_activations

    def backward(self, output_gradient):
        return tf.reshape(output_gradient, self.input_shape), []

    def initialize(self, input_shape, initialize_weights=True, dtype=np.float64):
        self.output_shape = input_shape.prod()

class Reshape(Layer):
    def __init__(self, output_shape):
        self.output_shape = output_shape

    def forward(self, input_activations, training=True):
        output_activations = tf.reshape(input_activations, output_shape)

        self.input_shape = input_activations.shape
        return output_activations

    def backward(self, output_gradient):
        return tf.reshape(output_gradient, self.input_shape), []

    def save(self):
        return [self.output_shape], None

class Conv2d(Layer):
    def __init__(self, depth, kernel_shape=[3, 3], stride=[1, 1], weight_initializer=HeNormal(), bias_initializer=Fill(0), padding = "VALID"):
        self.kernel_shape = np.array(kernel_shape)[::-1]
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer
        self.depth = depth

        if isinstance(stride, int):
            stride = [stride, stride]

        self.stride = stride
        self.padding = padding

    @property
    def size(self):
        if not hasattr(self, 'kernels') or not hasattr(self, 'biases'):
            raise AttributeError("Layer needs to be initialized first.")
        
        return (self._size_of(self.kernels) + self._size_of(self.biases))

    def forward(self, input_activations, training=True):
        # for i, kernels in enumerate(self.kernels):
        #     for kernel, channel in zip(kernels, input_activations):
        #         output_activations[i] += scipy.signal.correlate2d(channel, kernel, "valid")

        output_activations = tf.nn.conv2d(
            input_activations, 
            self.kernels, 
            strides=[1, *self.stride[::-1], 1], 
            padding=self.padding
        )

        output_activations += self.biases

        if training:
            self.input_activations = input_activations

        return output_activations

    def backward(self, output_gradient):
        # input_gradient = cp.zeros(self.input_activations.shape)
        # kernels_gradient = cp.zeros(self.kernels.shape)

        # for i, (kernels, kernel_output_gradient) in enumerate(zip(self.kernels, output_gradient)):
        #     for j, (image, kernel) in enumerate(zip(input_activations, kernels)):
                
        #         kernels_gradient[i, j] = scipy.signal.correlate2d(image, kernel_output_gradient, "valid")
        #         input_gradient[j] += scipy.signal.convolve2d(kernel_output_gradient, kernel, "full")

        input_gradient = tf.nn.conv2d_backprop_input(
            input_sizes = self.input_activations.shape,
            filters = self.kernels,
            out_backprop = output_gradient,
            strides = [1, *self.stride[::-1], 1],
            padding = self.padding,
        )

        kernels_gradient = tf.nn.conv2d_backprop_filter(
            input = self.input_activations,
            filter_sizes = self.kernels.shape,
            out_backprop = output_gradient,
            strides = [1, *self.stride[::-1], 1],
            padding = self.padding,
        )

        # kernels_gradient = tf.cast(kernels_gradient, output_gradient.dtype)
        # input_gradient = tf.cast(input_gradient, output_gradient.dtype)

        biases_gradient = tf.reduce_mean(output_gradient, axis=0)
        del self.input_activations

        return input_gradient, [
            kernels_gradient, 
            biases_gradient
        ]

    def initialize(self, input_shape, initialize_weights=True, dtype=np.float64):
        input_channels = int(input_shape[-1])
        output_channels = self.depth

        kernel_height, kernel_width = self.kernel_shape
        kernel_width = int(kernel_width)
        kernel_height = int(kernel_height)

        fan_in = input_channels * kernel_width * kernel_height
        fan_out = output_channels * kernel_width * kernel_height

        output_shape = input_shape
        output_shape[-1] = self.depth

        if self.padding.lower() == "valid":
            output_shape[:-1] = (input_shape[:-1] - self.kernel_shape + 1)
        
        output_shape[:-1] = np.ceil(output_shape[:-1] / np.array(self.stride)).astype(int)
        self.output_shape = output_shape

        if initialize_weights:
            self.kernels = tf.cast(self.weight_initializer(fan_in, fan_out, (kernel_height, kernel_width, input_channels, output_channels)), dtype=dtype)
            self.biases = tf.cast(self.bias_initializer(fan_in, fan_out, output_shape), dtype=dtype)

    def update(self, optimizer, gradient, descent_values, learning_rate, iteration):
        if not descent_values is None:
            kernel_descent_values, bias_descent_values = descent_values

        else:
            kernel_descent_values = None
            bias_descent_values = None

        kernels_gradient, biases_gradient = gradient

        self.kernels, new_kernel_descent_values = optimizer.apply_gradient(self.kernels, kernels_gradient, kernel_descent_values, learning_rate, iteration)
        self.biases, new_bias_descent_values = optimizer.apply_gradient(self.biases, biases_gradient, bias_descent_values, learning_rate, iteration)

        return [new_kernel_descent_values, new_bias_descent_values]

    def save(self):
        return [self.depth, self.kernel_shape, self.stride, self.weight_initializer, self.bias_initializer, self.padding], [self.kernels.numpy(), self.biases.numpy()]

    def load(self, data, dtype):
        self.kernels, self.biases = data
        self.kernels = tf.constant(self.kernels, dtype=dtype)
        self.biases = tf.constant(self.biases, dtype=dtype)

class Dense(Layer):
    def __init__(self, depth, weight_initializer=HeNormal(), bias_initializer=Fill(0)):
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer
        self.depth = np.array(depth)

    @property
    def size(self):
        if not hasattr(self, 'layer'):
            raise AttributeError("Layer needs to be initialized first.")

        return self._size_of(self.layer)

    def forward(self, input_activations, training=True):
        start_time = time.perf_counter()
        weights = self.layer[..., :-1]
        biases = self.layer[..., -1]

        output_activations = tf.tensordot(input_activations, weights, axes=[[1], [1]]) + biases

        end_time = time.perf_counter()

        if training:
            self.input_activations = input_activations

        return output_activations

    def backward(self, output_gradient):
        weights = self.layer[..., :-1]
        biases = self.layer[..., -1]

        batch_size = self.input_activations.shape[0]

        input_gradient = tf.tensordot(output_gradient, tf.transpose(weights), axes=[[1], [1]])

        weights_derivative = output_gradient[..., None] * tf.reshape(self.input_activations, (batch_size, 1, -1))
        bias_derivative = output_gradient

        gradient = tf.concat((weights_derivative, bias_derivative[..., None]), axis=-1)
        
        gradient = tf.reduce_mean(gradient, axis=0)
        del self.input_activations

        return input_gradient, gradient

    def initialize(self, input_shape, initialize_weights=True, dtype=np.float64):
        inputs = int(input_shape)

        if initialize_weights:
            weights = tf.cast(self.weight_initializer(inputs, self.depth, (self.depth, inputs)), dtype=dtype)
            biases = tf.cast(self.bias_initializer(inputs, self.depth, (self.depth,)), dtype=dtype)

            self.layer = tf.concat((weights, biases[:, None]), axis=-1)

        self.output_shape = self.depth

    def update(self, optimizer, gradient, descent_values, learning_rate, iteration):
        self.layer, new_descent_values = optimizer.apply_gradient(self.layer, gradient, descent_values, learning_rate, iteration)

        return new_descent_values

    def save(self):
        return [self.depth, self.weight_initializer, self.bias_initializer], self.layer.numpy()

    def load(self, data, dtype):
        self.layer = tf.constant(data, dtype=dtype)

class ConcatBlock(Layer):
    def __init__(self, layers, residual_layers, axis=-1):
        self.layers = layers
        self.residual_layers = residual_layers
        self.axis = axis

    @property
    def size(self):
        if not hasattr(self, 'layers') or not hasattr(self, 'residual_layers'):
            raise AttributeError("Layer needs to be initialized first.")
        
        return sum([layer.size for layer in self.layers]) + sum([layer.size for layer in self.residual_layers])

    def forward(self, input_activations, training=True):
        residual = input_activations
        activations = input_activations

        for layer in self.layers:
            activations = layer.forward(activations, training=training)
    
        for layer in self.residual_layers:
            residual = layer.forward(residual, training=training)

        output_activations = tf.concat((activations, residual), axis=self.axis)

        if training:
            self.main_activations_depth = activations.shape[axis]

        return output_activations

    def backward(self, output_gradient):
        gradients = [None] * len(self.layers)

        slices = [slice(None)] * output_gradient.ndim
        slices[self.axis] = slice(None, self.main_activations_depth)
        slices = tuple(slices)

        _output_gradient = output_gradient[slices]

        for idx, layer in enumerate(self.layers[::-1]):
            current_layer_index = -(idx + 1)

            _output_gradient, gradient = layer.backward(_output_gradient)
            gradients[current_layer_index] = gradient

            del gradient

        residual_gradients = [None] * len(self.residual_layers)

        slices = [slice(None)] * output_gradient.ndim
        slices[self.axis] = slice(self.main_activations_depth, None)
        slices = tuple(slices)

        residual_output_gradient = output_gradient[..., self.main_activations_depth:]

        for idx, layer in enumerate(self.residual_layers[::-1]):
            current_layer_index = -(idx + 1)

            residual_output_gradient, gradient = layer.backward(residual_output_gradient)
            residual_gradients[current_layer_index] = gradient

        return _output_gradient + residual_output_gradient, [gradients, residual_gradients]

    def initialize(self, input_shape, initialize_weights=True, dtype=np.float64):
        output_shape = input_shape.copy()
        
        for layer in self.layers:
            layer.initialize(output_shape, initialize_weights=initialize_weights, dtype=dtype)
            output_shape = layer.output_shape
            print(layer, output_shape, "MAIN")

        residual_output_shape = input_shape.copy()

        for layer in self.residual_layers:
            layer.initialize(residual_output_shape, initialize_weights=initialize_weights, dtype=dtype)
            residual_output_shape = layer.output_shape
            print(layer, residual_output_shape, "RES")

        self.output_shape = output_shape
        self.output_shape[self.axis] += residual_output_shape[self.axis]

    def update(self, optimizer, gradient, descent_values, learning_rate, iteration):
        gradients, residual_gradients = gradient

        if descent_values:
            descent_values, residual_descent_values = descent_values

        else:
            descent_values = [None] * len(self.layers)
            residual_descent_values = [None] * len(self.residual_layers)

        new_descent_values = []

        for layer, layer_gradient, descent_value in zip(self.layers, gradients, descent_values):
            new_descent_value = layer.update(optimizer, layer_gradient, descent_value, learning_rate, iteration)
            new_descent_values.append(new_descent_value)

        new_residual_descent_values = []

        for layer, layer_gradient, descent_value in zip(self.residual_layers, residual_gradients, residual_descent_values):
            new_residual_descent_value = layer.update(optimizer, layer_gradient, descent_value, learning_rate, iteration)
            new_residual_descent_values.append(new_residual_descent_value)

        return [new_descent_values, new_residual_descent_values]

    def save(self):
        layers_data = []
        residual_layers_data = []

        for layer in self.layers:
            layers_data.append(list(layer.save()) + [layer.__class__])

        for layer in self.residual_layers:
            residual_layers_data.append(list(layer.save()) + [layer.__class__])

        return [None, None], [layers_data, residual_layers_data]

    def load(self, data, dtype):
        layers_data, residual_layers_data = data
        layers = []
        residual_layers = []

        for layer_args, layer_data, layer_class in layers_data:
            layer = layer_class(*layer_args)
            layer.load(layer_data, dtype=dtype)
            layers.append(layer)

        for layer_args, layer_data, layer_class in residual_layers_data:
            layer = layer_class(*layer_args)
            layer.load(layer_data, dtype=dtype)
            residual_layers.append(layer)

        self.layers = layers
        self.residual_layers = residual_layers

class ResidualBlock(Layer):
    def __init__(self, layers):
        self.layers = layers

    @property
    def size(self):
        if not hasattr(self, 'layers'):
            raise AttributeError("Layer needs to be initialized first.")
        
        return sum([layer.size for layer in self.layers])

    def forward(self, input_activations, training=True):
        activations = input_activations

        for layer in self.layers:
            activations = layer.forward(activations, training=training)

        output_activations = activations + input_activations
            
        return output_activations

    def backward(self, output_gradient):
        gradients = [None] * len(self.layers)
        _output_gradient = output_gradient

        for idx, layer in enumerate(self.layers[::-1]):
            current_layer_index = -(idx + 1)
            
            _output_gradient, gradient = layer.backward(_output_gradient)
            gradients[current_layer_index] = gradient

            del gradient

        return _output_gradient + output_gradient, [gradients]

    def initialize(self, input_shape, initialize_weights=True, dtype=np.float64):
        output_shape = input_shape.copy()

        for layer in self.layers:
            layer.initialize(output_shape, initialize_weights=initialize_weights, dtype=dtype)
            output_shape = layer.output_shape
            print(layer, output_shape, "MAIN")

        self.output_shape = output_shape

    def update(self, optimizer, gradients, descent_values, learning_rate, iteration):
        new_descent_values = []
        if descent_values is None:
            descent_values = []

        for layer, layer_gradient, descent_value in zip(self.layers, gradients, descent_values):
            new_descent_value = layer.update(optimizer, layer_gradient, descent_value, learning_rate, iteration)
            new_descent_values.append(new_descent_value)

        return new_descent_values

    def save(self):
        layers_data = []

        for layer in self.layers:
            layers_data.append(list(layer.save()) + [layer.__class__])

        return [None], layers_data

    def load(self, data, dtype):
        layers = []

        if len(data) == 1:
            data = data[0]

        for layer_args, layer_data, layer_class in data:
            layer = layer_class(*layer_args)
            layer.load(layer_data, dtype=dtype)
            layers.append(layer)

        self.layers = layers

# The next 4 classes are independent from the ConcatBlock class
# Proper user order is as follows:
# ConcatStartPoint
# any layers...
# ConcatResStartPoint
# any layers...
# ConcatEndPoint

# Concat is intended for generating the other three layers, it should not be used as a layer itself
class Concat(Layer):
    def __init__(self, axis=-1):
        self.start_point = None
        self.start_res = None
        self.end_point = None
        self.axis = axis
    
    def generate_layers(self):
        start_point = ConcatStartPoint(self)
        start_res = ConcatResidualStartPoint(self)
        end_point = ConcatEndPoint(self)

        self.start_point, self.start_res, self.end_point = start_point, start_res, end_point
        return start_point, start_res, end_point

class ConcatStartPoint(Concat):
    def __init__(self, parent):
        self.parent = parent

    def forward(self, input_activations, training=True):
        self.parent.start_point.output_activations = input_activations # Store the inputs to be given to the res and main layers
        return self.parent.start_point.output_activations 

    def backward(self, output_gradient):
        output_gradient = output_gradient + self.parent.start_res.output_gradient # Add output_gradient coming from main layer to output_gradient from res layers
        del self.parent.start_res.output_gradient, self.parent.end_point.output_gradient
        return output_gradient, []

    def initialize(self, input_shape, initialize_weights=True, dtype=np.float64):
        self.parent.start_point.output_shape = input_shape

    def save(self):
        return [self.parent], None

class ConcatResidualStartPoint(Concat):
    def __init__(self, parent):
        self.parent = parent

    def forward(self, input_activations, training=True):
        self.parent.start_res.output_activations = input_activations # The outputs from the main layers
        return self.parent.start_point.output_activations # The same inputs that were given to the start of the main layers

    def backward(self, output_gradient):
        self.parent.start_res.output_gradient = output_gradient

        slices = [slice(None)] * self.parent.end_point.output_gradient.ndim
        slices[self.parent.axis] = slice(None, self.parent.end_point.main_activations_depth)
        slices = tuple(slices)

        return self.parent.end_point.output_gradient[slices], [] # Pass the output_gradient corresponding to the main layers

    def initialize(self, input_shape, initialize_weights=True, dtype=np.float64):
        self.parent.start_res.output_shape = self.parent.start_point.output_shape
        self.parent.start_res.input_shape = input_shape

    def save(self):
        return [self.parent], None

class ConcatEndPoint(Concat):
    def __init__(self, parent):
        self.parent = parent

    def forward(self, input_activations, training=True):
        activations = self.parent.start_res.output_activations
        residual = input_activations
        output_activations = tf.concat((activations, residual), axis=self.parent.axis)

        self.parent.end_point.main_activations_depth = self.parent.start_res.output_activations.shape[self.parent.axis]
        del self.parent.start_point.output_activations, self.parent.start_res.output_activations
        return output_activations

    def backward(self, output_gradient):
        self.parent.end_point.output_gradient = output_gradient # Save output_gradient for main layers

        slices = [slice(None)] * self.parent.end_point.output_gradient.ndim
        slices[self.parent.axis] = slice(self.parent.end_point.main_activations_depth, None)
        slices = tuple(slices)

        return self.parent.end_point.output_gradient[slices], [] # Pass the output_gradient corresponding to the residual layers

    def initialize(self, input_shape, initialize_weights=True, dtype=np.float64):
        self.parent.end_point.output_shape = input_shape
        self.parent.end_point.output_shape[self.parent.axis] += self.parent.start_res.input_shape[self.parent.axis]

    def save(self):
        return [self.parent], None

class BatchNorm(Layer):
    def __init__(self, momentum=0.9, epsilon=1e-5):
        self.momentum = momentum
        self.epsilon = epsilon

    @property
    def size(self):
        if not hasattr(self, 'gamma') or not hasattr(self, 'beta') or not hasattr(self, 'running_mean') or not hasattr(self, 'running_var'):
            raise AttributeError("Layer needs to be initialized first.")

        return self._size_of(self.gamma) + self._size_of(self.beta) + self._size_of(self.running_mean) + self._size_of(self.running_var)

    def forward(self, input_activations, training=True):
        axes = np.arange(input_activations.ndim - 1)

        if training:
            batch_mean = tf.reduce_mean(input_activations, axis=axes, keepdims=True)
            batch_var = tf.math.reduce_variance(input_activations, axis=axes, keepdims=True)

            x_centered = input_activations - batch_mean

            stddev_inv = 1 / tf.sqrt(batch_var + self.epsilon)
            x_norm = x_centered * stddev_inv

            self.batch_mean = batch_mean
            self.batch_var = batch_var
        else:
            x_norm = (input_activations - self.running_mean) / (tf.sqrt(self.running_var) + self.epsilon)
        
        output_activations = self.gamma * x_norm + self.beta

        if training:
            self.input_activations = input_activations
            self.batch_mean = batch_mean
            self.batch_var = batch_var

        return output_activations

    def backward(self, output_gradient):
        axes = np.arange(self.input_activations.ndim - 1)
        stddev_inv = 1 / tf.sqrt(self.batch_var + self.epsilon)
        x_centered = self.input_activations - self.batch_mean
        x_norm = x_centered * stddev_inv

        batch_size = tf.cast(tf.reduce_prod(tf.gather(tf.shape(self.input_activations), axes)), output_gradient.dtype)
        beta_gradient = tf.reduce_sum(output_gradient, axis=axes)
        gamma_gradient = tf.reduce_sum(output_gradient * x_norm, axis=axes)
        
        dx_norm = output_gradient * self.gamma
        dx_centered = dx_norm * stddev_inv
        
        dvar = tf.reduce_sum(dx_norm * x_centered * -0.5 * stddev_inv * stddev_inv * stddev_inv, axis=axes, keepdims=True)
        dmean = tf.reduce_sum(dx_centered * -1.0, axis=axes, keepdims=True) + dvar * tf.reduce_mean(-2.0 * x_centered, axis=axes, keepdims=True)
        
        input_gradient = dx_centered + dvar * 2.0 * x_centered / batch_size + dmean / batch_size
        
        del self.input_activations
        
        return input_gradient, [gamma_gradient, beta_gradient]

    def update(self, optimizer, gradient, descent_values, learning_rate, iteration):
        dgamma, dbeta = gradient

        if descent_values is not None:
            gamma_descent_values, beta_descent_values = descent_values
        else:
            gamma_descent_values = None
            beta_descent_values = None

        self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.batch_mean
        self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.batch_var

        self.gamma, new_gamma_descent_values = optimizer.apply_gradient(self.gamma, dgamma, gamma_descent_values, learning_rate, iteration)
        self.beta, new_beta_descent_values = optimizer.apply_gradient(self.beta, dbeta, beta_descent_values, learning_rate, iteration)

        return [new_gamma_descent_values, new_beta_descent_values]

    def initialize(self, input_shape, initialize_weights=True, dtype=np.float64):
        dims = len(input_shape)

        if dims > 1:
            features = np.array(input_shape)[-1]
        else:
            features = input_shape

        if initialize_weights:
            self.gamma = tf.ones((1,) * dims + (features,), dtype=dtype)
            self.beta = tf.zeros((1,) * dims + (features,), dtype=dtype)
            self.running_mean = tf.zeros((1,) * dims + (features,), dtype=dtype)
            self.running_var = tf.ones((1,) * dims + (features,), dtype=dtype)

        self.output_shape = input_shape

    def save(self):
        return [self.momentum], [self.gamma.numpy(), self.beta.numpy(), self.running_mean.numpy(), self.running_var.numpy()]

    def load(self, data, dtype):
        self.gamma, self.beta, self.running_mean, self.running_var = data
        self.gamma = tf.constant(self.gamma, dtype=dtype)
        self.beta = tf.constant(self.beta, dtype=dtype)
        self.running_mean = tf.constant(self.running_mean, dtype=dtype)
        self.running_var = tf.constant(self.running_var, dtype=dtype)

class Space2Depth(Layer):
    def __init__(self, block_size):
        self.block_size = block_size

    def forward(self, input_activations, training=True):
        output_activations = tf.nn.space_to_depth(
                input_activations, 
                self.block_size
            )

        return output_activations

    def backward(self, output_gradient):
        return tf.nn.depth_to_space(
                output_gradient, 
                self.block_size
            ), []

    def initialize(self, input_shape, initialize_weights=True, dtype=np.float64):
        self.output_shape = input_shape[:]
        self.output_shape[-1] *= self.block_size ** 2
        self.output_shape[:-1] //= self.block_size

    def save(self):
        return [self.block_size], None

class Depth2Space(Layer):
    def __init__(self, block_size):
        self.block_size = block_size

    def forward(self, input_activations, training=True):
        output_activations = tf.nn.depth_to_space(
                input_activations, 
                self.block_size
            )

        return output_activations

    def backward(self, output_gradient):
        return tf.nn.space_to_depth(
                output_gradient, 
                self.block_size
            ), []

    def initialize(self, input_shape, initialize_weights=True, dtype=np.float64):
        self.output_shape = input_shape[:]
        self.output_shape[-1] //= self.block_size ** 2
        self.output_shape[:-1] *= self.block_size

    def save(self):
        return [self.block_size], None

class Upsample(Layer):
    def __init__(self, scale_factor, method="nearest"):
        self.scale_factor = scale_factor
        self.method = method

    def forward(self, input_activations, training=True):
        input_shape = input_activations.shape

        new_height = input_shape[1] * self.scale_factor
        new_width = input_shape[2] * self.scale_factor

        output_activations = tf.image.resize(input_activations, size=(new_height, new_width), method=self.method)

        if training:
            self.input_shape = input_activations.shape

        return output_activations

    def backward(self, output_gradient):
        gradient = tf.image.resize(
            output_gradient,
            size=(self.input_shape[1], self.input_shape[2]),
            method=self.method
        )
        
        if self.method.lower() == "bilinear":
            gradient = gradient * (self.scale_factor ** 2)
        
        del self.input_shape

        return gradient, []

    def initialize(self, input_shape, initialize_weights=True, dtype=np.float64):
        self.output_shape = input_shape[:]
        self.output_shape[:2] *= self.scale_factor

    def save(self):
        return [self.scale_factor, self.method], None

class MaxPool(Layer):
    def __init__(self, pooling_shape = [2, 2], pooling_stride = None, padding="SAME"):
        self.pooling_shape = pooling_shape

        if not pooling_stride:
            self.pooling_stride = self.pooling_shape
        else:
            self.pooling_stride = pooling_stride

        self.padding = padding

    def forward(self, input_activations, training=True):
        output_activations, argmax = tf.nn.max_pool_with_argmax(
            input_activations,
            [1, *self.pooling_shape[::-1], 1],
            [1, *self.pooling_stride[::-1], 1],
            self.padding
        )

        if training:
            self.pooling_indices = argmax
            self.input_shape = tf.shape(input_activations)

        return output_activations

    def backward(self, output_gradient):
        input_shape = self.input_shape
        
        flat_output_shape = tf.reduce_prod(input_shape)

        output_gradient_flat = tf.reshape(output_gradient, [-1])
        indices = tf.reshape(self.pooling_indices, [-1, 1])
        
        output_shape = tf.cast(input_shape, tf.int64)
        
        input_gradient = tf.scatter_nd(
            indices,
            output_gradient_flat,
            tf.cast([tf.reduce_prod(output_shape)], tf.int64)
        )
        
        input_gradient = tf.reshape(input_gradient, input_shape)
        del self.pooling_indices, self.input_shape
        
        return input_gradient, []

    def initialize(self, input_shape, initialize_weights=True, dtype=np.float64):
        output_shape = input_shape

        if self.padding.upper() == "SAME":
            output_shape[:-1] = np.ceil(output_shape[:-1] / np.array(self.pooling_shape)).astype(int)
        else:
            output_shape[:-1] = (output_shape[:-1] // np.array(self.pooling_shape)).astype(int)

        self.output_shape = output_shape

    def save(self):
        return [self.pooling_shape, self.pooling_stride, self.padding], None

class Dropout(Layer):
    def __init__(self, dropout_rate=0.5):
        self.dropout_rate = dropout_rate
        
    def forward(self, input_activations, training=True):
        if training:
            self.mask = ((cp.random.rand(*input_activations.shape) > self.dropout_rate) / ( 1 - self.dropout_rate)).astype(self.dtype)
        else:
            return input_activations

        output_activations = input_activations * self.mask
            
        return output_activations

    def backward(self, output_gradient):
        input_gradient = output_gradient * self.mask
        del self.mask
        return input_gradient, []

    def save(self):
        return [self.dropout_rate], None