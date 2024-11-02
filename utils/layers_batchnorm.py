import tensorflow.compat.v1 as tf, skimage, scipy, numpy as np, cupy as cp, time, os
from tensorflow.experimental.dlpack import from_dlpack, to_dlpack
from utils.functions import Processing
from utils.activations import *
from utils.loss import *

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))

class Layer:
    def __init__(self):
        pass

    @property
    def size(self):
        return 0

    def initialize(self, input_shape, initialize_weights=True, dtype=np.float64):
        self.output_shape = input_shape
        self.dtype = dtype

    def forward(self, input_activations, training=True):
        return input_activations

    def backward(self, input_activations, node_values):
        return node_values, []

    def update(self, optimizer, gradient, descent_values, learning_rate):
        pass

    def save(self):
        return [], None

    def load(self, data):
        pass


class Conv2d(Layer):
    def __init__(self, depth, kernel_shape=[3, 3], stride=[1, 1], variance="He", padding = "VALID"):
        self.kernel_shape = np.array(kernel_shape)[::-1]
        self.variance = variance
        self.depth = depth

        if isinstance(stride, int):
            stride = [stride, stride]

        self.stride = stride
        self.padding = padding

    @property
    def size(self):
        if not hasattr(self, 'kernels') or not hasattr(self, 'biases'):
            raise AttributeError("Layer needs to be initialized first.")
        
        return (self.kernels.nbytes + self.biases.nbytes)

    def forward(self, input_activations, training=True):
        # for i, kernels in enumerate(self.kernels):
        #     for kernel, channel in zip(kernels, input_activations):
        #         output_activations[i] += scipy.signal.correlate2d(channel, kernel, "valid")

        output_activations = tf.nn.conv2d(
            input_activations, 
            Processing.to_tensorflow(self.kernels), 
            strides=[1, *self.stride[::-1], 1], 
            padding=self.padding
        )

        output_activations += Processing.to_tensorflow(self.biases)

        if training:
            self.output_activations = output_activations

        return output_activations

    def backward(self, input_activations, node_values):
        new_node_values = cp.zeros(input_activations.shape)
        kernels_gradient = cp.zeros(self.kernels.shape)

        # for i, (kernels, kernel_node_values) in enumerate(zip(self.kernels, node_values)):
        #     for j, (image, kernel) in enumerate(zip(input_activations, kernels)):
                
        #         kernels_gradient[i, j] = scipy.signal.correlate2d(image, kernel_node_values, "valid")
        #         new_node_values[j] += scipy.signal.convolve2d(kernel_node_values, kernel, "full")

        # To tensorflow
    
        new_node_values = tf.nn.conv2d_backprop_input(
            input_sizes = input_activations.shape,
            filters = Processing.to_tensorflow(self.kernels),
            out_backprop = node_values,
            strides = [1, *self.stride[::-1], 1],
            padding = self.padding,
        )

        kernels_gradient = tf.nn.conv2d_backprop_filter(
            input = input_activations,
            filter_sizes = self.kernels.shape,
            out_backprop = node_values,
            strides = [1, *self.stride[::-1], 1],
            padding = self.padding,
        )

        kernels_gradient = tf.cast(kernels_gradient, node_values.dtype)
        new_node_values = tf.cast(new_node_values, node_values.dtype)

        kernels_biases_gradient = node_values

        return new_node_values, [
            cp.array(kernels_gradient), 
            cp.array(kernels_biases_gradient)
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
            if (not self.variance) or self.variance == "He":
                variance = cp.sqrt(2 / (fan_in))
                self.kernels = cp.random.normal(0, variance, (kernel_height, kernel_width, input_channels, output_channels))

            elif self.variance == "lecun":
                variance = cp.sqrt(1 / (fan_in))
                self.kernels = cp.random.normal(0, variance, (kernel_height, kernel_width, input_channels, output_channels))

            elif self.variance == "xavier":
                variance = cp.sqrt(6 / (fan_in + fan_out))
                self.kernels = cp.random.uniform(-variance, variance, (kernel_height, kernel_width, input_channels, output_channels))

            else:
                variance = self.variance
                self.kernels = cp.random.uniform(-variance, variance, (kernel_height, kernel_width, input_channels, output_channels))

            self.biases = cp.zeros(output_shape).astype(dtype)
            self.kernels = self.kernels.astype(dtype)

    def update(self, optimizer, gradient, descent_values, learning_rate):
        if not descent_values is None:
            kernel_descent_values, bias_descent_values = descent_values

        else:
            kernel_descent_values = None
            bias_descent_values = None

        kernels_gradient, kernels_biases_gradient = gradient

        self.kernels, new_kernel_descent_values = optimizer.apply_gradient(self.kernels, kernels_gradient, kernel_descent_values, learning_rate)
        self.biases, new_bias_descent_values = optimizer.apply_gradient(self.biases, kernels_biases_gradient, bias_descent_values, learning_rate)

        return [new_kernel_descent_values, new_bias_descent_values]

    def save(self):
        return [self.depth, self.kernel_shape, self.stride, self.variance, self.padding], [self.kernels.get(), self.biases.get()]

    def load(self, data):
        self.kernels, self.biases = data
        self.kernels = cp.array(self.kernels)
        self.biases = cp.array(self.biases)

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
            self.output_activations = output_activations

        return output_activations

    def backward(self, input_activations, node_values):
        input_shape = tf.shape(input_activations)
        
        flat_output_shape = tf.reduce_prod(input_shape)

        node_values_flat = tf.reshape(node_values, [-1])
        indices = tf.reshape(self.pooling_indices, [-1, 1])
        
        output_shape = tf.cast(input_shape, tf.int64)
        
        new_node_values = tf.scatter_nd(
            indices,
            node_values_flat,
            tf.cast([tf.reduce_prod(output_shape)], tf.int64)
        )
        
        new_node_values = tf.reshape(new_node_values, input_shape)
        
        return new_node_values, []

    def initialize(self, input_shape, initialize_weights=True, dtype=np.float64):
        output_shape = input_shape

        if self.padding.upper() == "SAME":
            output_shape[:-1] = np.ceil(output_shape[:-1] / np.array(self.pooling_shape)).astype(int)
        else:
            output_shape[:-1] = (output_shape[:-1] // np.array(self.pooling_shape)).astype(int)

        self.output_shape = output_shape

    def save(self):
        return [self.pooling_shape, self.pooling_stride, self.padding], None

class ConcatBlock(Layer):
    def __init__(self, layers, residual_layers):
        self.layers = layers
        self.residual_layers = residual_layers

    @property
    def size(self):
        if not hasattr(self, 'layers') or not hasattr(self, 'residual_layers'):
            raise AttributeError("Layer needs to be initialized first.")
        
        return sum([layer.size for layer in self.layers]) + sum([layer.size for layer in self.residual_layers])

    def forward(self, input_activations, training=True):
        residual = input_activations
        activations = input_activations

        with tf.device('/GPU:1'):
            for layer in self.layers:
                activations = layer.forward(activations)
        
        with tf.device('/GPU:1'):
            for layer in self.residual_layers:
                residual = layer.forward(residual)

        with tf.device('/GPU:1'):
            output_activations = tf.concat((activations, residual), axis=-1)

        if training:
            self.output_activations = output_activations
            self.activations_shape = activations.shape[-1]

        return output_activations

    def backward(self, input_activations, node_values):
        gradients = [None] * len(self.layers)
        _node_values = node_values[..., :self.activations_shape]

        with tf.device('/GPU:1'):
            for idx, layer in enumerate(self.layers[::-1]):
                current_layer_index = -(idx + 1)
                
                if idx == len(self.layers) - 1:
                    _input_activations = input_activations # Because the input_activations data would be outside of the concat block
                else:
                    _input_activations = self.layers[current_layer_index - 1].output_activations

                _node_values, gradient = layer.backward(_input_activations, _node_values)
                gradients[current_layer_index] = gradient

                del gradient, _input_activations, layer.output_activations

        residual_gradients = [None] * len(self.residual_layers)
        residual_node_values = node_values[..., self.activations_shape:]

        with tf.device('/GPU:1'):
            for idx, layer in enumerate(self.residual_layers[::-1]):
                current_layer_index = -(idx + 1)

                if idx == len(self.residual_layers) - 1:
                    _input_activations = input_activations
                else:
                    _input_activations = self.residual_layers[current_layer_index - 1].output_activations

                residual_node_values, gradient = layer.backward(_input_activations, residual_node_values)
                residual_gradients[current_layer_index] = gradient

        return _node_values + residual_node_values, [gradients, residual_gradients]

    def initialize(self, input_shape, initialize_weights=True, dtype=np.float64):
        output_shape = input_shape.copy()

        for layer in self.layers:
            layer.initialize(output_shape, dtype=dtype)
            output_shape = layer.output_shape
            print(output_shape, "MAIN")

        residual_output_shape = input_shape.copy()

        for layer in self.residual_layers:
            layer.initialize(residual_output_shape, dtype=dtype)
            residual_output_shape = layer.output_shape
            print(residual_output_shape, "RES")

        self.output_shape = output_shape
        self.output_shape[-1] += residual_output_shape[-1]

    def update(self, optimizer, gradient, descent_values, learning_rate):
        gradients, residual_gradients = gradient

        if descent_values:
            descent_values, residual_descent_values = descent_values

        else:
            descent_values = [None] * len(self.layers)
            residual_descent_values = [None] * len(self.residual_layers)

        new_descent_values = []

        for layer, layer_gradient, descent_value in zip(self.layers, gradients, descent_values):
            new_descent_value = layer.update(optimizer, layer_gradient, descent_value, learning_rate)
            new_descent_values.append(new_descent_value)

        new_residual_descent_values = []

        for layer, layer_gradient, descent_value in zip(self.residual_layers, residual_gradients, residual_descent_values):
            new_residual_descent_value = layer.update(optimizer, layer_gradient, descent_value, learning_rate)
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

    def load(self, data):
        layers_data, residual_layers_data = data
        layers = []
        residual_layers = []

        for layer_args, layer_data, layer_class in layers_data:
            layer = layer_class(*layer_args)
            layer.load(layer_data)
            layers.append(layer)

        for layer_args, layer_data, layer_class in residual_layers_data:
            layer = layer_class(*layer_args)
            layer.load(layer_data)
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
        activations = tf.identity(input_activations)

        with tf.device('/GPU:1'):
            for layer in self.layers:
                activations = layer.forward(activations)

        with tf.device('/GPU:1'):
            output_activations = activations + input_activations

        if training:
            self.output_activations = activations
            
        return output_activations

    def backward(self, input_activations, node_values):
        gradients = [None] * len(self.layers)
        _node_values = tf.identity(node_values)

        with tf.device('/GPU:1'):
            for idx, layer in enumerate(self.layers[::-1]):
                current_layer_index = -(idx + 1)
                
                if idx == len(self.layers) - 1:
                    _input_activations = input_activations # Because the input_activations data would be outside of the residual block
                else:
                    _input_activations = self.layers[current_layer_index - 1].output_activations

                _node_values, gradient = layer.backward(_input_activations, _node_values)
                    
                gradients[current_layer_index] = gradient

                del gradient, _input_activations, layer.output_activations

        return _node_values + node_values, [gradients]

    def initialize(self, input_shape, initialize_weights=True, dtype=np.float64):
        output_shape = input_shape.copy()

        for layer in self.layers:
            layer.initialize(output_shape, dtype=dtype)
            output_shape = layer.output_shape
            print(output_shape, "MAIN")

        self.output_shape = output_shape

    def update(self, optimizer, gradients, descent_values, learning_rate):
        new_descent_values = []
        if descent_values is None:
            descent_values = []

        for layer, layer_gradient, descent_value in zip(self.layers, gradients, descent_values):
            new_descent_value = layer.update(optimizer, layer_gradient, descent_value, learning_rate)
            new_descent_values.append(new_descent_value)

        return new_descent_values

    def save(self):
        layers_data = []

        for layer in self.layers:
            layers_data.append(list(layer.save()) + [layer.__class__])

        return [None], [layers_data]

    def load(self, data):
        layers_data = data
        layers = []

        for layer_args, layer_data, layer_class in data:
            layer = layer_class(*layer_args)
            layer.load(layer_data)
            layers.append(layer)

        self.layers = layers

class Dense(Layer):
    def __init__(self, depth, variance="He"):
        self.variance = variance
        self.depth = np.array(depth)

    @property
    def size(self):
        if not hasattr(self, 'layer'):
            raise AttributeError("Layer needs to be initialized first.")

        return self.layer.nbytes

    def forward(self, input_activations, training=True):
        start_time = time.perf_counter()
        weights = Processing.to_tensorflow(self.layer[..., :-1])
        biases = Processing.to_tensorflow(self.layer[..., -1])

        with tf.device("/GPU:1"):
            output_activations = tf.tensordot(input_activations, weights, axes=[[1], [1]]) + biases

        end_time = time.perf_counter()

        if training:
            self.output_activations = output_activations

        return output_activations

    def backward(self, input_activations, old_node_values):
        weights = Processing.to_tensorflow(self.layer[..., :-1])
        biases = Processing.to_tensorflow(self.layer[..., -1])

        batch_size = input_activations.shape[0]

        with tf.device("/GPU:1"):
            new_node_values = tf.tensordot(old_node_values, tf.transpose(weights), axes=[[1], [1]])

            weights_derivative = old_node_values[..., None] * tf.reshape(input_activations, (batch_size, 1, -1))
            bias_derivative = old_node_values

        gradient = cp.dstack((weights_derivative, bias_derivative))
        
        gradient = gradient.mean(axis=0)

        return new_node_values, gradient

    def initialize(self, input_shape, initialize_weights=True, dtype=np.float64):
        inputs = int(input_shape)

        if initialize_weights:
            if (not self.variance) or self.variance == "He":
                variance = cp.sqrt(2 / (inputs))
                self.layer = cp.random.normal(0, variance, (self.depth, inputs + 1))

            elif self.variance == "lecun":
                variance = cp.sqrt(1 / (inputs))
                self.layer = cp.random.normal(0, variance, (self.depth, inputs + 1))

            elif self.variance == "xavier":
                variance = cp.sqrt(6 / (inputs + self.depth))
                self.layer = cp.random.uniform(-variance, variance, (self.depth, inputs + 1))

            else:
                variance = self.variance
                self.layer = cp.random.uniform(-variance, variance, (self.depth, inputs + 1))
            
            self.layer[..., -1] = 0 # Set biases to zero
            self.layer = self.layer.astype(dtype)

        self.output_shape = self.depth

    def update(self, optimizer, gradient, descent_values, learning_rate):
        self.layer, new_descent_values = optimizer.apply_gradient(self.layer, gradient, descent_values, learning_rate)

        return new_descent_values

    def save(self):
        return [self.depth, self.variance], self.layer.get()

    def load(self, data):
        self.layer = cp.array(data)

class Activation(Layer):
    def __init__(self, activation_function):
        self.activation_function = activation_function

    def forward(self, input_activations, training=True):
        output_activations = self.activation_function.forward(input_activations)

        if training:
            self.output_activations = output_activations

        return output_activations

    def backward(self, input_activations, node_values):
        return node_values * self.activation_function.backward(self.output_activations), []

    def save(self):
        return [self.activation_function], None

class BatchNorm(Layer):
    def __init__(self, momentum=0.9, epsilon=1e-8):
        self.momentum = momentum
        self.epsilon = epsilon

    @property
    def size(self):
        if not hasattr(self, 'gamma') or not hasattr(self, 'beta') or not hasattr(self, 'running_mean') or not hasattr(self, 'running_var'):
            raise AttributeError("Layer needs to be initialized first.")

        return self.gamma.nbytes + self.beta.nbytes + self.running_mean.nbytes + self.running_var.nbytes

    def forward(self, x, training=True):
        if x.ndim == 4:  # For 4D input [B, H, W, C]
            axes = (0, 1, 2)  # Batch, Height, Width
        elif x.ndim == 2:  # For 2D input [B, D]
            axes = 0  # Batch
        else:
            raise ValueError("Unsupported input shape")

        with tf.device("/GPU:1"):
            if training:
                batch_mean = tf.reduce_mean(x, axis=axes, keepdims=True)
                batch_var = tf.math.reduce_variance(x, axis=axes, keepdims=True)

                x_centered = x - batch_mean

                self.stddev_inv = 1 / tf.sqrt(batch_var + self.epsilon)
                x_norm = x_centered * self.stddev_inv

                self.batch_mean = batch_mean
                self.batch_var = batch_var
            else:
                x_norm = (x - Processing.to_tensorflow(self.running_mean)) / tf.sqrt(Processing.to_tensorflow(self.running_var) + self.epsilon)

            output_activations = Processing.to_tensorflow(self.gamma) * x_norm + Processing.to_tensorflow(self.beta)

        # print(self.stddev_inv, self.running_var, self.batch_mean, self.batch_var)
        # print(tf.reduce_mean(output_activations), tf.math.reduce_variance(output_activations))

        if training:
            self.output_activations = output_activations

        return output_activations

    def backward(self, input_activations, old_node_values):
        batch_size = input_activations.shape[0]

        if input_activations.ndim == 4:  # For convolutional layers
            axes = (0, 1, 2)
        else:  # For fully connected layers
            axes = 0

        x_centered = input_activations - self.batch_mean
        
        with tf.device("/GPU:1"):
            gamma_gradient = x_centered * self.stddev_inv * old_node_values
            beta_gradient = old_node_values

            grad_x_norm = old_node_values * Processing.to_tensorflow(self.gamma)
            grad_var = grad_x_norm * x_centered * -0.5 * (self.stddev_inv**3)
            grad_mean = grad_x_norm * -self.stddev_inv + grad_var * -2 * x_centered / batch_size

            new_node_values = grad_x_norm * self.stddev_inv + grad_var * 2 * x_centered / batch_size + grad_mean / batch_size

            return new_node_values, [cp.array(gamma_gradient), cp.array(beta_gradient)]

    def update(self, optimizer, gradient, descent_values, learning_rate):
        dgamma, dbeta = gradient

        if descent_values is not None:
            gamma_descent_values, beta_descent_values = descent_values
        else:
            gamma_descent_values = None
            beta_descent_values = None

        with tf.device("/GPU:1"):
            self.running_mean = self.momentum * self.running_mean + cp.array((1 - self.momentum) * self.batch_mean)
            self.running_var = self.momentum * self.running_var + cp.array((1 - self.momentum) * self.batch_var)

            self.gamma, new_gamma_descent_values = optimizer.apply_gradient(self.gamma, dgamma, gamma_descent_values, learning_rate)
            self.beta, new_beta_descent_values = optimizer.apply_gradient(self.beta, dbeta, beta_descent_values, learning_rate)

        return [new_gamma_descent_values, new_beta_descent_values]

    def initialize(self, input_shape, initialize_weights=True, dtype=np.float64):
        if np.array(input_shape).ndim > 1:
            features = np.array(input_shape)[-1]
        else:
            features = input_shape

        if initialize_weights:
            self.gamma = cp.ones(features).astype(dtype)
            self.beta = cp.zeros(features).astype(dtype)
            self.running_mean = cp.zeros(features).astype(dtype)
            self.running_var = cp.ones(features).astype(dtype)

        self.output_shape = input_shape

    def save(self):
        return [self.momentum], [self.gamma.get(), self.beta.get(), self.running_mean.get(), self.running_var.get()]

    def load(self, data):
        self.gamma, self.beta, self.running_mean, self.running_var = data
        self.gamma = cp.array(self.gamma)
        self.beta = cp.array(self.beta)
        self.running_mean = cp.array(self.running_mean)
        self.running_var = cp.array(self.running_var)

class Input(Layer):
    def __init__(self, input_shape):
        self.output_shape = np.array(input_shape)

    def forward(self, input_activations, training=True):
        output_activations = Processing.to_tensorflow(cp.array(input_activations).astype(self.dtype))
        if training:
            self.output_activations = output_activations

        return output_activations

    def initialize(self, input_shape, initialize_weights=True, dtype=np.float64):
        self.dtype = dtype
        self.output_shape = input_shape

    def save(self):
        return [self.output_shape], self.dtype

    def load(self, data):
        self.dtype = data

class Flatten(Layer):
    def __init__(self):
        pass

    def forward(self, input_activations, training=True):
        output_activations = tf.reshape(input_activations, (input_activations.shape[0], -1))

        if training:
            self.output_activations = output_activations

        self.input_shape = input_activations.shape
        return output_activations

    def backward(self, input_activations, node_values):
        return tf.reshape(node_values, self.input_shape), []

    def initialize(self, input_shape, initialize_weights=True, dtype=np.float64):
        self.output_shape = input_shape.prod()

class Reshape(Layer):
    def __init__(self, output_shape):
        self.output_shape = output_shape

    def forward(self, input_activations, training=True):
        output_activations = tf.reshape(input_activations, output_shape)

        if training:
            self.output_activations = output_activations

        self.input_shape = input_activations.shape
        return output_activations

    def backward(self, input_activations, node_values):
        return tf.reshape(node_values, self.input_shape), []

    def save(self):
        return [self.output_shape], None

class Space2Depth(Layer):
    def __init__(self, block_size):
        self.block_size = block_size

    def forward(self, input_activations, training=True):
        with tf.device('/GPU:1'):
            output_activations = tf.nn.space_to_depth(
                    input_activations, 
                    self.block_size
                )

        if training:
            self.output_activations = output_activations

        return output_activations

    def backward(self, input_activations, node_values):
        with tf.device('/GPU:1'):
            return tf.nn.depth_to_space(
                    node_values, 
                    self.block_size
                ), []

    def initialize(self, input_shape, initialize_weights=True, dtype=np.float64):
        self.output_shape = input_shape[:]
        self.output_shape[-1] *= self.block_size ** 2
        self.output_shape[:-1] //= self.block_size

    def save(self):
        return [self.block_size], None

class Dropout(Layer):
    def __init__(self, dropout_rate=0.5):
        self.dropout_rate = dropout_rate
        
    def forward(self, input_activations, training=True):
        if training:
            self.mask = Processing.to_tensorflow(((cp.random.rand(*input_activations.shape) > self.dropout_rate) / ( 1 - self.dropout_rate)).astype(self.dtype))
        else:
            return input_activations

        output_activations = input_activations * self.mask

        if training:
            self.output_activations = output_activations
            
        return output_activations

    def backward(self, input_activations, node_values):
        return node_values * self.mask, []

    def save(self):
        return [self.dropout_rate], None
