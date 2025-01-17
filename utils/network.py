import cupy as cp, utils.layers, time, math, os
from datetime import datetime
from utils.optimizers import *
from utils.activations import *
from utils.loss import *
from utils.functions import Processing

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow.compat.v1 as tf

class Network:
    def __init__(self, model=[], backprop_layer_indices=[-1], addon_layers=[], loss_function=MSE(), optimizer=SGD(), gpu_mem_frac=1, dtype=np.float64, scheduler=None):
        self.model = model
        self.loss_functions = loss_function if isinstance(loss_function, list) else [loss_function] * len(backprop_layer_indices)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dtype = dtype

        self.backprop_layer_indices = sorted(backprop_layer_indices)
        self.addon_layers = addon_layers

        self.learning_rate = None
        self.starting_epoch = 0

        physical_gpus = tf.config.experimental.list_physical_devices('GPU')

        print(physical_gpus)

        if physical_gpus:
            device = cp.cuda.Device(0)  # Assuming GPU index 0
            total_memory = device.mem_info[1]
            total_memory = total_memory // (1024 ** 2)

            
            try:
                tf.config.experimental.set_virtual_device_configuration(
                    physical_gpus[0],
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1),
                    tf.config.experimental.VirtualDeviceConfiguration(memory_limit=gpu_mem_frac * total_memory)]
                )

                # List the logical GPUs after setting up the virtual devices
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(f"Available Memory: {total_memory} (MiB), Physical GPUs: {len(physical_gpus)}")

            except RuntimeError as e:
                print(f"Error: {e}") 
        else:
            raise RuntimeError("Must have atleast one CUDA compatible GPU")

    @property
    def size(self):
        total_size = 128
        for layer in self.model[1:]:
            total_size += int(layer.size)

        for layers in self.addon_layers:
            for layer in layers:
                total_size += int(layer.size)

        return total_size

    def compile(self):
        input_shape = self.model[0].output_shape.copy()

        print(input_shape)

        with tf.device('/GPU:1'):
            for idx, layer in enumerate(self.model):
                layer.initialize(input_shape, dtype=self.dtype)
                input_shape = layer.output_shape.copy()

                print(layer, input_shape)

                negative_index = idx - len(self.model)

                if (idx in self.backprop_layer_indices) or (negative_index in self.backprop_layer_indices):
                    if self.addon_layers:
                        if idx in self.backprop_layer_indices:
                            index = self.backprop_layer_indices.index(idx)
                        else:
                            index = self.backprop_layer_indices.index(negative_index)

                        _input_shape = input_shape.copy()

                        for layer in self.addon_layers[index]:
                            layer.initialize(_input_shape, dtype=self.dtype)
                            _input_shape = layer.output_shape.copy()

                            print(layer, _input_shape, "ADDON")

        labels = {
            0: 'B',
            1: 'KiB',
            2: 'MiB',
            3: 'GiB',
            4: 'TiB', # Why are you using this library for a model this large...
            5: 'PiB' # Please use tensorflow or Pytorch.
        }

        size = self.size

        unit = math.floor(math.log(size, 1024))
        label = labels[unit]

        print(f"Model Size ({label}):", size / (1024) ** unit)

    def save(self):
        with tf.device('/GPU:1'):
            model_data = []
            for layer in self.model:
                model_data.append(list(layer.save()) + [layer.__class__.__name__])

            combined_addon_data = []

            for layers in self.addon_layers:
                addon_data = []
                for layer in layers:
                    addon_data.append(list(layer.save()) + [layer.__class__.__name__])

                combined_addon_data.append(addon_data)

            return (
                model_data, combined_addon_data, 
                self.backprop_layer_indices, self.loss_functions, self.scheduler, 
                self.learning_rate, self.epoch, self.dtype
            )

    def load(self, save_data):
        input_shape = None

        model_data, combined_addon_data, backprop_layer_indices, loss_functions, scheduler, learning_rate, starting_epoch, dtype = save_data

        self.backprop_layer_indices = backprop_layer_indices
        self.starting_epoch = starting_epoch
        self.loss_functions = loss_functions
        self.learning_rate = learning_rate
        self.scheduler = scheduler
        self.dtype = dtype

        model = []

        for layer_args, layer_data, layer_type in model_data:
            layer_class = getattr(utils.layers, layer_type)
            layer = layer_class(*layer_args)

            layer.load(layer_data, self.dtype)

            model.append(layer)

        addon_layers = []

        for addon_data in combined_addon_data:
            layers = []
            for layer_args, layer_data, layer_type in addon_data:
                layer_class = getattr(utils.layers, layer_type)
                layer = layer_class(*layer_args)

                layer.load(layer_data, self.dtype)

                layers.append(layer)

            addon_layers.append(layers)

        self.model = model
        self.addon_layers = addon_layers

    def forward(self, activations, training=True):
        outputs = []
        
        with tf.device('/GPU:1'):
            for idx, layer in enumerate(self.model):
                time1 = time.perf_counter()
                activations = layer.forward(activations, training=training)
                time2 = time.perf_counter()

                negative_index = idx - len(self.model)

                if (idx in self.backprop_layer_indices) or (negative_index in self.backprop_layer_indices):
                    _activations = tf.identity(activations)

                    if self.addon_layers:
                        if idx in self.backprop_layer_indices:
                            index = self.backprop_layer_indices.index(idx)
                        else:
                            index = self.backprop_layer_indices.index(negative_index)

                        for layer in self.addon_layers[index]:
                            _activations = layer.forward(_activations, training=training)

                    outputs.append(_activations)

                    del _activations

        return outputs

    def cost_gradient(self, outputs, expected_outputs, loss_function):
        costs = []
        node_values = []

        outputs = tf.constant(outputs, dtype=self.dtype)
        expected_outputs = tf.constant(expected_outputs, dtype=self.dtype)

        with tf.GradientTape() as tape:
            tape.watch(outputs)
            cost = loss_function.forward(outputs, expected_outputs)

            if isinstance(cost, tuple):
                true_cost = cost[1]
                cost = cost[0]
            else:
                true_cost = cost

        node_values = tape.gradient(cost, outputs)

        return true_cost.numpy().astype(self.dtype), node_values

    def backward(self, outputs, expected_outputs):
        node_values = None
        gradients = [None] * (len(self.model) - 1) 
        addon_gradients = [None] * len(self.addon_layers)
        costs = []

        with tf.device('/GPU:1'):
            for idx, layer in enumerate(self.model[::-1][:-1]):
                current_layer_index = -(idx + 1) # - counting_offset

                positive_index = len(self.model) - idx - 1

                time1 = time.perf_counter()

                if (current_layer_index in self.backprop_layer_indices) or (positive_index in self.backprop_layer_indices):
                    if current_layer_index in self.backprop_layer_indices:
                        addon_index = self.backprop_layer_indices.index(current_layer_index)
                    else:
                        addon_index = self.backprop_layer_indices.index(positive_index)

                    output = outputs[addon_index]

                    indexed_expected_outputs = tf.constant([
                        expected_output[addon_index] for expected_output in expected_outputs
                    ], dtype=self.dtype)

                    cost, _node_values = self.cost_gradient(output, indexed_expected_outputs, self.loss_functions[addon_index])
                    costs.append(cost.tolist())

                    if self.addon_layers: # Calculate Gradient for addon layers

                        addon_gradient = [None] * len(self.addon_layers[addon_index])
                        
                        for idx, addon_layer in enumerate(self.addon_layers[addon_index][::-1]):
                            current_layer_addon_index = -(idx + 1)

                            _node_values, gradient = addon_layer.backward(_node_values)
                            addon_gradient[current_layer_addon_index] = gradient
                        addon_gradients[addon_index] = addon_gradient
                        
                        del addon_gradient

                    if node_values is not None:
                        node_values += _node_values
                    else:
                        node_values = _node_values

                    del _node_values, indexed_expected_outputs, output, cost

                if node_values is None:
                    continue

                node_values, gradient = layer.backward(node_values)
                gradients[current_layer_index] = gradient

                time2 = time.perf_counter()

                del gradient

        return gradients, addon_gradients, costs[::-1]

    def _sum_gradients(self, gradients):
        summed_array = gradients[0]
        gradients = gradients[1:]

        for gradient in gradients:
            for idx, layer in enumerate(gradient):
                if not isinstance(layer, (list, tuple, None)):
                    summed_array[idx] += layer
                else:
                    summed_array[idx] = self._sum_gradients([gradient[idx] for gradient in gradients])

        return summed_array

    def _scale_gradient(self, gradient, scale):
        for idx, layer in enumerate(gradient):
            if not isinstance(layer, (list, tuple, None)):
                if len(layer) == 0:
                    continue
                gradient[idx] /= scale
            else:
                gradient[idx] = self._scale_gradient(layer, scale)

        return gradient

    def _numpify_values(self, values):
        for idx, layer in enumerate(values):
            if not isinstance(layer, (list, tuple, None)):
                if len(layer) == 0:
                    values[idx] = np.array([])
                values[idx] = layer.numpy()
            else:
                if layer is None:
                    continue
                values[idx] = self._numpify_values(layer)

        return values

    def _cupify_values(self, values):
        values = tf.identity(values)
        for idx, layer in enumerate(values):
            if isinstance(layer, np.ndarray):
                if layer.size == 0:
                    values[idx] = cp.array([])
                values[idx] = cp.array(layer)
            else:
                if layer is None:
                    continue
                values[idx] = self._cupify_values(layer)

        return values

    def fit(self, xdata=None, ydata=None, generator=None, batch_size=8, learning_rate=0.01, epochs=100, gradient_transformer=None):

        if not generator:
            xdata = np.array(xdata, dtype=self.dtype)
            dataset_size = xdata.shape[0]

        else:
            dataset_size = generator.dataset_size

        if self.learning_rate:
            learning_rate = self.learning_rate

        if (gradient_transformer is not None) and not isinstance(gradient_transformer, list):
            gradient_transformer = (gradient_transformer,)

        iterations = int(epochs * (dataset_size / batch_size))

        self.batch_size = batch_size

        cost = None
        delay = None

        optimizer_values = [None] * len(self.model)
        addon_optimizer_values = [[None] * len(layers) for layers in self.addon_layers]

        for iteration in range(iterations):
            start_time = time.perf_counter()

            epoch = self.starting_epoch + (batch_size / dataset_size) * iteration

            self.epoch = epoch
            self.learning_rate = learning_rate

            if self.scheduler:
                learning_rate = self.scheduler.forward(epoch)

            if not generator:
                choices = np.random.choice(xdata.shape[0], size=batch_size, replace=False)
            
                selected_xdata = xdata[choices].astype(self.dtype)
                selected_ydata = [ydata[choice] for choice in choices]

            else:
                selected_xdata, selected_ydata = generator()
                selected_xdata = selected_xdata.astype(self.dtype)
                selected_ydata = selected_ydata

            outputs = self.forward(selected_xdata)
            gradient, addon_gradients, cost = self.backward(outputs, selected_ydata)

            del selected_xdata, selected_ydata

            if gradient_transformer:
                for transformer in gradient_transformer:
                    gradient, addon_gradients = transformer.forward([gradient, addon_gradients])

            with tf.device('/GPU:1'):

                for addon_index, (addon_layers, addon_gradient, addon_optimizer_value) in enumerate(zip(self.addon_layers, addon_gradients, addon_optimizer_values)):
                    for idx, (layer, layer_gradient, descent_values) in enumerate(zip(addon_layers, addon_gradient, addon_optimizer_value)):
                        new_descent_values = layer.update(self.optimizer, layer_gradient, descent_values, learning_rate, iteration)

                        addon_optimizer_values[addon_index][idx] = new_descent_values

                if self.addon_layers:
                    del addon_gradient

                for idx, (layer, layer_gradient, descent_values) in enumerate(zip(self.model[1:], gradient, optimizer_values)):
                    if layer_gradient is None:
                        continue
                    new_descent_values = layer.update(self.optimizer, layer_gradient, descent_values, learning_rate, iteration)

                    optimizer_values[idx] = new_descent_values
                    del new_descent_values

                del gradient

            end_time = time.perf_counter()
            delay = end_time - start_time

            now = datetime.now()
            date_string = now.strftime("%H:%M:%S")

            print(f"[LOG {date_string}] Iteration: {iteration}, Epoch: {epoch}, Delay: {delay}, Learning Rate: {learning_rate}")
            end_time = time.perf_counter()

            yield cost