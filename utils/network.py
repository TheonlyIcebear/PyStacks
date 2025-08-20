import utils.layers, time, math, os
from datetime import datetime
from utils.optimizers import *
from utils.loss import *
from utils.functions import Processing

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow.compat.v1 as tf

gpu_mem_frac = 0.99  # Fraction of GPU memory to use

physical_gpus = tf.config.experimental.list_physical_devices('GPU')

if physical_gpus:
    device = cp.cuda.Device(0)  # Assuming GPU index 0
    total_memory = device.mem_info[1] / (1024 ** 2)
    print(total_memory)

    
    try:
        tf.config.experimental.set_virtual_device_configuration(
            physical_gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=gpu_mem_frac * total_memory)]
        )

        print(f"Available Memory: {total_memory} (MB), Physical GPUs: {len(physical_gpus)}")

    except RuntimeError as e:
        print(f"Error: {e}") 
else:
    raise RuntimeError("Must have atleast one CUDA compatible GPU")

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
        self.epoch = 0

    @property
    def size(self):
        total_size = 0
        for layer in self.model:
            total_size += int(layer.size)

        for layers in self.addon_layers:
            for layer in layers:
                total_size += int(layer.size)

        return total_size

    def compile(self):
        input_shape = self.model[0].output_shape.copy()

        print(input_shape)

        with tf.device('/GPU:0'):
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
        with tf.device('/GPU:0'):
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

    def load(self, save_data, new_dtype=None):
        input_shape = None

        model_data, combined_addon_data, backprop_layer_indices, loss_functions, scheduler, learning_rate, starting_epoch, dtype = save_data

        self.backprop_layer_indices = backprop_layer_indices
        self.starting_epoch = starting_epoch
        self.loss_functions = loss_functions
        self.learning_rate = learning_rate
        self.scheduler = scheduler
        self.dtype = dtype if not new_dtype else new_dtype

        with tf.device('/GPU:0'):
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
        
        with tf.device('/GPU:0'):
            for idx, layer in enumerate(self.model):
                time1 = time.perf_counter()
                activations = layer.forward(activations, training=training)
                time2 = time.perf_counter()

                negative_index = idx - len(self.model)

                if (idx in self.backprop_layer_indices) or (negative_index in self.backprop_layer_indices):
                    _activations = activations

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

        outputs = tf.cast(outputs, dtype=self.dtype)
        expected_outputs = tf.cast(expected_outputs, dtype=self.dtype)

        with tf.GradientTape() as tape:
            tape.watch(outputs)
            cost = loss_function.forward(outputs, expected_outputs)

            if isinstance(cost, tuple):
                true_cost = cost[1]
                cost = cost[0]
            else:
                true_cost = cost

        node_values = tape.gradient(cost, outputs)

        return true_cost, node_values

    def backward(self, outputs, expected_outputs):
        node_values = None
        gradients = [None] * (len(self.model) - 1) 
        addon_gradients = [None] * len(self.addon_layers)
        costs = []

        with tf.device('/GPU:0'):
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

                    indexed_expected_outputs = tf.gather(expected_outputs, addon_index, axis=1)

                    cost, _node_values = self.cost_gradient(output, indexed_expected_outputs, self.loss_functions[addon_index])
                    costs.append(cost)

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

        return tuple(gradients), tuple(addon_gradients), costs[::-1]


    def _sum_gradients(self, gradients):
        return list(tf.nest.map_structure(lambda *gs: tf.add_n([g for g in gs if g is not None]), *gradients))

    def _scale_gradient(self, gradient, scale):
        return list(tf.nest.map_structure(lambda g: g / scale if g is not None else None, gradient))

    # def _numpify_values(self, values):
    #     for idx, layer in enumerate(values):
    #         if not isinstance(layer, (list, tuple, type(None))):
    #             if len(layer) == 0:
    #                 values[idx] = np.array([])
    #             values[idx] = layer.numpy()
    #         else:
    #             if layer is None:
    #                 continue
    #             values[idx] = self._numpify_values(layer)

    #     return values

    def _cupify_values(self, values):
        values = values[:]
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
    
    def _tensorify_values(self, values, dtype):
        if tf.is_tensor(values):
            return tf.cast(values, dtype)
        
        for idx, layer in enumerate(values):
            if not isinstance(layer, (list, tuple, type(None))):
                values[idx] = tf.cast(layer, dtype)
            else:
                if layer is None:
                    continue
                values[idx] = self._tensorify_values(layer, dtype)

        return values
    
    def _zero_grad(self, values, dtype):
        values = tf.nest.map_structure(
            lambda t: tf.zeros_like(t, dtype=dtype) if isinstance(t, tf.Tensor) else t,
            values
        )

        return values
    
    def _apply_gradient_transformer(self, gradient, transformer):
        for t in transformer:
            gradient = t(gradient)
    
    @tf.function(reduce_retracing=True)
    def _train_step(self, selected_xdata, selected_ydata, accumulated_gradient, accumulated_addon_gradient, optimizer_values, addon_optimizer_values, accumulate, gradient_transformer, iteration, learning_rate):
        outputs = self.forward(selected_xdata)
        gradient, addon_gradients, cost = self.backward(outputs, selected_ydata)

        iteration = tf.cast(iteration, self.dtype)
        learning_rate = tf.cast(learning_rate, self.dtype)

        gradient = list(gradient)
        addon_gradients = list(addon_gradients)

        del selected_xdata, selected_ydata

        if not accumulated_gradient:
            accumulated_gradient = gradient
            accumulated_addon_gradient = addon_gradients
        else:
            accumulated_gradient = self._sum_gradients([accumulated_gradient, gradient])
            accumulated_addon_gradient = self._sum_gradients([accumulated_addon_gradient, addon_gradients])

        accumulated_gradient = accumulated_gradient
        accumulated_addon_gradient = accumulated_addon_gradient

        # Update step
        if tf.equal(tf.math.floormod(int(iteration) + 1, accumulate), 0):

            with tf.device('/GPU:0'):
                if gradient_transformer:
                    transformed_accumulated_gradient = self._apply_gradient_transformer(accumulated_gradient, gradient_transformer)
                    transformed_accumulated_addon_gradient = self._apply_gradient_transformer(accumulated_addon_gradient, gradient_transformer)

            accumulated_gradient = self._zero_grad(gradient, self.dtype)
            accumulated_addon_gradient = self._zero_grad(addon_gradients, self.dtype)

            start_time2 = time.perf_counter()

            if accumulate > 1:
                gradient = self._scale_gradient(transformed_accumulated_gradient, accumulate)
                addon_gradients = self._scale_gradient(transformed_accumulated_addon_gradient, accumulate)

            _optimizer_values = [tf.constant(0, self.dtype)] * len(self.model)
            _addon_optimizer_values = [[tf.constant(0, self.dtype)] * len(layers) for layers in self.addon_layers]

            for addon_index, (addon_layers, addon_gradient, addon_optimizer_value) in enumerate(zip(self.addon_layers, addon_gradients, addon_optimizer_values)):
                for idx, (layer, layer_gradient, descent_values) in enumerate(zip(addon_layers, addon_gradient, addon_optimizer_value)):

                    new_descent_values = layer.update(self.optimizer, layer_gradient, descent_values, learning_rate, iteration)

                    _addon_optimizer_values[addon_index][idx] = new_descent_values
                    del new_descent_values

            if self.addon_layers:
                del addon_gradients

            for idx, (layer, layer_gradient, descent_values) in enumerate(zip(self.model[1:], gradient, optimizer_values)):
                if layer_gradient is None:
                    continue

                new_descent_values = layer.update(self.optimizer, layer_gradient, descent_values, learning_rate, iteration)

                _optimizer_values[idx] = new_descent_values

            end_time2 = time.perf_counter()

        else:
            _optimizer_values = optimizer_values
            _addon_optimizer_values = addon_optimizer_values

            accumulated_gradient = [i for i in accumulated_gradient]
            accumulated_addon_gradient = [i for i in accumulated_addon_gradient]

        return accumulated_gradient, accumulated_addon_gradient, _optimizer_values, _addon_optimizer_values, cost


    def fit(self, xdata=None, ydata=None, generator=None, batch_size=8, learning_rate=0.01, epochs=100, accumulate=1, gradient_transformer=None):

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

        optimizer_values = [tf.constant(0, self.dtype)] * len(self.model)
        addon_optimizer_values = [[tf.constant(0, self.dtype)] * len(layers) for layers in self.addon_layers]

        for addon_index, (addon_layers, addon_optimizer_value) in enumerate(zip(self.addon_layers, addon_optimizer_values)):
            for idx, (layer, descent_values) in enumerate(zip(addon_layers, addon_optimizer_value)):

                new_descent_values = self._tensorify_values(layer.update(self.optimizer, None, descent_values, learning_rate, 0, apply=False), dtype=self.dtype)

                addon_optimizer_values[addon_index][idx] = new_descent_values
                del new_descent_values

        if self.addon_layers:
            del addon_gradients

        for idx, (layer, descent_values) in enumerate(zip(self.model[1:], optimizer_values)):

            new_descent_values = self._tensorify_values(layer.update(self.optimizer, None, descent_values, learning_rate, 0, apply=False), dtype=self.dtype)

            optimizer_values[idx] = new_descent_values

        accumulated_gradient = None
        accumulated_addon_gradient = None

        for iteration in range(iterations):
            start_time = time.perf_counter()

            epoch = self.starting_epoch + (batch_size / dataset_size) * iteration

            self.epoch = epoch
            self.learning_rate = learning_rate

            if self.scheduler:
                learning_rate = tf.constant(self.scheduler(epoch), dtype=self.dtype)

            if not generator:
                choices = np.random.choice(xdata.shape[0], size=batch_size, replace=False)
            
                selected_xdata = xdata[choices].astype(self.dtype)
                selected_ydata = [ydata[choice] for choice in choices]

            else:
                selected_xdata, selected_ydata = generator()
                selected_xdata = selected_xdata.astype(self.dtype)
                selected_ydata = selected_ydata
            
            accumulated_gradient, accumulated_addon_gradient, optimizer_values, addon_optimizer_values, cost = self._train_step(selected_xdata, selected_ydata, accumulated_gradient, accumulated_addon_gradient, optimizer_values, addon_optimizer_values, accumulate, gradient_transformer, tf.constant(iteration, dtype=self.dtype), tf.constant(learning_rate, dtype=self.dtype))
            del selected_xdata, selected_ydata

            end_time = time.perf_counter()
            delay = end_time - start_time

            now = datetime.now()
            date_string = now.strftime("%H:%M:%S")

            print(f"[LOG {date_string}] Iteration: {iteration}, Epoch: {epoch}, Delay: {delay}, Learning Rate: {learning_rate}")
            end_time = time.perf_counter()

            yield cost