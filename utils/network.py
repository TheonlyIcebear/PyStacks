import multiprocessing, threading, cupy as cp, awkward as ak, utils.layers, pickle, time, copy, math, gc, os
from tqdm.auto import tqdm, trange
from datetime import datetime
from utils.optimizers import *
from utils.activations import *
from utils.loss import *
from utils.functions import Processing

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow.compat.v1 as tf
from multiprocessing import Manager, Pool, Queue
from multiprocessing.shared_memory import SharedMemory

class Network:
    def __init__(self, model=[], loss_function=MSE(), optimizer=SGD(), gpu_mem_frac=1, dtype=np.float32, scheduler=None):
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dtype = dtype

        physical_gpus = tf.config.experimental.list_physical_devices('GPU')

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
            raise RuntimeError("Must have atleast one GPU")

    @property
    def size(self):
        total_size = 128
        for layer in self.model[1:]:
            total_size += int(layer.size)

        return total_size

    def compile(self):
        input_shape = self.model[0].output_shape.copy()

        print(input_shape)
        for layer in self.model:
            layer.initialize(input_shape, dtype=self.dtype)
            input_shape = layer.output_shape.copy()

            print(layer, input_shape)

        print(self.model[0].output_shape)

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
        model_data = []
        for layer in self.model:
            model_data.append(list(layer.save()) + [layer.__class__.__name__])

        return model_data, self.loss_function, self.scheduler, self.dtype

    def load(self, save_data):
        input_shape = None

        model_data, loss_function, scheduler, dtype = save_data
        self.loss_function = loss_function
        self.scheduler = scheduler
        self.dtype = dtype

        model = []
        size = 0

        for layer_args, layer_data, layer_type in model_data:
            layer_class = getattr(utils.layers, layer_type)
            layer = layer_class(*layer_args)

            layer.load(layer_data)

            model.append(layer)

        self.model = model

    def forward(self, activations, training=True):
        with tf.device('/GPU:1'):
            for idx, layer in enumerate(self.model):
                print(idx / len(self.model) * 100)
                activations = layer.forward(activations, training=training)
                
        return activations

    def backward(self, outputs, expected_outputs):
        expected_outputs = Processing.to_tensorflow(cp.array(expected_outputs))

        with tf.device('/GPU:1'):
            
            costs = []
            node_values = []

            total_time = 0
            
            for (output, expected_output) in zip(outputs, expected_outputs):
                time1 = time.perf_counter()
                costs.append(self.loss_function.forward(output, expected_output))
                time2 = time.perf_counter()
                total_time += time2 - time1
                node_values.append(self.loss_function.backward(output, expected_output))

            cost = tf.reduce_mean(tf.stack(costs), axis=0).numpy().astype(self.dtype)
            node_values = tf.cast(tf.stack(node_values), self.dtype)

        gradients = [None] * (len(self.model) - 1)

        with tf.device('/GPU:1'):
            for idx, layer in enumerate(self.model[::-1][:-1]):
                print(idx / len(self.model) * 100)
                current_layer_index = -(idx + 1)

                input_activations = self.model[current_layer_index - 1].output_activations

                node_values, gradient = layer.backward(input_activations, node_values)

                gradients[current_layer_index] = gradient

                del input_activations, layer.output_activations, gradient

            del expected_outputs, node_values

        return gradients, cost

    def _sum_gradients(self, gradients):
        summed_array = gradients[0]
        gradients = gradients[1:]

        for gradient in gradients:
            for idx, layer in enumerate(gradient):
                if isinstance(layer, cp.ndarray):
                    summed_array[idx] += layer
                else:
                    summed_array[idx] = self._sum_gradients([gradient[idx] for gradient in gradients])

        return summed_array

    def _scale_gradient(self, gradient):
        for idx, layer in enumerate(gradient):
            if isinstance(layer, cp.ndarray):
                if layer.size == 0:
                    continue
                gradient[idx] /= self.batch_size
            else:
                gradient[idx] = self._scale_gradient(layer)

        return gradient

    def _clip_gradient(self, gradient, clip_norm):
        for idx, layer in enumerate(gradient):
            if isinstance(layer, cp.ndarray):
                if layer.size == 0:
                    continue
                gradient[idx] = tf.clip_by_norm(gradient[idx] + 1.0e-9, clip_norm)
            else:
                gradient[idx] = self._scale_gradient(layer)

        return gradient

    def fit(self, xdata=None, ydata=None, generator=None, batch_size=8, learning_rate=0.01, epochs=100, clip_norm=None):

        if not generator:
            xdata = np.array(xdata, dtype=self.dtype)
            ydata = np.array(ydata, dtype=self.dtype)

            dataset_size = xdata.shape[0]

        else:
            dataset_size = generator.dataset_size

        iterations = int(epochs * (dataset_size / batch_size))
        optimizer_values = [None] * len(self.model)

        self.batch_size = batch_size

        cost = None
        delay = None

        for iteration in range(iterations):
            start_time = time.perf_counter()

            epoch = (batch_size / dataset_size) * iteration

            if self.scheduler:
                learning_rate = self.scheduler.forward(learning_rate, epoch)

            if not generator:
                choices = np.random.choice(xdata.shape[0], size=batch_size, replace=False)
            
                selected_xdata = xdata[choices]
                selected_ydata = ydata[choices]

            else:
                selected_xdata, selected_ydata = generator.get()
                selected_xdata = selected_xdata.astype(self.dtype)
                selected_ydata = selected_ydata.astype(self.dtype)

            gradient = None
            cost = 0

            time1 = time.perf_counter()
            model_output = self.forward(selected_xdata)
            gradient, cost = self.backward(model_output, selected_ydata)
            time2 = time.perf_counter()

            print(time2 - time1, cost)

            if clip_norm:
                gradient = self._clip_gradient(gradient, clip_norm)

            for idx, (layer, layer_gradient, descent_values) in enumerate(zip(self.model[1:], gradient, optimizer_values)):
                new_descent_values = layer.update(self.optimizer, layer_gradient, descent_values, learning_rate)

                # self.optimizer_values[idx] = new_descent_values
                del new_descent_values

            end_time = time.perf_counter()
            delay = end_time - start_time

            now = datetime.now()
            date_string = now.strftime("%H:%M:%S")

            print(f"[LOG {date_string}] Iteration: {iteration}, Epoch: {epoch}, Delay: {delay}, Learning Rate: {learning_rate}")
            end_time = time.perf_counter()

            yield cost