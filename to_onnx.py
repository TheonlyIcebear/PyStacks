from utils.layers import *
from utils.schedulers import *
from utils.network import Network
from utils.initializers import *
from utils.loss import *
from utils.activations import *
import tf2onnx

import multiprocessing, threading, collections, numpy as np, pickle, time, json, cv2, os

import onnxruntime as ort
import numpy as np
import time
import tf2onnx
import onnx
from onnxsim import simplify

network = Network(dtype=tf.float16)
network.load(pickle.load(open('model-training-data.json', 'rb')))

image_width, image_height = network.model[0].output_shape[0:2]

print(image_width, image_height)

spec = tf.TensorSpec([1, image_height, image_width, 3], tf.float16)


graphed_forward = tf.function(lambda x: network.forward(x, training=False), input_signature=[spec])

onnx_model, _ = tf2onnx.convert.from_function(
    graphed_forward,
    input_signature=[spec],
    opset=17,
    output_path="model-training-data.onnx",
    inputs_as_nchw=["input:0"]
)

# Convert model to onnx format
model = onnx.load("model-training-data.onnx")


model_simplified, check = simplify(model)

if not check:
    raise RuntimeError("Simplified ONNX model could not be validated")

onnx.save(model_simplified, "model-training-data.onnx")


so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

session = ort.InferenceSession(
    "model-training-data.onnx",
    sess_options=so,
    providers=["DmlExecutionProvider", "CPUExecutionProvider"]
)

so.intra_op_num_threads = 4

input_name = session.get_inputs()[0].name
dummy = np.random.randn(1, image_height, image_width, 3).astype(np.float16)

# Warm up the model

for _ in range(50):
    session.run(None, {
        input_name: dummy,
    })


runs = 300
start = time.time()

times = []

# Test model performance

for _ in range(runs):
    start = time.perf_counter()
    session.run(None, {
        input_name: dummy,
    })
    times.append(time.perf_counter() - start)

print("Mean latency (ms):", np.mean(times) * 1000)
print("P95 latency (ms):", np.percentile(times, 95) * 1000)
print("FPS:", 1 / np.mean(times))