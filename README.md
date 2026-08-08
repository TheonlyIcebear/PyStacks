# PyStacks

A modular deep learning framework built from scratch using low-level **TensorFlow** primitives and **CuPy**. 

Rather than relying on high-level APIs (`nn.Module`, `tf.keras.Model`, or automatic differentiation wrappers), **PyStacks** explicitly handles **forward execution, gradient tracking, stack-based graph routing, custom layer backpropagation, and flexible loss functions** across both vision classification and multi-scale object detection tasks.

---

## 💡 Motivation & Overview

After building earlier machine learning projects (such as my [Image Recognition AI](https://github.com/TheonlyIcebear/Image-Recognition-AI) and [Tic-Tac-Toe ML](https://github.com/TheonlyIcebear/Tic-Tac-Toe-Machine-Learning)), I ran into performance bottlenecks and structural rigidity from hardcoded models. 

To solve this, I designed **PyStacks** as a complete, ground-up rewrite. Inspired by sequential and DAG-style framework semantics, PyStacks allows assembling modular network topographies: ranging from standard MLPs and CNN classifiers to complex **C2f bottlenecks, SPPF blocks, and anchor-free detection heads**, while retaining manual control over tensor movement, memory allocation, and gradient propagation.

<img width="1598" height="664" alt="image" src="https://github.com/user-attachments/assets/134feffa-6e68-4005-99b8-364fe7c5c740" />

---

## 🔨 Development & Engineering Challenges

Building a deep learning framework from scratch meant encountering complex optimization traps and mathematical edge cases:

* **Noisy Datasets, Overfitting, and Convergence Traps:** During initial model training runs, I faced issues with noisy ground truth data, early overfitting, and unexpected loss divergence. Resolving these required introducing custom data augmentation pipelines (Mosaic, color shifts, random scale cropping) and tightening model capacity.

* **The "Vanishing Objectness" Trap in YOLOv5:** The hardest optimization issue occurred when training an early anchor-based YOLOv5 model, where objectness loss steadily increased instead of decreasing. Objectness predicts whether a grid cell contains a valid bounding box. Early in training, when model predictions were naturally poor, the optimizer quickly learned to predict near-zero confidence scores across all cells. As regression quality improved, the high momentum memory in Adam ($0.99$) retained too much historical signal from those early bad steps, overpowering new positive feedback and forcing the model to keep predicting low confidence scores.
  * **Solution:** Reduced the Adam momentum hyperparameters to around $0.95$ (effectively shortening the optimizer's memory window to ~20 batches) and introduced learning rate warmup schedules.

* **Manual Backpropagation & Softmax Jacobian Calculations:** Writing manual backward passes for activation functions without automatic differentiation required careful mathematical handling. Standard element-wise activations (like ReLU or LeakyReLU) have diagonal derivatives, but **Softmax** produces an output vector where every output depends on every input. Computing its backward pass requires evaluating the full **Jacobian matrix**:
  $$\frac{\partial S_i}{\partial z_j} = S_i (\delta_{ij} - S_j)$$
  Correctly deriving and vectorizing this derivative in code (to avoid explicitly constructing massive 4D batch-wise Jacobian matrices) was essential for numerical stability and backward pass speed in multi-class classification and distribution losses.

---

## 🛠️ System Control Flow & Architecture

### 1. Tag-Based Stack Graph Engine
Routing tensors through skip connections or dynamic branches without relying on implicit `autograd` graphs requires careful activation tracking. Naive approaches clone intermediate tensors, which causes severe memory bloat.

PyStacks uses a LIFO **tag-based stack routing mechanism** (`Concat.Start`, `Residual.Start`, `Concat.End`):

```
              ┌─── [Branch 1: Channel Operations / Conv] ───┐
[Input Tensor]┤                                             ├─► [Concat.End] ─► [Output]
              └─── [Branch 2: Direct Skip Connection] ──────┘
                                 │
                    (Pushed to Activation Stack)
```

* **Forward Pass:** Hitting a `Concat.Start` or `Residual.Start` marker pushes the current tensor onto an execution stack. When reaching `Concat.End` or `Concat.ResidualEnd`, the framework pops the saved tensor and concatenates or adds it along the targeted dimension (`axis=-1`).
* **Backward Pass:** During backpropagation, `Concat.End` splits incoming gradients using saved tensor shape metadata. `Concat.Start` pops the corresponding route gradient and accumulates it with the main branch gradient (`prev_grad + route_grad`).

### 2. End-to-End Training Execution Loop

```
┌────────────────────────┐      ┌────────────────────────┐      ┌────────────────────────┐
│  Multi-Process Prefetch │─────►│  Forward Execution     ├─────►│  Loss Engine           │
│  (Custom Data / Disk)  │      │  (Tag Stack Tracking)  │      │  (BCE / TAL / DFL / CIoU)│
└────────────────────────┘      └────────────────────────┘      └───────────┬────────────┘
                                                                            │
┌────────────────────────┐      ┌────────────────────────┐                  │
│ Optimizer Update Step  │◄─────┤ Manual Backward Pass   │◄─────────────────┘
│ (Grad Scale / AutoClip)│      │ (Stack Gradient Split) │    Calculates Gradients
└────────────────────────┘      └────────────────────────┘    Across Graph Tags
```

---

## ⚙️ Key Engine Features

### General-Purpose Layer & Topology Engine
* **Modular Block Composition:** Layers inherit from a unified `Layer` interface, executing explicit `forward()`, `backward()`, `initialize()`, and `update()` passes.
* **Custom Conv & Pooling Primitives:** Features manual implementations of spatial `Conv2d`, `MaxPooling`, `Dropout`, and dense transformations.
* **Flexible Task Support:** Supports diverse loss pipelines ranging from multi-class categorical cross-entropy for image classification to complex **Task-Aligned Assignment (TAL)** and **Distribution Focal Loss (DFL)** for object detection.

### Memory & Performance Optimization
* **Mixed-Precision Memory Management:** Operations like convolutions and matrix multiplications execute in `float16` for execution speed and reduced VRAM footprint, while **Batch Normalization** running statistics (`running_mean`, `running_var`) and numerical loss reductions stay in `float32` to prevent underflow.
* **Adaptive Gradient Clipping (AutoClip):** Monitors historical gradient norms across training steps to dynamically suppress instability from exploding gradients.
* **Multi-Process Data Pipeline:** Prefetches incoming batches using background worker processes (`multiprocessing.Process`) with local queue buffers to reduce GPU starvation during disk I/O and augmentation steps.

---

## 🏗️ Code Examples

### 1. Image Classifier Architecture (`classify.py`)
```python
import numpy as np
from utils.layers import Input, Conv2d, Dense, Activation, MaxPooling
from utils.activations import LeakyReLU, Softmax
from utils.network import Network
from utils.optimizers import Adam
from utils.schedulers import StepLR
from utils.loss import BCE

# Modular sequential list setup
model = [
    Input((128, 128, 3)),
    Conv2d(32, kernel_size=(3, 3), padding="SAME"),
    Activation(LeakyReLU()),
    MaxPooling(pooling_shape=(2, 2), pooling_stride=(2, 2)),
    
    Dense(64),
    Activation(LeakyReLU()),
    Dense(10),
    Activation(Softmax())
]

network = Network(
    model=model,
    loss_function=BCE,
    optimizer=Adam(momentum=0.99, beta_constant=0.999),
    scheduler=StepLR(initial_learning_rate=1e-4, decay_rate=0.5, decay_interval=5),
    dtype=np.float16
)

network.compile()
```

### 2. Complex Graph Routing with Concat Markers
```python
from utils.layers import Conv2d, BatchNormalization, SiLU, Concat

# Building custom CSP-style skip blocks using stack tags
def csp_block(in_channels, out_channels):
    concat_start, concat_end = Concat.Start(), Concat.End()
    return [
        concat_start,
        Conv2d(out_channels // 2, kernel_size=(1, 1), padding="SAME"),
        Conv2d(out_channels // 2, kernel_size=(3, 3), padding="SAME"),
        BatchNormalization(),
        SiLU(),
        concat_end,
        Conv2d(out_channels, kernel_size=(1, 1), padding="SAME"),
        BatchNormalization(),
        SiLU(),
    ]
```

---

## 📌 Disclaimer

This framework was developed purely as an educational deep-dive into execution engines, memory allocation, CUDA operations, and manual backpropagation mechanics. It is designed to demonstrate deep learning fundamentals rather than replace production frameworks like PyTorch or TensorFlow.

---

## 🔌 References & Resources
- [AutoClip: Adaptive Gradient Clipping](https://github.com/pseeth/autoclip)
- [Adam Optimization Architecture & Hyperparameters](https://optimization.cbe.cornell.edu/index.php?title=Adam)
- [He/Kaiming Weight Initialization](https://paperswithcode.com/method/he-initialization)
