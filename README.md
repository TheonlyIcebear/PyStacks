# Neural-Net-Framework
A custom library I made for training neural networks from scratch, using Tensorflow and CuPy

# Information ℹ
I've been working on neural network projects for the past year, with my earliest project finished in April 2024. I've learned a lot through trial and error and my end goal was to create a bounding box regression model similar to `yolo-v5`.

### Why I made this:
- My old code was really limiting and inefficient:
  - https://github.com/TheonlyIcebear/Image-Recognition-AI
  - https://github.com/TheonlyIcebear/Tic-Tac-Toe-Machine-Learning
- So I did a complete rewrite making everything modular, similar to Keras' `Sequential` class, which let me quickly build up features like `Batch Normalization`, `CSP blocks`, and multi-scale detection heads without rewriting everything each time

YOLO-v5 Object Detection training loss:<br>
<img width="1599" height="667" alt="image" src="https://github.com/user-attachments/assets/2858dd63-9c60-4295-a189-b2be1face738" />

## ✨ Features
- **Fully Modular Layer System:**
  Build models by composing layers exactly like Keras or PyTorch, but every forward pass, backward pass, and weight update is written manually — no autograd shortcuts

- **YOLO-Style Object Detection:**
  Multi-scale anchor-based detection with custom loss functions (DIoU, CIoU, SIoU, Focal Loss, BCE) and a contraction loss on inactive anchors to stop bounding boxes from exploding during early training

- **Concat Graph Optimization:**
  Skip connections use a `ConcatStartPoint / ConcatResidualStartPoint / ConcatEndPoint` system that avoids storing redundant intermediate activations — I came up with this myself after the naive implementation was eating too much memory

- **Custom Training Loop:**
  Gradient accumulation, AutoClipper (adaptive gradient clipping from the original paper), learning rate schedulers, and full optimizer state saving/resuming

- **GPU Acceleration:**
  TensorFlow (v1 compat) and CuPy for GPU computation with manual memory management

- **Data Pipeline:**
  Multi-process prefetching with a local buffer to keep the GPU fed, plus Albumentations for augmentation (color shifts, random scaled crops, etc.)

- **Anchor Box Clustering:**
  IoU-based k-means clustering to generate anchors fitted to the actual training data distribution, same approach as the original YOLO papers

- **Visualization & Monitoring:**
  Real-time loss plotting across all 3 detection heads and all 5 loss components while training

- **Easy Saving/Loading:**
  Saves and resumes full training state including weights, BatchNorm running stats, optimizer momentum, scheduler, and loss history. Also exports to ONNX via `tf2onnx`

---

## 🏗️ Example: YOLO-Style Model Construction
```python
model = [
    Input((384, 384, 3)),
    *conv(64, (6, 6), stride=2),
    *conv(128, (3, 3), stride=2),
    *csp_block(64, 3),
    *conv(128, (1, 1)),
    # ... more layers ...
]

network = Network(
    model=model,
    addon_layers=addon_layers,
    loss_function=YoloLoss(...),
    optimizer=Adam(momentum=0.9, weight_decay=2e-5),
    scheduler=StepLR(initial_learning_rate=0.0001, decay_rate=0.6, decay_interval=50),
    dtype=np.float16
)

for cost in network.fit(generator=generator, batch_size=32, epochs=100):
    print(cost)
```

# Notes
- This is **NOT** a replacement for actual neural network libraries like `TensorFlow` or `PyTorch` — I made this because I wanted to understand what's actually happening inside the black box
- If you value your time, please don't use this on serious projects lol
- BatchNorm intentionally stays in `float32` even when the rest of the network runs `float16`, for numerical stability

# Sources 🔌
- https://www.youtube.com/watch?v=Lakz2MoHy6o&t=337s
- https://www.youtube.com/watch?v=pj9-rr1wDhM
- https://optimization.cbe.cornell.edu/index.php?title=Adam
- https://paperswithcode.com/method/he-initialization
- https://builtin.com/machine-learning/adam-optimization
- https://dev.to/afrozchakure/all-you-need-to-know-about-yolo-v3-you-only-look-once-e4m
- https://github.com/pseeth/autoclip
