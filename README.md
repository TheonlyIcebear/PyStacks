# Neural-Net-Framework
I custom library I made for training neural networks from scratch, using numpy and scipy



# Imformation ℹ

I've been working on many neural network projects for the past year now, with my earliest project finished in April 2024

And I've learned a lot through much trial and error in the Machine Learning field and my end goal is to create a boudning box regression model similiar to `yolo-v1`.

### Purpose:

 - I noticed my old code was really limiting and innefecient:
 
   - https://github.com/TheonlyIcebear/Image-Recognition-AI
   - https://github.com/TheonlyIcebear/Tic-Tac-Toe-Machine-Learnin

 - I decided to do a complete rewrite allowing my code to be more modular much like keras' `Sequential` class allowing me to quickly add features like `Batch Normalization`


Image classification problem training loss:<br>
![image](https://github.com/user-attachments/assets/b28288b5-8901-4b4f-8299-c2c0b0dfaff8)

## ✨ Features

- **Fully Modular Layer System:**  
  Compose models with custom layers, CSP blocks, SPPF, residuals, and more—just like building with Keras or PyTorch, but with full control.

- **YOLO-Style Object Detection:**  
  Out-of-the-box support for multi-scale, anchor-based detection heads, custom loss functions (CIoU, SIoU, Focal, BCE), and advanced augmentation.

- **Custom Training Loop:**  
  Fine-grained control over batching, accumulation, learning rate scheduling, and gradient clipping.

- **GPU Acceleration:**  
  Uses TensorFlow (v1 compat) and CuPy for fast computation, with manual memory management for large models.

- **Visualization & Monitoring:**  
  Real-time loss plotting across multiple detection heads and loss components.

- **Data Augmentation:**  
  Powerful augmentations via **Albumentations**: flips, color shifts, affine, perspective, grid distortion, and more.

- **Anchor Box Clustering:**  
  K-means and IoU-based anchor clustering for optimal detection performance.

- **Easy Saving/Loading:**  
  Save and resume training, including optimizer state, scheduler, and loss history.

---

## 🏗️ Example: YOLO-Style Model Construction

```python
model = [
    Input((512, 512, 3)),
    *conv(64, (6, 6), stride=2),
    *conv(128, (3, 3), stride=2),
    *csp_block(64, 3),
    *conv(128, (1, 1)),
    # ... more layers ...
]
```

# Notes

 - This is **NOT** a replacement for actual Neural Network libraries such as `Tensorflow` or `PyTorch`, this is simply a library i made because I want to understand neural networks on a deep level.

- If you value your time, please don't use this on serious projects lol

 - For some reason (aka im too dumb to fix it) I cant get the custom striding to work but I decided to leave it in anyways

 - This is CPU bound only meaning it is really slow compared to other GPU based models, I plan to add modules such as `CuPy` once I get a Nividia GPU so I can use and test CUDA, but for now multiprocessing it is.

# Sources 🔌

 - https://www.youtube.com/watch?v=Lakz2MoHy6o&t=337s

 - https://www.youtube.com/watch?v=pj9-rr1wDhM

 - https://optimization.cbe.cornell.edu/index.php?title=Adam

 - https://paperswithcode.com/method/he-initialization#:~:text=Kaiming%20Initialization%2C%20or%20He%20Initialization,magnitudes%20of%20input%20signals%20exponentially.

 - https://builtin.com/machine-learning/adam-optimization

 - https://www.youtube.com/watch?v=Lakz2MoHy6o&t=337s
 
 - https://dev.to/afrozchakure/all-you-need-to-know-about-yolo-v3-you-only-look-once-e4m 