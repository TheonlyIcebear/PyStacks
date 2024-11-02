# PyStacks
A CUDA based modular neural network library

# Information ℹ

Likely my final Neural Network repository, I've been working on this neural netork projects for a couple months now. But I've limited by the fact I had a AMD card, so I was stuck to slow CPU based training. 

Now that I have a much faster Nividia card I've modified my previous NumPy and SciPy based [library](https://github.com/TheonlyIcebear/Neural-Net-Framework) to work with tensorflow and cupy. And it's over 100x faster


# Object Detection 🤖
I stated before that my goal was to create a object detection model based off of a YOLO type architecture. So far I've been able to replicate Yolo V1 to Yolo V3 with somewhat accurate results

Heres a couple showcases:
 - https://www.youtube.com/watch?v=uV8g4THpF6g
 - https://www.youtube.com/watch?v=15d-3FqNH-g

![image](https://github.com/user-attachments/assets/7efdd2a7-1c00-428b-b568-7378fd7805bc)


# Updates 📰

My previous model was purely sequential, meaning the output from the previous layer would only go straight into the next layer. I've now added both a `ResidualBlock` and a `ConcatBlock` layer which can skip connections. This is very important for feature pyramid architectures.

I've also fixed a couple errors and misconceptions in my code.

I've also added new activation functions, loss functions and schedulers.

The code is also much more optimized, apart from being GPU based.

I plan to add proper documentation soon, but for now just look at the examples

# Sources 🔌

 - https://www.youtube.com/watch?v=Lakz2MoHy6o&t=337s
 - https://www.youtube.com/watch?v=pj9-rr1wDhM
 - https://optimization.cbe.cornell.edu/index.php?title=Adam
 - https://paperswithcode.com/method/he-initialization#:~:text=Kaiming%20Initialization%2C%20or%20He%20Initialization,magnitudes%20of%20input%20signals%20exponentially.
 - https://builtin.com/machine-learning/adam-optimization
 - https://www.youtube.com/watch?v=Lakz2MoHy6o&t=337s
 - https://towardsdatascience.com/yolo-v3-explained-ff5b850390f
