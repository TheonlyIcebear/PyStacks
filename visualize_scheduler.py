import numpy as np
import matplotlib.pyplot as plt
from utils.schedulers import CosineAnnealingDecay, ExponentialDecay, InverseTimeDecay, StepLR

cosine = CosineAnnealingDecay(initial_learning_rate=1.0, min_learning_rate=0, initial_cycle_size=50, cycle_mult=2)
exponential = ExponentialDecay(initial_learning_rate=1.0, decay_rate=0.995)
inverse = InverseTimeDecay(initial_learning_rate=1.0, decay_rate=0.7, decay_interval=100)
step = StepLR(initial_learning_rate=1.0, decay_rate=0.5, decay_interval=100)

x = np.linspace(0, 1000, 1000)
y = []

for scheduler in [cosine, exponential, inverse, step]:
    y.append([scheduler.forward(point) for point in x])

y = np.array(y).T

plt.plot(x, y, label=('CosineAnnealingDecay', 'ExponentialDecay', 'InverseTimeDecay', 'StepLR'))
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Learning Rate')
plt.show()