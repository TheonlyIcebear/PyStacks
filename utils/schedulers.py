import numpy as np, cupy as cp

class StepLR:
    def __init__(self, decay_rate, decay_interval=1):
        self.decay_rate = decay_rate
        self.decay_interval = decay_interval
        self.target = decay_interval

    def forward(self, learning_rate, epoch):
        if epoch > self.target:
            learning_rate *= self.decay_rate
            self.target += self.decay_interval
            
        return learning_rate 

class ExponentialDecay:
    def __init__(self, learning_rate, decay_rate):
        self.decay_rate = decay_rate
        self.initial_learning_rate = learning_rate

    def forward(self, learning_rate, epoch):
        return self.initial_learning_rate * (self.decay_rate ** epoch)

class InverseTimeDecay:
    def __init__(self, learning_rate, decay_rate, decay_interval=1):
        self.decay_rate = decay_rate 
        self.decay_interval = decay_interval
        self.initial_learning_rate = learning_rate

    def forward(self, learning_rate, epoch):
        return self.initial_learning_rate / (1 + self.decay_rate * (epoch // self.decay_interval))

class CosineAnnealingDecay:
    def __init__(self, initial_learning_rate, min_learning_rate, initial_max_epochs, cycle_mult=1):
        self.initial_learning_rate = initial_learning_rate
        self.min_learning_rate = min_learning_rate
        self.initial_max_epochs = initial_max_epochs
        self.cycle_mult = cycle_mult
        self.current_cycle = 0

    def forward(self, learning_rate, epoch):
        cycle_lengths = self.initial_max_epochs * (self.cycle_mult ** np.arange(self.current_cycle))
        cycle_position = epoch - np.sum(cycle_lengths)

        if cycle_position >= self.initial_max_epochs * (self.cycle_mult ** self.current_cycle):
            self.current_cycle += 1
            cycle_position = 0

        current_cycle_epochs = self.initial_max_epochs * (self.cycle_mult ** self.current_cycle)

        learning_rate = self.min_learning_rate + (self.initial_learning_rate - self.min_learning_rate) * (1 + np.cos(np.pi * cycle_position / current_cycle_epochs)) / 2

        return learning_rate