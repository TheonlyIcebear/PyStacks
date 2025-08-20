import numpy as np, cupy as cp

class StepLR:
    def __init__(self, initial_learning_rate, decay_rate, decay_interval=1):
        self.initial_learning_rate = initial_learning_rate
        self.decay_interval = decay_interval
        self.decay_rate = decay_rate
        self.target = decay_interval

    def __call__(self, epoch):
        return self.initial_learning_rate * (self.decay_rate ** (epoch // self.decay_interval))

class ExponentialDecay:
    def __init__(self, initial_learning_rate, decay_rate):
        self.initial_learning_rate = initial_learning_rate
        self.decay_rate = decay_rate

    def __call__(self, epoch):
        return self.initial_learning_rate * (self.decay_rate ** epoch)

class InverseTimeDecay:
    def __init__(self, initial_learning_rate, decay_rate, decay_interval=1):
        self.initial_learning_rate = initial_learning_rate
        self.decay_interval = decay_interval
        self.decay_rate = decay_rate 
        
    def __call__(self, epoch):
        return self.initial_learning_rate / (1 + self.decay_rate * (epoch // self.decay_interval))

class CosineAnnealingDecay:
    def __init__(self, initial_learning_rate, min_learning_rate, initial_cycle_size, cycle_mult=1):
        self.initial_learning_rate = initial_learning_rate
        self.min_learning_rate = min_learning_rate
        self.initial_cycle_size = initial_cycle_size
        self.cycle_mult = cycle_mult
        self.current_cycle = 0

    def __call__(self, epoch):
        total_epochs_in_completed_cycles = 0
        current_cycle_epochs = self.initial_cycle_size
        
        while epoch >= total_epochs_in_completed_cycles + current_cycle_epochs:
            total_epochs_in_completed_cycles += current_cycle_epochs
            current_cycle_epochs *= self.cycle_mult
            self.current_cycle += 1
        
        cycle_position = epoch - total_epochs_in_completed_cycles
        learning_rate = self.min_learning_rate + (self.initial_learning_rate - self.min_learning_rate) * (1 + np.cos(np.pi * cycle_position / current_cycle_epochs)) / 2

        return learning_rate