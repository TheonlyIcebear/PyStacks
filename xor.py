from utils.layers import *
from utils.schedulers import *
from utils.network import Network
from utils.optimizers import Adam
from utils.activations import *
from utils.loss import *
import matplotlib.pyplot as plt
import numpy as np, pickle, time

if __name__ == "__main__":
    model = [
        Input(2),
        Dense(3),
        Activation(Mish()),
        Dense(2),
        Activation(Sigmoid()),
    ]

    print(model)

    network = Network(model, loss_function=BCE(), optimizer=Adam(momentum = 0.7, beta_constant = 0.8))
    network.compile()
    
    training_percent = 1
    batch_size = 4

    save_file = 'model-training-data.json'

    xdata = [[i % 2, i // 2] for i in range(4)]
    ydata = [[[(i % 2) ^ (i // 2), 1 - ((i % 2) ^ (i // 2))]] for i in range(4)]


    costs = []
    plt.ion()

    for idx, cost in enumerate(network.fit(xdata, ydata, learning_rate=0.01, batch_size = batch_size, epochs = 1000)):
        costs.append(cost)

        if not idx % 10:
            plt.plot(np.arange(len(costs)) * (batch_size / (len(xdata) * training_percent)), costs, label='training')

            plt.legend()
            plt.draw()
            plt.pause(0.001)
            plt.clf()