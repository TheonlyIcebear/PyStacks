from utils.layers import *
from utils.network import Network
from utils.optimizers import Adam
from utils.schedulers import *
from utils.activations import *
from utils.loss import *
from PIL import Image
from tqdm import tqdm
import albumentations as A, matplotlib.pyplot as plt
import  threading, numpy as np, pickle, os

def preprocess_data():
    xdata = np.empty((0, image_height, image_width, 3))
    ydata = np.empty((0, 8))

    folders = ['airplane', 'bicycle', 'boat', 'motorbus', 'motorcycle', 'seaplane', 'train', 'truck']

    largest_folder = max([len(os.listdir(f'images\\{folder}')) for folder in folders])

    print(largest_folder)

    for idx, folder in enumerate(tqdm(folders)):
        folder = f'images\\{folder}'
        folder_xdata = []
        folder_ydata = []
        for filename in os.listdir(folder):
            image = Image.open(f'{folder}\\{filename}').resize((image_width, image_height)).convert('RGB')

            input_data = np.asarray(image) / 255

            expected_output = np.zeros(len(folders))
            expected_output[idx] = 1

            folder_xdata.append(input_data)
            folder_ydata.append(expected_output)

        folder_xdata = np.array(folder_xdata)
        folder_ydata = np.array(folder_ydata)

        repeats = largest_folder // folder_xdata.shape[0]
        remainder = largest_folder % folder_xdata.shape[0]

        folder_xdata = np.concatenate((np.tile(folder_xdata, (repeats, 1, 1, 1)), folder_xdata[:remainder]))
        folder_ydata = np.concatenate((np.tile(folder_ydata, (repeats, 1)), folder_ydata[:remainder]))

        print(folder_xdata.shape, folder_ydata.shape)

        xdata = np.concatenate((xdata, folder_xdata))
        ydata = np.concatenate((ydata, folder_ydata))

    return np.array(xdata), np.array(ydata)

def save():
    save_data = network.save()

    with open(save_file, 'wb') as file:
        file.write(pickle.dumps(save_data))

if __name__ == "__main__":
    training_percent = 1
    batch_size = 16
    image_width, image_height = [128, 128]

    lrelu = LRelu(0.1)

    model = [
        Input((image_height, image_width, 3)),

        Conv2d(6, (3, 3)),
        BatchNorm(),
        Activation(lrelu),
        
        MaxPool((2, 2)),

        Conv2d(12, (3, 3)),
        BatchNorm(),
        Activation(lrelu),
        
        MaxPool((2, 2)),

        Conv2d(18, (3, 3)),
        BatchNorm(),
        Activation(lrelu),
        
        MaxPool((2, 2)),

        Flatten(),

        Dense(256),
        Activation(lrelu),
        
        Dense(128),
        Activation(lrelu),

        Dense(64),
        Activation(lrelu),

        Dense(32),
        Activation(lrelu),

        Dense(8),
        Activation(Softmax())
    ]

    network = Network(model, loss_function=CrossEntropy(), optimizer=Adam(momentum = 0.8, beta_constant = 0.9), scheduler=StepLR(decay_rate=0.5, decay_interval=10))
    network.compile()

    save_file = 'model-training-data.json'

    xdata, ydata = preprocess_data()

    # choices = np.random.choice(xdata.shape[0], size=int(len(xdata) * training_percent), replace=False)

    # xdata = xdata[choices]
    # ydata = ydata[choices]

    # with open('training-files.json', 'w+') as file:
    #     file.write(json.dumps(choices.tolist()))

    costs = []
    val_costs = []
    plt.ion()

    for idx, cost in enumerate(network.fit(xdata, ydata, learning_rate=0.0001, batch_size = batch_size, epochs = 200)):
        # print(cost)

        threading.Thread(target=save).start()

        # choice = np.random.randint(len(xdata))

        # model_output = network.forward(xdata[choice], training=False)
        # print(model_output)
        # val_loss = network.loss_function(model_output, ydata[choice])

        # val_costs.append(val_loss)
        costs.append(cost)

        plt.plot(np.arange(len(costs)) * (batch_size / len(xdata)), costs, label='training')
        # plt.plot(np.arange(len(val_costs)) * (batch_size / (len(xdata) * training_percent)), val_costs, color='orange', label='validation')

        plt.legend()
        plt.draw()
        plt.pause(0.1)
        plt.clf()