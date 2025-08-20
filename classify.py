from utils.layers import *
from utils.network import Network
from utils.optimizers import Adam
from utils.activations import *
from utils.schedulers import *
from utils.functions import ClipGradient, AutoClipper
from utils.loss import *
from PIL import Image
from tqdm import tqdm
import albumentations as A, matplotlib.pyplot as plt
import  threading, numpy as np, pickle, os

class Generate:
    def __init__(self):
        self.folders = os.listdir(f'images')
        self.dataset_size = sum([len(os.listdir(f'images\\{folder}')) for folder in self.folders])

    def __call__(self):
        folder_xdata = []
        folder_ydata = []

        for _ in range(batch_size):
            choice = np.random.randint(0, len(self.folders))

            folder = self.folders[choice]
            folder = f'images\\{folder}'

            filename = np.random.choice(os.listdir(folder))

            image = Image.open(f'{folder}\\{filename}').resize((image_width, image_height)).convert('RGB')

            input_data = np.asarray(image) / 255

            expected_output = np.zeros(len(self.folders))
            expected_output[choice] = 1

            folder_xdata.append(input_data)
            folder_ydata.append([expected_output])

        folder_xdata = np.array(folder_xdata)
        folder_ydata = np.array(folder_ydata)

        return folder_xdata, folder_ydata

def preprocess_data():
    xdata = np.empty((0, image_height, image_width, 3))
    ydata = np.empty((0, 1, 11))

    folders = os.listdir(folder)

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
            folder_ydata.append([expected_output])

        folder_xdata = np.array(folder_xdata)
        folder_ydata = np.array(folder_ydata)

        repeats = largest_folder // folder_xdata.shape[0]
        remainder = largest_folder % folder_xdata.shape[0]

        folder_xdata = np.concatenate((np.tile(folder_xdata, (repeats, 1, 1, 1)), folder_xdata[:remainder]))
        folder_ydata = np.concatenate((np.tile(folder_ydata, (repeats, 1, 1)), folder_ydata[:remainder]))

        xdata = np.concatenate((xdata, folder_xdata))
        ydata = np.concatenate((ydata, folder_ydata))

    return np.array(xdata), np.array(ydata)

def save():
    save_data = network1.save()

    with open(save_file, 'wb') as file:
        file.write(pickle.dumps(save_data))

if __name__ == "__main__":
    training_percent = 1
    batch_size = 16
    image_width, image_height = [128, 128]

    activation_function = Silu()

    model = [
        Input((image_height, image_width, 3)),

        Conv2d(16, (3, 3), padding='SAME'),
        BatchNorm(),
        Activation(activation_function),

        ConcatBlock([

            MaxPool((2,2)),
            
            Conv2d(32, (3, 3), padding='SAME'),
            BatchNorm(),
            Activation(activation_function),

            MaxPool((2,2)),
            
            Conv2d(64, (3, 3), padding='SAME'),
            BatchNorm(),
            Activation(activation_function),

        ], [
            Space2Depth(4)
        ]),

        Conv2d(64, (3, 3), stride=[2,2], padding='SAME'),
        BatchNorm(),
        Activation(activation_function),

        
        Flatten(),

        Dense(256),
        Activation(activation_function),
        
        Dense(128),
        Activation(activation_function),

        Dense(64),
        Activation(activation_function),

        Dense(32),
        Activation(activation_function),

        Dense(11),
        Activation(Softmax())
    ]
    
    network1 = Network(model, dtype=np.float16, loss_function=BCE, optimizer=Adam(momentum = 0.99, beta_constant = 0.999), scheduler=StepLR(initial_learning_rate=0.001, decay_rate=0.5, decay_interval=5))
    network1.compile()

    network2 = Network(model, dtype=np.float16, loss_function=BCE, optimizer=Adam(momentum = 0.9, beta_constant = 0.99), scheduler=StepLR(initial_learning_rate=0.00001, decay_rate=0.5, decay_interval=5))
    network2.compile()

    save_file = 'model-training-data.json'

    generator = Generate()

    costs = []
    val_costs = []
    plt.ion()

    for idx, combined_costs in enumerate(zip(
                network1.fit(generator=generator, batch_size = batch_size, epochs = 200, gradient_transformer=AutoClipper(10)), 
                network1.fit(generator=generator, batch_size = batch_size, epochs = 200),
            )
        ):
        print(combined_costs)
        costs.append(sum(combined_costs, []))

        threading.Thread(target=save).start()
        plt.plot(np.arange(len(costs)) * (batch_size / generator.dataset_size), costs, label=['Autoclip', 'No Autoclip'])

        plt.legend()
        plt.draw()
        plt.pause(0.1)
        plt.clf()