from utils.layers import *
from utils.network import Network
from utils.optimizers import Adam
from utils.loss import *
from PIL import Image
from tqdm import tqdm
import albumentations as A, matplotlib.pyplot as plt
import requests, random, numpy as np, pickle, os

class Main:
    def __init__(self):
        save_data = pickle.load(open('model-training-data.json', 'rb'))

        network = Network(loss_function="cross_entropy")
        network.load(save_data)
        # network.compile()

        self.image_height, self.image_width = network.model[0].output_shape[:-1]
        self.network = network

    def test(self):
        while True:
            options = ['airplane', 'bicycle', 'boat', 'motorbus', 'motorcycle', 'seaplane', 'train', 'truck', 'Custom File']

            while True:
                for idx, option in enumerate(options):
                    print(f"{idx + 1}. {option.capitalize()}")

                print('\n')

                print("Enter a integer 1-9")
                choice = int(input(">> "))
                
                if choice == 9:
                    print("Enter Image Url: ")
                    url = input(">> ")
                    image = Image.open(requests.get(url, stream=True).raw).resize((self.image_width, self.image_height))
                else:

                    folder = options[choice - 1]

                    folder = f'images\\{folder}'
                    
                    filename = random.choice(os.listdir(folder))
                    image = Image.open(f'{folder}\\{filename}').resize((self.image_width, self.image_height))

                image.show()

                input_data = np.asarray(image).astype(np.float32) / 255

                print(input_data.shape)

                output = self.network.forward(input_data[None, ...], training=False)[0][0]

                answer = int(np.argmax(output))
                print(f"It's a {options[answer]}")
                print(output, '\n\n')

if __name__ == "__main__":
    main = Main()
    main.test()