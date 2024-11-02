from utils.layers import *
from utils.schedulers import *
from utils.network import Network
from utils.loss import *
from utils.activations import *
from utils.functions import Processing
from utils.optimizers import Adam, Momentum, SGD
from PIL import Image
from tqdm import tqdm
import matplotlib.animation as animation
import albumentations as A, matplotlib.pyplot as plt
import threading, numpy as np, pickle, time, json, cv2, os

class Generate:
    def __init__(self, batch_size, training_percent, queue_size):
        self.dataset_size = len(os.listdir('gameplay'))
        self.batch_size = batch_size

        self.choices = np.random.choice(self.dataset_size, size=int(self.dataset_size * training_percent), replace=False)
        self.data = np.array(os.listdir('gameplay'))
        self.queue_size = queue_size

        with open('training-files.json', 'w+') as file:
            file.write(json.dumps(self.choices.tolist()))

        self.xdata = []
        self.ydata = []
        self.val_xdata = []
        self.val_ydata = []

        threading.Thread(target=self.fill_queue).start()

    def fill_queue(self):
        count = 0
        while True:
            if len(self.xdata) >= self.queue_size:
                continue

            validation = count % 2

            if validation:
                mask = np.arange(self.dataset_size)
            
                mask[:] = True
                mask[self.choices] = False
            else:
                mask = self.choices

            filename = np.random.choice(self.data[mask])

            locations_filename = f'annotations\\{filename.replace(".png", ".txt")}'

            if os.path.exists(locations_filename):
                with open(locations_filename, "r+") as file:
                    lines = file.read().splitlines()
                    location_data = np.array([
                        [
                            np.float64(value) for value in line.split(' ')[1:]
                        ] for line in lines
                    ], dtype=np.float64)

                midpoints = location_data[:, 2] * location_data[:, 3]
                    
                location_data = location_data[midpoints.argsort()[::-1]][:objects]
                objects_present = location_data.shape[0]

            else:
                location_data = np.array([])
                objects_present = 0

            location_data += 10e-8

            image = Image.open(f'gameplay\\{filename}').resize((image_width, image_height))

            if location_data.shape[0] != 0:

                classes = ['enemy'] * len(location_data)

                augmenter = A.Compose([
                    A.HorizontalFlip(p=0.5),

                    A.ShiftScaleRotate(
                        shift_limit=0.0625, 
                        scale_limit=0.0, 
                        rotate_limit=10,
                        border_mode=cv2.BORDER_REPLICATE,
                        interpolation=1,
                        p=0.5
                    ),

                    A.RandomResizedCrop(
                        scale=(0.6, 0.9),
                        size=(image_height, image_width),
                        ratio=(1, 1),
                        p=0.5,
                    ),

                    A.RGBShift(p=0.45),
                    A.HueSaturationValue(hue_shift_limit=(-20, 20), sat_shift_limit=(-30, 30), val_shift_limit=(-20, 20), p=0.45),

                    A.RandomBrightnessContrast(p=0.25),
                    A.Emboss(
                        alpha=(0.2, 0.3),
                        strength=(0.1, 0.2),
                        p=0.25,
                    ),

                    A.CLAHE(p=0.1),
                ], bbox_params=A.BboxParams(format='yolo', min_visibility=0.25, label_fields=['class_labels']))

                augmented_result = augmenter(image=np.array(image), bboxes=location_data, class_labels=classes)

                image = augmented_result['image']
                bboxes = np.array(augmented_result['bboxes'])
                objects_present = bboxes.shape[0]
                bboxes = bboxes.reshape((int(objects_present), 4))

            else:
                imgae = np.asarray(image)
                bboxes = []

            input_data = image / 255

            cells = np.zeros((grid_size, grid_size, 5 * anchors))

            standing_ratio = 4.789523615911094

            for true_center_x, true_center_y, width, height in bboxes:
                grid_x_index = int(true_center_x * grid_size)
                grid_y_index = int(true_center_y * grid_size)

                relative_center_x = true_center_x * grid_size - grid_x_index
                relative_center_y = true_center_y * grid_size - grid_y_index

                height_ratio = height / width

                anchor_index = np.floor((height_ratio / standing_ratio) * anchors).astype(int)
                anchor_index = min(anchors-1, anchor_index)

                cells[grid_x_index, grid_y_index, anchor_index * 5: (anchor_index + 1) * 5] = [1, relative_center_x, relative_center_y, width, height]

            expected_output = cells.flatten()

            if not validation:
                self.xdata.append(input_data)
                self.ydata.append(expected_output)

            else:
                self.val_xdata.append(input_data)
                self.val_ydata.append(expected_output)

    def get(self, validation=False):
        while len(self.xdata) < self.batch_size:
            print(len(self.xdata))
            continue

        if not validation:
            xdata = self.xdata[:self.batch_size]
            ydata = self.ydata[:self.batch_size]

            self.xdata = self.xdata[self.batch_size:]
            self.ydata = self.ydata[self.batch_size:]

        else:
            xdata = self.val_xdata[:self.batch_size]
            ydata = self.val_ydata[:self.batch_size]

            self.val_xdata = self.val_xdata[self.batch_size:]
            self.val_ydata = self.val_ydata[self.batch_size:]

        return np.array(xdata), np.array(ydata)

    
def preprocess_data():
    xdata = []
    ydata = []

    for filename in tqdm(os.listdir('gameplay')):
        locations_filename = f'annotations\\{filename.replace(".png", ".txt")}'

        if os.path.exists(locations_filename):
            with open(locations_filename, "r+") as file:
                lines = file.read().splitlines()
                location_data = np.array([
                    [
                        np.float64(value) for value in line.split(' ')[1:]
                    ] for line in lines
                ], dtype=np.float64)

            midpoints = location_data[:, 2] * location_data[:, 3]
                
            location_data = location_data[midpoints.argsort()[::-1]][:objects]
            objects_present = location_data.shape[0]

        else:
            location_data = np.array([])
            objects_present = 0

        location_data += 10e-8

        image = Image.open(f'gameplay\\{filename}').resize((image_width, image_height))

        for _ in range(3):
            if location_data.shape[0] != 0:

                classes = ['enemy'] * len(location_data)

                augmenter = A.Compose([
                    A.HorizontalFlip(p=0.5),

                    A.ShiftScaleRotate(
                        shift_limit=0.0625, 
                        scale_limit=0.0, 
                        rotate_limit=10,
                        border_mode=cv2.BORDER_REPLICATE,
                        interpolation=1,
                        p=0.5
                    ),

                    A.RandomResizedCrop(
                        scale=(0.6, 0.9),
                        size=(image_height, image_width),
                        ratio=(1, 1),
                        p=0.5,
                    ),

                    A.RGBShift(p=0.45),
                    A.HueSaturationValue(hue_shift_limit=(-20, 20), sat_shift_limit=(-30, 30), val_shift_limit=(-20, 20), p=0.45),

                    A.RandomBrightnessContrast(p=0.25),
                    A.Emboss(
                        alpha=(0.2, 0.3),
                        strength=(0.1, 0.2),
                        p=0.25,
                    ),

                    A.CLAHE(p=0.1),
                ], bbox_params=A.BboxParams(format='yolo', min_visibility=0.25, label_fields=['class_labels']))

                augmented_result = augmenter(image=np.array(image), bboxes=location_data, class_labels=classes)

                image = augmented_result['image']
                bboxes = np.array(augmented_result['bboxes'])
                objects_present = bboxes.shape[0]
                bboxes = bboxes.reshape((int(objects_present), 4))

            else:
                imgae = np.asarray(image)
                bboxes = []

            input_data = image / 255

            cells = np.zeros((grid_size, grid_size, 5 * anchors))

            standing_ratio = 4.789523615911094

            for true_center_x, true_center_y, width, height in bboxes:
                grid_x_index = int(true_center_x * grid_size)
                grid_y_index = int(true_center_y * grid_size)

                relative_center_x = true_center_x * grid_size - grid_x_index
                relative_center_y = true_center_y * grid_size - grid_y_index

                height_ratio = height / width

                anchor_index = np.floor((height_ratio / standing_ratio) * anchors).astype(int)
                anchor_index = min(anchors-1, anchor_index)

                cells[grid_x_index, grid_y_index, anchor_index * 5: (anchor_index + 1) * 5] = [1, relative_center_x, relative_center_y, width, height]

            expected_output = cells.flatten()

            xdata.append(input_data)
            ydata.append(expected_output)

    return np.array(xdata), np.array(ydata)

def save():
    save_data = network.save()

    with open(save_file, 'wb') as file:
        file.write(pickle.dumps(save_data))

def conv(depth, kernel_shape, stride=1, padding="SAME"):
    return [
        Conv2d(depth=depth, kernel_shape=kernel_shape, stride=stride, padding=padding),
        # BatchNorm(epsilon=1e-5),
        Activation(activation_function)
    ]

def res_block(filters, extra_layers=[]):
    return ResidualBlock([
        *conv(filters, (1, 1)),
        *conv(filters * 2, (3, 3)),
        *extra_layers
    ])

def long_res_block(filters, repeats):
    block = res_block(filters)
    for _ in range(repeats - 1):
        block = res_block(filters, [block])

    return block

if __name__ == "__main__":
    training_percent = 0.8
    batch_size = 16
    image_width, image_height = [416, 416]

    grid_size = 13
    anchors = 3
    objects = 4

    dropout_rate = 0
    activation_function = Silu()
    variance = "He"
    dtype = np.float32

    save_file = 'model-training-data.json'

    # model = [
    #     Input((image_height, image_width, 3)),

    #     Conv2d(32, (3, 3), padding="SAME"),
    #     MaxPool((2, 2)),
    #     BatchNorm(epsilon=1e-5),
    #     Activation(activation_function),

    #     Conv2d(64, (3, 3), padding="SAME"),
    #     MaxPool((2, 2)),
    #     BatchNorm(epsilon=1e-5),
    #     Activation(activation_function),

    #     Conv2d(128, (3, 3), padding="SAME"),
    #     BatchNorm(epsilon=1e-5),
    #     Activation(activation_function),

    #     Conv2d(64, (1, 1), padding="VALID"),
    #     BatchNorm(epsilon=1e-5),
    #     Activation(activation_function),

    #     Conv2d(128, (3, 3), padding="SAME"),
    #     MaxPool((2, 2)),
    #     BatchNorm(epsilon=1e-5),
    #     Activation(activation_function),

    #     Conv2d(256, (3, 3), padding="SAME"),
    #     BatchNorm(epsilon=1e-5),
    #     Activation(activation_function),

    #     Conv2d(128, (1, 1), padding="VALID"),
    #     BatchNorm(epsilon=1e-5),
    #     Activation(activation_function),

    #     Conv2d(256, (3, 3), padding="SAME"),
    #     MaxPool((2, 2)),
    #     BatchNorm(epsilon=1e-5),
    #     Activation(activation_function),

    #     Conv2d(512, (3, 3), padding="SAME"),
    #     BatchNorm(epsilon=1e-5),
    #     Activation(activation_function),

    #     Conv2d(256, (1, 1), padding="VALID"),
    #     BatchNorm(epsilon=1e-5),
    #     Activation(activation_function),

    #     Conv2d(512, (3, 3), padding="SAME"),
    #     BatchNorm(epsilon=1e-5),
    #     Activation(activation_function),

    #     Conv2d(256, (1, 1), padding="VALID"),
    #     BatchNorm(epsilon=1e-5),
    #     Activation(activation_function),

    #     Conv2d(512, (3, 3), padding="SAME"),
    #     BatchNorm(epsilon=1e-5),
    #     Activation(activation_function),

    #     # EXTRA

    #     # Conv2d(256, (1, 1), padding="VALID"),
    #     # BatchNorm(epsilon=1e-5),
    #     # Activation(activation_function),

    #     # Conv2d(512, (3, 3), padding="SAME"),
    #     # BatchNorm(epsilon=1e-5),
    #     # Activation(activation_function),

    #     # Conv2d(256, (1, 1), padding="VALID"),
    #     # BatchNorm(epsilon=1e-5),
    #     # Activation(activation_function),

    #     # Conv2d(512, (3, 3), padding="SAME"),
    #     # BatchNorm(epsilon=1e-5),
    #     # Activation(activation_function),

    #     ConcatBlock(
    #         [
    #             # MaxPool((2, 2)),
    #             Conv2d(1024, (3, 3), padding="SAME"),
    #             # BatchNorm(epsilon=1e-5),
    #             Activation(activation_function),

    #             Conv2d(512, (1, 1), padding="VALID"),
    #             # BatchNorm(epsilon=1e-5),
    #             Activation(activation_function),

    #             Conv2d(1024, (3, 3), padding="SAME"),
    #             # BatchNorm(epsilon=1e-5),
    #             Activation(activation_function),

    #             Conv2d(512, (1, 1), padding="VALID"),
    #             # BatchNorm(epsilon=1e-5),
    #             Activation(activation_function),

    #             Conv2d(1024, (3, 3), padding="SAME"),
    #             # BatchNorm(epsilon=1e-5),
    #             Activation(activation_function),

    #             Conv2d(1024, (3, 3), padding="SAME"),
    #             # BatchNorm(epsilon=1e-5),
    #             Activation(activation_function),

    #             Conv2d(1024, (3, 3), padding="SAME", stride=2),
    #             # BatchNorm(epsilon=1e-5),
    #             Activation(activation_function),

    #             # EXTRA

    #             # Conv2d(1024, (3, 3), padding="SAME"),
    #             # BatchNorm(epsilon=1e-5),
    #             # Activation(activation_function),

    #         ],
    #         [
    #             Space2Depth(2)
    #         ]
    #     ),

    #     Conv2d(1024, (3, 3), padding="SAME"),
    #     # BatchNorm(epsilon=1e-5),
    #     Activation(activation_function),

    #     Conv2d(anchors * 5, (1, 1), padding="VALID"),
    #     Activation(Sigmoid()),

    #     Flatten()
    # ]

    model = [
        Input((image_height, image_width, 3)),

        *conv(32, (3, 3)),
        *conv(64, (3, 3), stride=2, padding="SAME"),
        res_block(32),

        *conv(128, (3, 3), stride=2, padding="SAME"),
        long_res_block(64, 2),
        
        *conv(256, (3, 3), stride=2, padding="SAME"),
        long_res_block(128, 4),

        # ROUTE 1

        *conv(512, (3, 3), stride=2, padding="SAME"),
        long_res_block(256, 8),

        # ROUTE 2
        *conv(1024, (3, 3), stride=2, padding="SAME"),
        long_res_block(512, 8),

        # ROUTE: 3
        Conv2d(anchors * 5, (1, 1), padding="VALID"),
        Activation(Sigmoid()),

        Flatten()
    ]


    network = Network(model, 
        loss_function = YoloLoss(grid_size=grid_size, anchors=anchors, coordinate_weight=1, no_object_weight=0.01, object_weight=1), 
        optimizer = Adam(momentum = 0.9, beta_constant = 0.99), 
        # scheduler = StepLR(decay_rate=0.5, decay_interval=5), 
        scheduler=ExponentialDecay(learning_rate=0.005, decay_rate=0.99),
        gpu_mem_frac = 1, 
        dtype = dtype
    )

    if os.path.exists("model-training-data.json"):
        network.load(pickle.load(open('model-training-data.json', 'rb')))
    else:
        network.compile()

    costs = []
    val_costs = []

    plt.ion()

    best_cost = float("inf")
    val_best_cost = float("inf")

    # generator = Generate(batch_size=batch_size, training_percent=training_percent, queue_size=32)
    # dataset_size = generator.xdata
    xdata, ydata = preprocess_data()
    dataset_size = len(xdata)

    # while len(generator.xdata) < generator.queue_size:
    #     print(f"{100 * len(generator.xdata) / generator.queue_size}%")

    for idx, cost in enumerate(network.fit(xdata=xdata, ydata=ydata, batch_size=batch_size, learning_rate=0.00001, epochs = 200000)):

        # selected_val_xdata, selected_val_ydata = generator.get(validation=True)
        # selected_val_xdata = selected_val_xdata.astype(dtype)
        # selected_val_ydata = selected_val_ydata.astype(dtype)

        # outputs = network.forward(selected_val_xdata, training=False)

        # val_cost = tf.stack([
        #     network.loss_function.forward(output, expected_output) for output, expected_output in zip(outputs, selected_val_ydata)
        # ]).numpy().mean(axis=0)

        # val_costs.append(val_cost)
        costs.append(cost)

        try:
            plt.plot(np.arange(len(costs)) * (batch_size / dataset_size), costs, label=('training_object_loss', 'training_no_object_loss', 'training_coordinate_loss'))
            # plt.plot(np.arange(len(val_costs)) * (batch_size / dataset_size), val_costs, label=('validation_object_loss', 'validation_no_object_loss', 'validation_coordinate_loss'))

            plt.legend()
            plt.draw()
            plt.pause(0.01)
            plt.clf()

            if not idx % 10:
                threading.Thread(target=save).start()
        except Exception as e:
            print(e)