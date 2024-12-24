from utils.layers import *
from utils.schedulers import *
from utils.network import Network
from utils.loss import *
from utils.activations import *
from utils.functions import Processing, AutoClipper, ClipGradient
from utils.optimizers import Adam, Momentum, RMSProp, SGD
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
                            value for value in line.split(' ')[1:]
                        ] for line in lines
                    ], dtype=np.float32)

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
                        border_mode=cv2.BORDER_DEFAULT,
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
                image = np.asarray(image)
                bboxes = []

            input_data = image / 255

            expected_outputs = []

            # for grid_size in [52, 26, 13]:
            for i in range(len(backprop_layer_indices)):
                _grid_size = grid_size * 2 ** i

                cells = np.zeros((_grid_size, _grid_size, 5 * anchors))

                standing_ratio = 4.789523615911094

                for true_center_x, true_center_y, width, height in bboxes:
                    grid_x_index = int(true_center_x * _grid_size)
                    grid_y_index = int(true_center_y * _grid_size)

                    relative_center_x = true_center_x * _grid_size - grid_x_index
                    relative_center_y = true_center_y * _grid_size - grid_y_index

                    height_ratio = height / width

                    anchor_index = np.floor((height_ratio / standing_ratio) * anchors).astype(int)
                    anchor_index = min(anchors-1, anchor_index)

                    cells[grid_x_index, grid_y_index, anchor_index * 5: (anchor_index + 1) * 5] = [1, relative_center_x, relative_center_y, width, height]

                expected_output = cells.flatten()
                expected_outputs.append(expected_output[::-1])

            if not validation:
                self.xdata.append(input_data)
                self.ydata.append(expected_outputs)

            else:
                self.val_xdata.append(input_data)
                self.val_ydata.append(expected_outputs)

    def get(self, validation=False):
        while len(self.xdata) < self.batch_size:
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

def get_bboxes(choices):
    bboxes = np.empty((0, 5))
    images = []

    augmenter = A.Compose([
        A.HorizontalFlip(p=0.5),

        A.RandomSizedBBoxSafeCrop(
            width=image_width,
            height=image_height,
            erosion_rate=0.01,
            p=0.5
        ),

        A.Affine(
            scale=(0.9, 1.1),
            rotate=(-5, 5),
            translate_percent=(-0.025, 0.025),
            shear=(-5, 5),
            rotate_method="largest_box",
            keep_ratio=True,
            balanced_scale=True,
            p=0.8
        ),

        A.HueSaturationValue(hue_shift_limit=(-2, 2), sat_shift_limit=(-20, 20), val_shift_limit=(-20, 20), p=0.5),
        A.ColorJitter(brightness=(0.9, 1.1), contrast=(0.9, 1.1), saturation=(0.9, 1.1), hue=(-0.05, 0.05), p=0.5),
        A.RandomBrightnessContrast(p=0.5),

    ], bbox_params=A.BboxParams(format='yolo', min_visibility=0.25, label_fields=['class_labels']))

    for idx, filename in tqdm(list(enumerate(np.array(os.listdir('gameplay'))[choices]))):
        locations_filename = f'annotations\\{filename.replace(".png", ".txt")}'

        if os.path.exists(locations_filename):
            with open(locations_filename, "r+") as file:
                lines = file.read().splitlines()
                location_data = np.array([
                    [
                        np.float64(value) for value in line.split(' ')[1:]
                    ] for line in lines
                ], dtype=np.float64)

            location_data = location_data[:objects]
            objects_present = location_data.shape[0]

        else:
            print("[LOG] Empty image detected")
            continue

        location_data += 10e-8

        try:
            root_image = cv2.resize(
                            cv2.cvtColor(
                                    cv2.imread(f'gameplay\\{filename}'), 
                                    cv2.COLOR_BGR2RGB
                                ), (image_width, image_height)
                            )
        except Exception as e:
            print(filename, e, "[LOG] BROKEN")
            continue

        for i in range(repeats):
            if (location_data.shape[0] != 0):

                classes = ['enemy'] * len(location_data)
                augmented_result = augmenter(image=root_image, bboxes=location_data, class_labels=classes)

                image = augmented_result['image']

                images.append(image / 255)

                _bboxes = np.array(augmented_result['bboxes'])
                indices = np.repeat(idx * repeats + i, _bboxes.size // 4)[:, None]
                _bboxes = np.hstack((indices, _bboxes.reshape((-1, 4))))

                bboxes = np.concatenate((bboxes, _bboxes))

    return bboxes, np.array(images)

def get_anchor_data(grid_size, bboxes, choices):

    # BBoxes Format: [idx, x, y, w, h]

    aspect_ratios = bboxes[..., 3] / bboxes[..., 4]
    order = np.argsort(aspect_ratios)

    split_boxes = np.array_split(bboxes[..., 3:][order], anchors)

    points = np.floor((np.arange(anchors) / anchors) * len(bboxes)).astype(int) # Evenly distributed indices
    deciding_aspect_ratios = np.array([np.max(group[..., 0] / group[..., 1]) for group in split_boxes])
    anchor_dimensions = np.array([np.mean(group, axis=0) for group in split_boxes])

    print("[LOG] Anchor data:\n", deciding_aspect_ratios, anchor_dimensions)

    return deciding_aspect_ratios, anchor_dimensions
    
def preprocess_data(grid_size, bboxes, images, choices):
    xdata = []
    ydata = []

    anchors_count = np.zeros(anchors)

    for (idx, true_center_x, true_center_y, width, height) in tqdm(bboxes):
        idx = int(idx)
        input_data = images[idx]

        expected_outputs = []

        aspect_ratio = width / height

        indices = np.where((aspect_ratio // deciding_aspect_ratios) >= 1)[0]

        if indices.size == 0:
            anchor_index = anchors - 1
        else:
            anchor_index = int(indices[-1])

        anchors_count[anchor_index] += 1

        _width = np.log(width / anchor_dimensions[anchor_index, 0])
        _height = np.log(height / anchor_dimensions[anchor_index, 1])

        for i in range(len(backprop_layer_indices)):
            _grid_size = grid_size * 2 ** i
            cells = np.zeros((_grid_size, _grid_size, 5 * anchors))

            grid_x_index = int(true_center_x * _grid_size)
            grid_y_index = int(true_center_y * _grid_size)

            relative_center_x = true_center_x * _grid_size - grid_x_index
            relative_center_y = true_center_y * _grid_size - grid_y_index

            cells[grid_x_index, grid_y_index, anchor_index * 5: (anchor_index + 1) * 5] = [1, relative_center_x, relative_center_y, _width, _height]

            expected_output = cells.flatten()
            expected_outputs.append(expected_output)

        expected_outputs = expected_outputs[::-1]

        if len(xdata) < (idx + 1):
            xdata.append(input_data)
            ydata.append(expected_outputs)
        else:
            update_masks = [expected_output != 0 for expected_output in expected_outputs]

            for scale_idx, (expected_output, update_mask) in enumerate(zip(expected_outputs, update_masks)):
                ydata[idx][scale_idx][update_mask] = expected_output[update_mask]

    print("[LOG] Anchor distribution:", anchors_count)

    return np.array(xdata), ydata

def save():
    save_data = network.save()

    with open(save_file, 'wb') as file:
        file.write(pickle.dumps(save_data))

    with open("cost-overtime.json", "w+") as file:
        file.write(json.dumps(costs))

def conv(depth, kernel_shape, stride=1, padding="SAME"):
    return [
        Conv2d(depth=depth, kernel_shape=kernel_shape, stride=stride, padding=padding),
        BatchNorm(),
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
    training_percent = 0.012
    batch_size = 16
    image_width, image_height = [416, 416]
    
    grid_size = int(image_width / 32)
    anchors = 3
    objects = 4

    dropout_rate = 0
    activation_function = Mish()
    variance = "He"
    dtype = np.float16

    save_file = 'model-training-data.json'
    repeats = 1

    # model = [
    #     Input((image_height, image_width, 3)),

    #     Conv2d(32, (3, 3), padding="SAME"),
    #     MaxPool((2, 2)),
    #     # BatchNorm(epsilon=1e-5),
    #     Activation(activation_function),

    #     Conv2d(64, (3, 3), padding="SAME"),
    #     MaxPool((2, 2)),
    #     # BatchNorm(epsilon=1e-5),
    #     Activation(activation_function),

    #     Conv2d(128, (3, 3), padding="SAME"),
    #     # BatchNorm(epsilon=1e-5),
    #     Activation(activation_function),

    #     Conv2d(64, (1, 1), padding="VALID"),
    #     # BatchNorm(epsilon=1e-5),
    #     Activation(activation_function),

    #     Conv2d(128, (3, 3), padding="SAME"),
    #     MaxPool((2, 2)),
    #     # BatchNorm(epsilon=1e-5),
    #     Activation(activation_function),

    #     Conv2d(256, (3, 3), padding="SAME"),
    #     # BatchNorm(epsilon=1e-5),
    #     Activation(activation_function),

    #     Conv2d(128, (1, 1), padding="VALID"),
    #     # BatchNorm(epsilon=1e-5),
    #     Activation(activation_function),

    #     Conv2d(256, (3, 3), padding="SAME"),
    #     MaxPool((2, 2)),
    #     # BatchNorm(epsilon=1e-5),
    #     Activation(activation_function),

    #     Conv2d(512, (3, 3), padding="SAME"),
    #     # BatchNorm(epsilon=1e-5),
    #     Activation(activation_function),

    #     Conv2d(256, (1, 1), padding="VALID"),
    #     # BatchNorm(epsilon=1e-5),
    #     Activation(activation_function),

    #     Conv2d(512, (3, 3), padding="SAME"),
    #     # BatchNorm(epsilon=1e-5),
    #     Activation(activation_function),

    #     Conv2d(256, (1, 1), padding="VALID"),
    #     # BatchNorm(epsilon=1e-5),
    #     Activation(activation_function),

    #     Conv2d(512, (3, 3), padding="SAME"),
    #     # BatchNorm(epsilon=1e-5),
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
    #             MaxPool((2, 2)),
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

    #             Conv2d(1024, (3, 3), padding="SAME"),
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
        
        # ROUTE 1

        *conv(256, (3, 3), stride=2, padding="SAME"),
        long_res_block(128, 8),
        
        # ROUTE 2
        ConcatBlock(
            [
                *conv(512, (3, 3), stride=2, padding="SAME"),
                long_res_block(256, 8),
            ], 
            [
                Space2Depth(2)
            ]
        ),

        # ROUTE: 3
        *conv(1024, (3, 3), stride=2, padding="SAME"),
        long_res_block(512, 4),
    ]

    backprop_layer_indices = [-1, -5, -6]

    addon_layers = [
        [
            Conv2d(anchors * 5, (1, 1), padding="VALID"),
            Activation(YoloActivation()),
            Flatten()
        ],
        [
            Conv2d(anchors * 5, (1, 1), padding="VALID"),
            Activation(YoloActivation()),
            Flatten()
        ],
        [
            Conv2d(anchors * 5, (1, 1), padding="VALID"),
            Activation(YoloActivation()),
            Flatten()
        ]
    ]

    grid_count = grid_size ** 2

    cooridnate_weight = 0.5
    no_object_weight = 0.5
    object_weight = 1

    dataset_size = len(os.listdir('gameplay'))
    choices = np.random.choice(dataset_size, size=int(dataset_size * training_percent), replace=False)

    with open('training-files.json', 'w+') as file:
        file.write(json.dumps(choices.tolist()))
    
    bboxes, images = get_bboxes(choices)
    deciding_aspect_ratios, anchor_dimensions = get_anchor_data(grid_size, bboxes, choices)
    xdata, ydata = preprocess_data(grid_size, bboxes, images, choices)
    dataset_size = xdata.shape[0] / repeats

    network = Network(
        model=model,
        addon_layers=addon_layers,
        backprop_layer_indices=backprop_layer_indices,
        loss_function = [
            YoloLoss(coordinate_loss_function=MSE, objectness_loss_function=MSE, grid_size=grid_size * 4, anchors=anchors, coordinate_weight=cooridnate_weight, no_object_weight=no_object_weight / 16, object_weight=object_weight, anchor_dimensions=anchor_dimensions),
            YoloLoss(coordinate_loss_function=MSE, objectness_loss_function=MSE, grid_size=grid_size * 2, anchors=anchors, coordinate_weight=cooridnate_weight, no_object_weight=no_object_weight / 4, object_weight=object_weight, anchor_dimensions=anchor_dimensions),
            YoloLoss(coordinate_loss_function=MSE, objectness_loss_function=MSE, grid_size=grid_size, anchors=anchors, coordinate_weight=cooridnate_weight, no_object_weight=no_object_weight, object_weight=object_weight, anchor_dimensions=anchor_dimensions)
        ],
        # loss_function = YoloLoss(grid_size=grid_size, anchors=anchors, coordinate_weight=5, no_object_weight=no_object_weight, object_weight=1),
        optimizer = Adam(momentum = 0.9, beta_constant = 0.99, weight_decay=1e-4), 
        # optimizer = RMSProp(beta_constant = 0.9),
        # optimizer = Momentum(momentum=0.9),
        scheduler = StepLR(initial_learning_rate=0.00005, decay_rate=0.5, decay_interval=100 / repeats), 
        # scheduler=CosineAnnealingDecay(initial_learning_rate=0.00003, min_learning_rate=0.00001, initial_cycle_size=50 / repeats, cycle_mult=2),
        # scheduler=ExponentialDecay(initial_learning_rate=0.00007, decay_rate=0.995),
        gpu_mem_frac = 1.0, 
        dtype = dtype
    )

    if os.path.exists("model-training-data.json"):
        network.load(pickle.load(open('model-training-data.json', 'rb')))
        costs = json.load(open("cost-overtime.json", "r+"))
        starting_idx = len(costs)
    else:
        costs = []
        starting_idx = 0
        network.compile()

    plt.ion()
    plt.figure(figsize=(16, 8))

    titles = ['object_loss', 'no_object_loss', 'coordinate_loss']
    colors = ['C0', 'C1', 'C2']

    for idx, cost in enumerate(network.fit(xdata=xdata, ydata=ydata, batch_size=batch_size, learning_rate=0.0001, epochs = 200000, gradient_transformer=ClipGradient(10))):

        print(cost)

        costs.append(cost)

        try:
            for i in range(len(backprop_layer_indices)):
                reversed_idx = len(backprop_layer_indices) - (i + 1)
                for j in range(3):
                    plt.subplot(3, len(backprop_layer_indices), (i * 3) + j + 1)
                    plt.plot(np.arange(idx + starting_idx + 1) * (batch_size / dataset_size), np.array(costs)[:, i, j], colors[j], label=titles[j])
                    plt.title(f"{grid_size * 2 ** (reversed_idx)}x{grid_size * 2 ** (reversed_idx)} ({titles[j]})")

            plt.legend()
            plt.draw()
            plt.pause(0.01)
            plt.clf()

            if not idx % 5:
                threading.Thread(target=save).start()
        except Exception as e:
            print(e)