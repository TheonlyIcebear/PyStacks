from utils.layers import *
from utils.schedulers import *
from utils.network import Network
from utils.loss import *
from utils.activations import *
from sklearn.cluster import KMeans
from utils.functions import Processing, AutoClipper, ClipGradient
from utils.optimizers import Adam, Momentum, RMSProp, SGD
from PIL import Image
from tqdm import tqdm
import matplotlib.animation as animation
from matplotlib.ticker import MaxNLocator
import albumentations as A, matplotlib.pyplot as plt
import multiprocessing, threading, numpy as np, pickle, time, json, cv2, os

class Generate:
    def __init__(self, batch_size, anchor_dimensions, dimensions, grid_size, anchors, choices, iou_ignore_threshold = 0.72, data_augmentation=True):
        self.batch_size = batch_size

        self.anchor_dimensions = anchor_dimensions
        self.data_augmentation = data_augmentation

        self.image_width, self.image_height = dimensions

        self.dataset_size = len(choices)
        self.choices = choices

        manager = multiprocessing.Manager()
        self.buffer = manager.list()
        self.buffer_size = batch_size

        self.grid_size = grid_size
        self.anchors = anchors

        self.iou_ignore_threshold = iou_ignore_threshold

        multiprocessing.Process(target=self.fill_buffer).start()

    def fill_buffer(self):
        augmentor = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.HueSaturationValue(hue_shift_limit=(-5.0, 5.0), sat_shift_limit=(-5.0, 5.0), val_shift_limit=(-5.0, 5.0), p=0.5),

            A.Affine( 
                rotate=(-1, 1),
                translate_percent=(-0.05, 0.05),
                rotate_method="largest_box",
                keep_ratio=True,
                balanced_scale=True,
                p=0.5,
                border_mode=cv2.BORDER_REFLECT
            ),

            A.Perspective(scale=(0.05, 0.1), p=0.5),

            A.GridDistortion(
                num_steps=5, 
                distort_limit=(-0.2, 0.2),
                normalized=True,
                p=0.5
            ),
        

        ], bbox_params=A.BboxParams(format='yolo', min_visibility=0.25, label_fields=['class_labels']))
        
        while True:
            if len(self.buffer) >= self.buffer_size:
                continue

            choices = np.random.choice(self.choices, size=self.buffer_size - len(self.buffer), replace=False)

            filenames = np.array(os.listdir('gameplay'))[choices]

            buffer_extension = []

            for idx, filename in enumerate(filenames):
                locations_filename = f'annotations\\{filename.replace(".png", ".txt")}'

                if os.path.exists(locations_filename):
                    with open(locations_filename, "r+") as file:
                        lines = file.read().splitlines()
                        location_data = np.array([
                            [
                                np.float64(value) for value in line.split(' ')[1:]
                            ] for line in lines
                        ], dtype=np.float64)


                else:
                    print("[LOG] Empty image detected")
                    location_data = np.array([])

                location_data += 10e-8

                try:
                    root_image = cv2.resize(
                                    cv2.cvtColor(
                                            cv2.imread(f'gameplay\\{filename}'), 
                                            cv2.COLOR_BGR2RGB
                                        ), (self.image_width, self.image_height)
                                    )
                except Exception as e:
                    print(filename, e, "[LOG] BROKEN")
                    continue

                classes = ['enemy'] * len(location_data)

                if self.data_augmentation:
                    augmented_result = augmentor(image=root_image, bboxes=location_data, class_labels=classes)
                    bboxes = np.array(augmented_result['bboxes'])
                    image = augmented_result['image']

                else:
                    bboxes = np.array(location_data)
                    image = root_image

                image = image / 255

                ydata = [np.zeros((self.grid_size * 2 ** i, self.grid_size * 2 ** i, self.anchors, 5)) for i in range(3)]

                for (true_center_x, true_center_y, width, height) in bboxes:
                    formatted_anchor_dimensions = np.concatenate(
                        (np.full((self.anchor_dimensions.shape[0], 2), 0), 
                        self.anchor_dimensions
                    ), axis=-1), # [[0, 0, w, h], ...]

                    iou_values = Processing.iou(
                        cp.array(formatted_anchor_dimensions),
                        cp.array([0, 0, width, height])
                    )[0, :, 0].get()

                    anchor_indices = iou_values.argsort()[::-1]

                    has_anchor = [False, False, False]

                    for anchor_index in anchor_indices:
                        scale_idx = anchor_index // self.anchors
                        anchor_on_scale = anchor_index % self.anchors

                        _grid_size = self.grid_size * 2 ** scale_idx

                        _width = np.log(width / self.anchor_dimensions[anchor_index, 0])
                        _height = np.log(height / self.anchor_dimensions[anchor_index, 1])

                        grid_x_index = int(true_center_x * _grid_size)
                        grid_y_index = int(true_center_y * _grid_size)

                        relative_center_x = true_center_x * _grid_size - grid_x_index
                        relative_center_y = true_center_y * _grid_size - grid_y_index

                        occupied = ydata[scale_idx][grid_x_index, grid_y_index, anchor_on_scale, 0]
                        if not occupied and not has_anchor[scale_idx]:
                            has_anchor[scale_idx] = True
                            ydata[scale_idx][grid_x_index, grid_y_index, anchor_on_scale] = [1, relative_center_x, relative_center_y, _width, _height]

                        elif not occupied and iou_values[anchor_index] > self.iou_ignore_threshold:
                            ydata[scale_idx][grid_x_index, grid_y_index, anchor_on_scale, 0] = -1
                    
                ydata = [scale.reshape(self.grid_size * 2 ** i, self.grid_size * 2 ** i, self.anchors * 5) for i, scale in enumerate(ydata)]
                buffer_extension.append((image, ydata))

            self.buffer += buffer_extension

    def __call__(self):
        choices = np.random.choice(self.choices, size=self.batch_size, replace=False)
        
        xdatas = []
        ydatas = []

        while len(self.buffer) < self.batch_size:
            pass

        for idx in range(self.batch_size):
            xdata, ydata = self.buffer[idx]

            xdatas.append(xdata)
            ydatas.append(ydata)

        del self.buffer[:self.batch_size]

        return np.array(xdatas), ydatas

def draw_boxes(image, points, color):
    predicted_points = np.array(points.reshape((-1, 2, 2)))

    draw = ImageDraw.Draw(image)

    dimensions = np.array(image.size)

    for center, distances in predicted_points:
        top_left = (center - (distances / 2))  * dimensions
        bottom_right = (center + (distances / 2)) * dimensions

        draw.line([(top_left[0], top_left[1]), (bottom_right[0], top_left[1])], fill=color, width=10)
        draw.line([(top_left[0], bottom_right[1]), (bottom_right[0], bottom_right[1])], fill=color, width=10)

        draw.line([(top_left[0], top_left[1]), (top_left[0], bottom_right[1])], fill=color, width=10)
        draw.line([(bottom_right[0], top_left[1]), (bottom_right[0], bottom_right[1])], fill=color, width=10)

def parse_output(outputs, grid_size, anchor_dimensions):
    _outputs = cp.empty((0, 5))
    for idx, output in enumerate(outputs):

        grid_size = (2 ** (len(outputs) - (idx + 1))) * grid_size
        output = cp.array(output).reshape(-1, 5)

        idx = cp.arange(output.shape[0])

        grid_x_index = (idx // anchors) // grid_size
        grid_y_index = (idx // anchors) % grid_size

        print(output.shape, grid_size, anchor_dimensions)

        relative_center_x = output[:, 1]
        relative_center_y = output[:, 2]

        center_x = (grid_x_index + relative_center_x) / grid_size
        center_y = (grid_y_index + relative_center_y) / grid_size

        output[:, 1] = center_x
        output[:, 2] = center_y

        output[:, [3, 4]] = (cp.exp(output[:, [3, 4]]).reshape(-1, anchors, 2) * anchor_dimensions).reshape(-1, 2)

        _outputs = cp.concatenate((_outputs, output))

    outputs = _outputs

    object_presence_scores = outputs[:, 0]
    present_boxes_indices = object_presence_scores >= 0.95
    
    object_presence_scores = object_presence_scores[present_boxes_indices]
    print(object_presence_scores)
    unprocessed_box_data = outputs[present_boxes_indices][object_presence_scores.argsort()][::-1]

    unprocessed_box_data = unprocessed_box_data[:, 1:]

    box_data = []
    while len(unprocessed_box_data) > 0:
        current = unprocessed_box_data[0]
        unprocessed_box_data = unprocessed_box_data[1:]
        
        iou = Processing.iou(current, unprocessed_box_data)
        # print(current, unprocessed_box_data)

        # print(iou, "IOU")

        box_data.append(current.get())
        
        unprocessed_box_data = unprocessed_box_data[iou < 0.1]

    box_data = np.array(box_data)
    return box_data

def get_bboxes(choices):
    bboxes = np.empty((0, 4))

    augmentor = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.HueSaturationValue(hue_shift_limit=(-5.0, 5.0), sat_shift_limit=(-5.0, 5.0), val_shift_limit=(-5.0, 5.0), p=0.5),

        A.Affine( 
            rotate=(-1, 1),
            translate_percent=(-0.05, 0.05),
            rotate_method="largest_box",
            keep_ratio=True,
            balanced_scale=True,
            p=0.5,
            border_mode=cv2.BORDER_REFLECT
        ),

        A.Perspective(scale=(0.05, 0.1), p=0.5),

        A.GridDistortion(
            num_steps=5, 
            distort_limit=(-0.2, 0.2),
            normalized=True,
            p=0.5
        ),
    

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

        if (location_data.shape[0] != 0):

            classes = ['enemy'] * len(location_data)

            augmented_result = augmentor(image=root_image, bboxes=location_data, class_labels=classes)
            image = augmented_result['image']
            _bboxes = np.array(augmented_result['bboxes'])

            # image = root_image
            # _bboxes = location_data

            bboxes = np.concatenate((bboxes, _bboxes))

    return bboxes

def get_anchor_data(grid_size, bboxes):

    # BBoxes Format: [idx, x, y, w, h]

    if len(bboxes) < 3 * anchors:
        print(np.tile(bboxes[0][2:], (3 * anchors, 1)))
        return np.tile(bboxes[0][2:], (3 * anchors, 1))

    dimensions_count = anchors * yolo_head_count
    dimensions = bboxes[..., 2:4]

    kmeans = KMeans(n_clusters=dimensions_count)
    kmeans.fit(dimensions)

    clusters = [np.empty((2)) for _ in range(dimensions_count)]

    plt.figure(figsize=(8, 8))

    anchor_dimensions = kmeans.cluster_centers_
    order = np.argsort(anchor_dimensions[:, 0] * anchor_dimensions[:, 1])
    anchor_dimensions = anchor_dimensions[order] # Sort by metric

    for idx, (label, wh) in enumerate(zip(kmeans.labels_, dimensions)):
        formatted_anchor_dimensions = np.concatenate(
            (np.zeros((anchor_dimensions.shape[0], 2)), 
            anchor_dimensions
        ), axis=-1), # [[0, 0, w, h], ...]

        iou_values = Processing.iou(
            cp.array(formatted_anchor_dimensions),
            cp.array(np.concatenate(([0, 0], wh)))
        )[0, :, 0].get()

        label = iou_values.argmax()

        clusters[label] = np.vstack((clusters[label], wh))

    # clusters = [clusters[idx] for idx in order]

    count = 0
    for idx, cluster in zip(order, clusters):

        cluster = cluster[(np.prod(cluster, axis=-1) < 5) & (np.prod(cluster, axis=-1) > 0.0001)]

        plt.scatter(
            cluster[:, 0], 
            cluster[:, 1], 
            alpha=0.6, 
            label = f"Group {idx}: {anchor_dimensions[idx][0] * anchor_dimensions[idx][1]}"
        )


    plt.scatter(
        anchor_dimensions[:, 0], 
        anchor_dimensions[:, 1], 
        alpha=1,
        color='red',
        marker='x',
        s=100
    )
    
    plt.title("Bounding Box Dimensions vs. Anchor Boxes", fontsize=14)
    plt.xlabel("Width", fontsize=12)
    plt.ylabel("Height", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    # plt.show()

    print("[LOG] Anchor data:\n", anchor_dimensions)

    return anchor_dimensions
    
def preprocess_data(grid_size, bboxes, images):
    xdata = []
    ydata = []

    anchors_count = np.zeros(anchor_dimensions.shape[0])

    xdata = images
    expected_outputs = lambda: [np.zeros((grid_size * 2 ** i, grid_size * 2 ** i, anchors, 5)) for i in range(yolo_head_count)]
    ydata = [expected_outputs() for _ in range(int(bboxes[-1][0] + 1))]

    for (idx, true_center_x, true_center_y, width, height) in tqdm(bboxes):
        idx = int(idx)
        input_data = images[idx]

        formatted_anchor_dimensions = np.concatenate(
            (np.full((anchor_dimensions.shape[0], 2), 0), 
            anchor_dimensions
        ), axis=-1), # [[0, 0, w, h], ...]

        iou_values = Processing.iou(
            cp.array(formatted_anchor_dimensions),
            cp.array([0, 0, width, height])
        )[0, :, 0].get()

        anchor_indices = iou_values.argsort()[::-1]

        has_anchor = [False, False, False]

        for anchor_index in anchor_indices:
            scale_idx = anchor_index // anchors
            anchor_on_scale = anchor_index % anchors

            _grid_size = grid_size * 2 ** scale_idx

            _width = np.log(width / anchor_dimensions[anchor_index, 0])
            _height = np.log(height / anchor_dimensions[anchor_index, 1])

            grid_x_index = int(true_center_x * _grid_size)
            grid_y_index = int(true_center_y * _grid_size)

            relative_center_x = true_center_x * _grid_size - grid_x_index
            relative_center_y = true_center_y * _grid_size - grid_y_index

            occupied = ydata[idx][scale_idx][grid_x_index, grid_y_index, anchor_on_scale, 0]
            if not occupied and not has_anchor[scale_idx]:
                anchors_count[anchor_index] += 1
                has_anchor[scale_idx] = True
                ydata[idx][scale_idx][grid_x_index, grid_y_index, anchor_on_scale] = [1, relative_center_x, relative_center_y, _width, _height]

            elif not occupied and iou_values[anchor_index] > iou_ignore_threshold:
                print("GET OUT!")
                ydata[idx][scale_idx][grid_x_index, grid_y_index, anchor_on_scale, 0] = -1

    ydata = [[scale.flatten() for scale in data] for data in ydata]

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
        *conv(filters * 2, (3, 3))
    ])

def long_res_block(filters, repeats):
    block = []
    for _ in range(repeats):
        block.append(res_block(filters))

    return block

if __name__ == "__main__":
    training_percent = 1700/1839
    batch_size = 16

    image_width, image_height = [416, 416]
    yolo_head_count = 3
    
    grid_size = int(image_width / 32)
    grid_count = grid_size ** 2

    anchors = 3
    objects = 4

    dropout_rate = 0
    activation_function = Mish()
    variance = "He"
    dtype = np.float32

    save_file = 'model-training-data.json'
    dataset_size = len(os.listdir('gameplay'))
    choices = np.random.choice(dataset_size, size=int(dataset_size * training_percent), replace=False)

    with open('training-files.json', 'w+') as file:
        file.write(json.dumps(choices.tolist()))
    
    bboxes = get_bboxes(choices)
    anchor_dimensions = get_anchor_data(grid_size, bboxes)
    del bboxes

    median_dimension = np.mean(anchor_dimensions[anchor_dimensions.shape[0] // 2])

    print(anchor_dimensions)

    first_concat = Concat()
    second_concat = Concat()

    concat_start1, residual_start1, concat_end1 = first_concat.generate_layers()
    concat_start2, residual_start2, concat_end2 = second_concat.generate_layers()

    weight_initializer = YoloSplit(presence_initializer=HeNormal(), xy_initializer=HeNormal(), dimensions_initializer=LecunNormal())
    bias_initializer = YoloSplit(presence_initializer=Fill(-5), xy_initializer=Fill(0), dimensions_initializer=Fill(np.log(0.1)))

    model = [
        Input((image_height, image_width, 3)),

        *conv(32, (3, 3)),
        *conv(64, (3, 3), stride=2, padding="SAME"),
        res_block(32),

        *conv(128, (3, 3), stride=2, padding="SAME"),
        *long_res_block(64, 2),

        *conv(256, (3, 3), stride=2, padding="SAME"), 
        *long_res_block(128, 8),
        
        concat_start1,
            *conv(512, (3, 3), stride=2, padding="SAME"), 
            *long_res_block(256, 8),
            
            concat_start2,
                *conv(1024, (3, 3), stride=2, padding="SAME"), 
                
                *long_res_block(512, 4),

                *conv(512, (1, 1)),
                *conv(1024, (3, 3)),
                # ROUTE: 1

                Upsample(2),

            residual_start2,
            concat_end2,

            *conv(256, (1, 1)),
            *conv(512, (3, 3)),
            # ROUTE: 2

            Upsample(2),

        residual_start1,
        concat_end1,

        *conv(128, (1, 1)),
        *conv(256, (3, 3)),

        # ROUTE: 3
    ]

    backprop_layer_indices = [-1, -10, -19]

    addon_layers = [
        [
            

            *conv(512, (1, 1)),
            *conv(1024, (3, 3)),
            *conv(512, (1, 1)),

            Conv2d(anchors * 5, (1, 1), padding="VALID", weight_initializer=weight_initializer, bias_initializer=bias_initializer),
            Activation(YoloActivation())
        ],
        [
            
            *conv(256, (1, 1)),
            *conv(512, (3, 3)),
            *conv(256, (1, 1)),

            Conv2d(anchors * 5, (1, 1), padding="VALID", weight_initializer=weight_initializer, bias_initializer=bias_initializer),
            Activation(YoloActivation())
        ],
        [
            *conv(128, (1, 1)),
            *conv(256, (3, 3)),
            *conv(128, (1, 1)),

            Conv2d(anchors * 5, (1, 1), padding="VALID", weight_initializer=weight_initializer, bias_initializer=bias_initializer),
            Activation(YoloActivation())
        ],
    ]

    cooridnate_weight = 10
    no_object_weight = 10
    object_weight = 1

    network = Network(
        model=model,
        addon_layers=addon_layers,
        backprop_layer_indices=backprop_layer_indices,
        loss_function = [
            YoloLoss(coordinate_loss_function=DIoU, objectness_loss_function=BCE, grid_size=grid_size, anchors=anchors, coordinate_weight=cooridnate_weight, no_object_weight=no_object_weight, object_weight=object_weight, anchor_dimensions=anchor_dimensions, dtype=dtype),
            YoloLoss(coordinate_loss_function=DIoU, objectness_loss_function=BCE, grid_size=grid_size, anchors=anchors, coordinate_weight=cooridnate_weight, no_object_weight=no_object_weight, object_weight=object_weight, anchor_dimensions=anchor_dimensions, dtype=dtype),
            YoloLoss(coordinate_loss_function=DIoU, objectness_loss_function=BCE, grid_size=grid_size, anchors=anchors, coordinate_weight=cooridnate_weight, no_object_weight=no_object_weight, object_weight=object_weight, anchor_dimensions=anchor_dimensions, dtype=dtype),
        ],
        # loss_function = YoloLoss(grid_size=grid_size, anchors=anchors, coordinate_weight=5, no_object_weight=no_object_weight, object_weight=1),
        optimizer = Adam(momentum = 0.8, beta_constant = 0.9, weight_decay=1e-7), 
        # optimizer = RMSProp(beta_constant = 0.9),
        # optimizer = Momentum(momentum=0.9),
        scheduler = StepLR(initial_learning_rate=0.0003, decay_rate=0.5, decay_interval=100), 
        # scheduler=CosineAnnealingDecay(initial_learning_rate=0.00003, min_learning_rate=0.00001, initial_cycle_size=50, cycle_mult=2),
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

    generator = Generate(batch_size, anchor_dimensions, (image_width, image_height), grid_size, anchors, choices, data_augmentation=True)
    dataset_size = generator.dataset_size

    for idx, cost in enumerate(network.fit(generator=generator, batch_size=batch_size, learning_rate=0.0001, epochs = 20000000)):

        print(cost)

        costs.append(cost)
        plt.clf()

        try:
            if not idx % 5:
                for i in range(yolo_head_count):
                    for j in range(3):
                        plt.subplot(3, yolo_head_count, (i * 3) + j + 1)

                        plt.plot(np.arange(idx + starting_idx + 1) * (batch_size / dataset_size), np.array(costs)[:, i, j], colors[j], label=titles[j])

                        plt.xscale('linear')
                        plt.yscale('linear')

                        plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=10))  # For 5 ticks on the x-axis
                        plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=10))

                        plt.title(f"{grid_size * 2 ** i}x{grid_size * 2 ** i} ({titles[j]})")

                plt.draw()
                plt.pause(0.001)
        except Exception as e:
            print(e)

        if not idx % 5 and not np.isnan(cost).any():
            threading.Thread(target=save).start()
