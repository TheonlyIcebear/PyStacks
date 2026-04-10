from utils.layers import *
from utils.schedulers import *
from utils.network import Network
from utils.initializers import *
from utils.loss import *
from utils.activations import *
from datetime import datetime
from itertools import islice
from sklearn.cluster import KMeans
from utils.functions import Processing, AutoClipper, ClipGradient
from utils.optimizers import Adam, Momentum, RMSProp, SGD
from PIL import Image
from tqdm import tqdm
from queue import Queue
from collections import deque
import matplotlib.animation as animation
from matplotlib.ticker import MaxNLocator
import albumentations as A, matplotlib.pyplot as plt
from typing import Annotated, Any, Literal, Union, cast
import multiprocessing, threading, collections, numpy as np, pickle, time, json, cv2, os
import time
import random
import tf2onnx

class Generate:
    def __init__(self, batch_size, anchor_dimensions, dimensions, grid_size, anchors, classes, choices, buffer_size, iou_ignore_threshold = 0.8, data_augmentation=True):
        global augmentor
        self.batch_size = batch_size

        anchor_dimensions = anchor_dimensions
        self.formatted_anchor_dimensions = np.concatenate(
            (np.full((anchor_dimensions.shape[0], 2), 0), 
            anchor_dimensions
        ), axis=-1), # [[0, 0, w, h], ...]

        self.data_augmentation = data_augmentation

        self.image_width, self.image_height = dimensions

        self.dataset_size = len(choices)
        self.choices = choices

        self.buffer = multiprocessing.Manager().Queue(maxsize=buffer_size)
        self.buffer_size = buffer_size

        self.grid_size = grid_size
        self.anchors = anchors
        self.classes = classes

        self.iou_ignore_threshold = iou_ignore_threshold
        self.local_buffer = deque(maxlen=self.batch_size * self.buffer_size)

        with open("annotations\\classes.txt", "r+") as file:
            self.class_names = file.read().splitlines()

        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}

        self.count = 0

        self.workers = 2  # or 6–8 depending on CPU

        threading.Thread(target=self.prefetch_to_local, daemon=True).start()

        for _ in range(self.workers):
            multiprocessing.Process(target=fill_buffer, args=(
                self.buffer,
                self.choices,
                augmentor,
                anchor_dimensions,
                self.formatted_anchor_dimensions,
                self.image_width,
                self.image_height,
                self.batch_size,
                self.grid_size,
                self.anchors,
                self.classes,
                self.class_names,
                self.class_to_idx,
                self.iou_ignore_threshold,
                self.data_augmentation
            ),
            daemon=True).start()

    def prefetch_to_local(self): 
        while True: 
            batch_xdata = []
            batch_ydata = []
            for _ in range(self.batch_size):
                image, ydata = self.buffer.get()
                batch_xdata.append(image)
                batch_ydata.append(ydata)
            self.local_buffer.append([batch_xdata, batch_ydata])

    def __call__(self):
        start_time = time.perf_counter()
        while len(self.local_buffer) == 0:
            time.sleep(0.001)
        end_time = time.perf_counter()
        print(f"[LOG] Buffer wait time: {end_time - start_time:.4f} seconds")

        start_time = time.perf_counter()

        batch_xdata, batch_ydata = self.local_buffer.popleft()

        end_time = time.perf_counter()
        print(f"[LOG] Batch retrieval time: {end_time - start_time:.4f} seconds")

        start_time = time.perf_counter()
        batch_ydata = [
            np.stack([y[i] for y in batch_ydata], axis=0)
            for i in range(3)
        ]
        end_time = time.perf_counter()
        print(f"[LOG] Batch stacking time: {end_time - start_time:.4f} seconds")

        return (
            np.array(batch_xdata, dtype=np.float32),
            tuple(np.array(y, dtype=np.float32) for y in batch_ydata)
        )

def fill_buffer(
        buffer,
        choices,
        augmentor,
        anchor_dimensions,
        formatted_anchor_dimensions,
        image_width,
        image_height,
        batch_size,
        grid_size,
        anchors,
        classes,
        class_names,
        class_to_idx,
        iou_ignore_threshold,
        data_augmentation
    ):
    all_filenames = np.array(os.listdir('Training'))

    while True:
        choices = np.random.choice(choices, size=batch_size, replace=False)
        filenames = all_filenames[choices]

        buffer_extension = []

        for idx, filename in enumerate(filenames):
            locations_filename = f'annotations\\{filename.replace(".png", ".txt").replace(".jpg", ".txt")}'

            if os.path.exists(locations_filename):
                with open(locations_filename, "r+") as file:
                    lines = file.read().splitlines()

                    if not lines:
                        continue

                    location_data = np.clip(np.array([
                        [
                            float(value) for value in line.split(' ')
                        ] for line in lines
                    ]), [[0, 0, 0, 0 ,0 ]], [[classes-1, 1, 1, 1, 1]])


            else:
                location_data = np.array([])

            img = cv2.imread(f'Training\\{filename}')
            if img is None:
                continue
            root_image = img[..., ::-1]
        

            class_labels = [class_names[int(bbox[0])] for bbox in location_data]

            # Properly define class_names for every bbox then intepret new classs from augmented_result['class_labels']

            if data_augmentation:
                try:
                    augmented_result = augmentor(image=root_image, bboxes=[np.clip(data[1:5], 0, 1) for data in location_data], class_labels=class_labels)
                    bboxes = np.array(augmented_result['bboxes'])
                    class_labels = augmented_result['class_labels']
                    image = cv2.resize(augmented_result['image'], (image_width, image_height))
                except ValueError as e:
                    print(location_data, filename, "[LOG] ValueError in augmentation")
                    print(f"[LOG] Error in augmentation for {filename}: {e}")
                    continue

            else:
                bboxes = np.array(location_data)
                image = cv2.resize(root_image, (image_width, image_height))

            class_ints = np.array([class_to_idx[label] for label in class_labels])

            image = (image / 255).astype(np.float32)

            ydata = [np.zeros((grid_size * 2 ** (2 - i), grid_size * 2 ** (2 - i), anchors, 5 + classes)) for i in range(3)]

            for class_int, (true_center_x, true_center_y, width, height) in zip(class_ints, bboxes):

                iou_values = Processing.iou(
                    np.array(formatted_anchor_dimensions),
                    np.array([0, 0, width, height]),
                    api=np
                )[0, :, 0]

                anchor_indices = iou_values.argsort()[::-1]

                has_anchor = [False, False, False]

                _classes = np.zeros((classes))
                _classes[int(class_int)] = 1

                for anchor_index in anchor_indices:
                    scale_idx = anchor_index // anchors
                    anchor_on_scale = anchor_index % anchors


                    _grid_size = grid_size * 2 ** (2-scale_idx)

                    grid_x_index = int(true_center_x * _grid_size)
                    grid_y_index = int(true_center_y * _grid_size)

                    relative_center_x = true_center_x * _grid_size - grid_x_index
                    relative_center_y = true_center_y * _grid_size - grid_y_index

                    # Adjust for grid sensitivity
                    relative_center_x = (relative_center_x + 0.5) / 2
                    relative_center_y = (relative_center_y + 0.5) / 2
            
                    # t_w and t_h calculation:
                    _width = 0.5 * np.sqrt(width / anchor_dimensions[anchor_index, 0])
                    _height = 0.5 * np.sqrt(height / anchor_dimensions[anchor_index, 1])

                    occupied = ydata[scale_idx][grid_y_index, grid_x_index, anchor_on_scale, 0]
                    if not occupied and not has_anchor[scale_idx]:
                        has_anchor[scale_idx] = True
                        ydata[scale_idx][grid_y_index, grid_x_index, anchor_on_scale] = np.concatenate(([1, relative_center_x, relative_center_y, _width, _height], _classes))

                    elif not occupied and iou_values[anchor_index] > iou_ignore_threshold:
                        ydata[scale_idx][grid_y_index, grid_x_index, anchor_on_scale, 0] = -1
            
            buffer.put((image, ydata))
            

class RandomScaledCenterCrop(A.CenterCrop):
    """
    Center crop with random scale.

    Args:
        min_scale (float): Minimum fraction of the smallest image dimension to crop.
        max_scale (float): Maximum fraction of the smallest image dimension to crop.
        pad_if_needed, pad_position, border_mode, fill, fill_mask, p:
            Same as CenterCrop.
    """

    def __init__(
        self,
        min_scale: float,
        max_scale: float,
        pad_if_needed: bool = False,
        pad_position: Literal[
            "center", "top_left", "top_right", "bottom_left", "bottom_right", "random"
        ] = "center",
        border_mode: int = cv2.BORDER_CONSTANT,
        fill: float | tuple[float, ...] = 0,
        fill_mask: float | tuple[float, ...] = 0,
        p: float = 1.0,
    ):
        super().__init__(height=1, width=1,  # placeholders, will override
                        pad_if_needed=pad_if_needed,
                        pad_position=pad_position,
                        border_mode=border_mode,
                        fill=fill,
                        fill_mask=fill_mask,
                        p=p)
        self.min_scale = min_scale
        self.max_scale = max_scale

    def get_params_dependent_on_data(
        self, params: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, Any]:
        # Get original image shape
        image_shape = params["shape"][:2]
        image_height, image_width = image_shape

        # Determine random crop size
        scale = random.uniform(self.min_scale, self.max_scale)
        crop_size = int(min(image_height, image_width) * scale)

        # Temporarily override height/width
        self.height = crop_size
        self.width = crop_size

        # Call original CenterCrop method
        return super().get_params_dependent_on_data(params, data)
    

def get_bboxes(choices):
    bboxes = np.empty((0, 5))
    with open("annotations\\classes.txt", "r+") as file:
        class_names = file.read().splitlines()

    for idx, filename in tqdm(list(enumerate(np.array(os.listdir('Training'))[choices]))):
        locations_filename = f'annotations\\{filename.replace(".png", ".txt").replace(".jpg", ".txt")}'

        if os.path.exists(locations_filename):
            with open(locations_filename, "r+") as file:
                lines = file.read().splitlines()
                if not lines:
                    continue
                location_data = np.clip(np.array([
                    [
                        float(value) for value in line.split(' ')
                    ] for line in lines
                ]), [[0, 0, 0, 0 ,0 ]], [[classes-1, 1, 1, 1, 1]])

        else:
            print("[LOG] Empty image detected")
            continue

        try:
            root_image = cv2.resize(
                            cv2.cvtColor(
                                    cv2.imread(f'Training\\{filename}'), 
                                    cv2.COLOR_BGR2RGB
                                ), (image_width, image_height)
                            )
        except Exception as e:
            print(filename, e, "[LOG] BROKEN")
            continue

        if (location_data.shape[0] != 0):
            class_labels = [class_names[int(bbox[0])] for bbox in location_data]
            location_data[:, 3] -= 1e-5
            _bboxes = np.array(location_data)
            # augmented_result = augmentor(image=root_image, bboxes=location_data[:, 1:5], class_labels=class_labels)
            # _bboxes = np.array(augmented_result['bboxes'])

            # class_labels = augmented_result['class_labels']
            # class_idx = np.array([class_names.index(label) for label in class_labels]).reshape(-1, 1)

            # _bboxes = np.hstack((class_idx, _bboxes.reshape((-1, 4))))

            bboxes = np.concatenate((bboxes, _bboxes))

    return bboxes

def iou_kmeans(boxes, centroid_count, stop_iter=100):
    boxes_count = boxes.shape[0]
    idxs = np.random.choice(boxes_count, centroid_count, replace=False)
    clusters = boxes[idxs]

    last_clusters = np.zeros(boxes_count)
    iteration = 0

    while True:
        
        formatted_clusters_dimensions = np.concatenate(
            (np.zeros((clusters.shape[0], 2)), 
            clusters,
        ), axis=-1), # [[0, 0, w, h], ...]

        formatted_boxes_dimensions = np.concatenate(
            (np.zeros((boxes.shape[0], 2)), 
            boxes,
        ), axis=-1), # [[0, 0, w, h], ...]

        ious = Processing.iou(
            np.array(formatted_boxes_dimensions),
            np.array(formatted_clusters_dimensions),
            api=np
        )[0]

        
        nearest_clusters = ious.argmax(axis=1)

        if (last_clusters == nearest_clusters).all():
            iteration += 1
            if iteration >= stop_iter:
                break
        else:
            iteration = 0

        for cluster_idx in range(centroid_count):
            if np.any(nearest_clusters == cluster_idx):
                clusters[cluster_idx] = boxes[nearest_clusters == cluster_idx].mean(axis=0)
        
        last_clusters = nearest_clusters.copy()

    return clusters, nearest_clusters

def get_anchor_data(grid_size, bboxes):

    # BBoxes Format: [[class_idx, x, y, w, h], ...]

    if len(bboxes) < 3 * anchors:
        print(np.tile(bboxes[0][3:5], (3 * anchors, 1)))
        return np.tile(bboxes[0][3:5], (3 * anchors, 1))

    classes = bboxes[:, 0]
    _, class_occurrences = np.unique(classes, return_counts=True)

    dimensions_count = anchors * yolo_head_count
    
    dimensions = bboxes[..., 3:5]
    aspect_ratio = np.maximum(dimensions[:, 0] / dimensions[:, 1], dimensions[:, 1] / dimensions[:, 0])

    # Remove outliers
    
    dimensions = dimensions[aspect_ratio < 6] 
    dimensions = dimensions[dimensions[:, 0] < 0.3] 
    dimensions = dimensions[dimensions[:, 1] < 0.8]
    dimensions = dimensions[dimensions[:, 1] > 0.05]
    dimensions = dimensions[dimensions[:, 0] * dimensions[:, 1] < 0.3] 
    dimensions = dimensions[dimensions[:, 0] * dimensions[:, 1] > 0.001]
    

    # kmeans = KMeans(n_clusters=dimensions_count)
    # kmeans.fit(dimensions)



    clusters = [np.empty((2)) for _ in range(dimensions_count)]

    plt.figure(figsize=(8, 8))

    anchor_dimensions, labels = iou_kmeans(dimensions, dimensions_count)

    for idx, (label, wh) in enumerate(zip(labels, dimensions)):
        clusters[label] = np.vstack((clusters[label], wh))

    clusters = [cluster[1:] for cluster in clusters] # Remove initial empty row

    order = np.argsort(anchor_dimensions[:, 0] * anchor_dimensions[:, 1])

    anchor_dimensions = anchor_dimensions[order] # Sort by metric
    clusters = [clusters[idx] for idx in order] # Sort by metric

    num_images = int(len(os.listdir('Training')) * training_percent)
    objects_per_scale = np.zeros(yolo_head_count)

    for idx, (anchor_dimension, cluster) in enumerate(zip(anchor_dimensions, clusters)):
        objects_per_scale[idx // anchors] += cluster.shape[0] / num_images

        formmated_anchor_dimension = np.concatenate(
            (np.zeros(2),
            anchor_dimension,
            ), axis=-1
        )

        formatted_cluster = np.concatenate(
            (np.zeros((cluster.shape[0], 2)),
            cluster
            ), axis=-1
        )

        iou_values = Processing.iou(
            np.array(formmated_anchor_dimension).reshape(1, 4),
            np.array(formatted_cluster),
            api=np
        )

        plt.scatter(
            cluster[:, 0], 
            cluster[:, 1], 
            alpha=0.6, 
            label = f"Group {idx} : {iou_values[0].mean()}"
        )


    plt.scatter(
        anchor_dimensions[:, 0], 
        anchor_dimensions[:, 1], 
        alpha=1,
        color='red',
        marker='x',
        s=100
    )

    print("[LOG] Anchor data:\n", anchor_dimensions)
    
    plt.title("Bounding Box Dimensions vs. Anchor Boxes", fontsize=14)
    plt.xlabel("Width", fontsize=12)
    plt.ylabel("Height", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()

    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.plot()
    plt.show()

    return anchor_dimensions, class_occurrences, objects_per_scale

def save():
    save_data = network.save()

    with open(save_file, 'wb') as file:
        file.write(pickle.dumps(save_data))

    
    spec = tf.TensorSpec([1, image_height, image_width, 3], tf.float16)

    graphed_forward = tf.function(lambda x: network.forward(x, training=False), input_signature=[spec])

    onnx_model, _ = tf2onnx.convert.from_function(
        graphed_forward,
        input_signature=[spec],
        opset=17,
        output_path="model-training-data.onnx"
    )

    with open("cost-overtime.json", "w+") as file:
        file.write(json.dumps(costs.tolist()))

def init_plot(yolo_head_count, titles):
    for i in range(yolo_head_count):
        row = []
        ax_row = []
        for j in range(len(titles)):
            ax = fig.add_subplot(yolo_head_count, len(titles),
                                 (i * len(titles)) + j + 1)
            (line,) = ax.plot([], [])  # empty line object
            ax_row.append(ax)
            row.append(line)
        lines.append(row)
        axes.append(ax_row)

def live_plot(costs_np, x_values, yolo_head_count, titles, colors, grid_size):
    for i in range(yolo_head_count):
        for j in range(len(titles)):
            ax = axes[i][j]
            line = lines[i][j]

            # Update the line data (VERY fast)
            line.set_data(x_values, costs_np[:, i, j])

            ax.relim()  # recompute data limits
            ax.autoscale_view()

            grid = grid_size * (2 ** (yolo_head_count - i - 1))
            ax.set_title(f"{grid}x{grid} ({titles[j]})")

            line.set_color(colors[j])

    
def conv(depth, kernel_shape, stride=1, padding="SAME"):
    return [
        Conv2d(depth=depth, kernel_shape=kernel_shape, stride=stride, padding=padding, batch_norm=BatchNorm(
            momentum=0.9,
            baked=True
        ), 
        activation_function=activation_function,
        weight_initializer=HeNormal(), bias_initializer=Fill(0)),
    ]

def res_block(filters):
    return ResidualBlock([
        *conv(filters, (1, 1), padding="SAME"),
        *conv(filters, (3, 3), padding="SAME")
    ])

def long_res_block(filters, repeats):
    block = []
    for _ in range(repeats):
        block.append(res_block(filters))

    return block

def csp_block(filters, repeats, residual=True):
    concat_start, residual_start, concat_end = Concat(external_concat=optimize_concats).generate_layers()
    return [
        concat_start, # 1 
            *conv(filters, (1, 1), padding="SAME"), # 2
            *(long_res_block(filters, repeats) if residual else [ # 4
                    *conv(filters, (1, 1), padding="SAME"), 
                    *conv(filters, (3, 3), padding="SAME")
                ]),

        residual_start, # 5
            *conv(filters, (1, 1), padding="SAME"), # 6

        concat_end, # 7
    ] # 7 layers

def sppf():
    global scale
    concat_start1, residual_start1, concat_end1 = Concat(external_concat=optimize_concats).generate_layers()
    concat_start2, residual_start2, concat_end2 = Concat(external_concat=optimize_concats).generate_layers()
    concat_start3, residual_start3, concat_end3 = Concat(external_concat=optimize_concats).generate_layers()

    return [
        
        *conv(int(512 * scale), (1, 1), padding="SAME"),
        concat_start1,
            MaxPool((5,5), pooling_stride=(1, 1), padding="SAME"),

            concat_start2,
                MaxPool((5,5), pooling_stride=(1, 1), padding="SAME"),

                concat_start3,
                    MaxPool((5,5), pooling_stride=(1, 1), padding="SAME"),

                residual_start3,
                concat_end3,

            residual_start2,
            concat_end2,

        residual_start1,
        concat_end1,

        *conv(int(1024 * scale), (1,1), padding="SAME")
    ]

if __name__ == "__main__":
    training_percent = 0.975
    batch_size = 32
    accumulate = 1

    image_width, image_height = [384, 384]
    yolo_head_count = 3
    
    grid_size = int(image_width / 32)
    grid_count = grid_size ** 2

    anchors = 3
    classes = 3

    yolo_size = 2 # 1: small, 2: medium, 3: large

    scale = 0.5 + (0.25) * (yolo_size - 1)
    depth_mult = yolo_size / 3

    def R(x):
        return max(1, np.int32(np.ceil(x * depth_mult)))

    dropout_rate = 0
    activation_function = Mish()
    optimize_concats = True
    variance = "He"
    dtype = np.float16

    save_file = 'model-training-data.json'
    dataset_size = int(len(os.listdir('Training')) * training_percent)
    choices = np.random.choice(dataset_size, size=dataset_size, replace=False)

    with open('training-files.json', 'w+') as file:
        file.write(json.dumps(choices.tolist()))

    

    augmentor = A.Compose([
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.75),
        A.RandomBrightnessContrast(
            brightness_limit=[-0.07, 0.07],
            contrast_limit=[-0.07, 0.07],
            p=0.5
        ),
        A.HueSaturationValue(
            hue_shift_limit=5,
            sat_shift_limit=5,
            val_shift_limit=5,
            p=0.5
        ),

        RandomScaledCenterCrop(
            min_scale=0.15, max_scale=0.4,
        ),
    ], bbox_params=A.BboxParams(format='yolo', min_visibility=0.1, label_fields=['class_labels']))

    bboxes = get_bboxes(choices)
    anchor_dimensions, class_occurences, objects_per_scale = get_anchor_data(grid_size, bboxes)
    del bboxes

    median_dimension = np.mean(anchor_dimensions[anchor_dimensions.shape[0] // 2])

    concat_start1, residual_start1, concat_end1 = Concat(external_concat=optimize_concats).generate_layers()
    concat_start2, residual_start2, concat_end2 = Concat(external_concat=optimize_concats).generate_layers()
    concat_start3, residual_start3, concat_end3 = Concat(external_concat=optimize_concats).generate_layers()
    concat_start4, residual_start4, concat_end4 = Concat(external_concat=optimize_concats).generate_layers()

    weight_initializer = YoloSplit(presence_initializer=LecunNormal(), xy_initializer=HeNormal(), dimensions_initializer=LecunNormal(), class_initializer=HeNormal(), classes=classes, anchors=3)
    bias_initializer = YoloSplit(presence_initializer=Fill(-5), xy_initializer=Fill(0), dimensions_initializer=Fill(0), class_initializer=Fill(0), classes=classes, anchors=3)

    model = [
        Input((image_height, image_width, 3)),

        *conv(int(64 * scale), (6, 6), stride=2, padding="SAME"),
        *conv(int(128 * scale), (3, 3), stride=2, padding="SAME"),

        *csp_block(int(64 * scale), R(3)),
        *conv(int(128 * scale), (1, 1), stride=1, padding="SAME"),

        *conv(int(256 * scale), (3, 3), stride=2, padding="SAME"),

        *csp_block(int(128 * scale), 6, R(6)),
        *conv(int(256 * scale), (1, 1), stride=1, padding="SAME"),

        concat_start1,

            *conv(int(512 * scale), (3, 3), stride=2, padding="SAME"),

            *csp_block(int(256 * scale), R(9)),
            *conv(int(512 * scale), (1, 1), stride=1, padding="SAME"),

            concat_start2,

                *conv(int(1024 * scale), (3, 3), stride=2, padding="SAME"),

                *csp_block(int(512 * scale), R(3)),
                *conv(int(1024 * scale), (1, 1), stride=1, padding="SAME"),

                *sppf(),

                *conv(int(512 * scale), (1, 1), padding="SAME"),

                concat_start4,
                Upsample(2),

            residual_start2,
            concat_end2,

            *csp_block(int(512 * scale), R(3), residual=False),
            *conv(int(512 * scale), (1, 1), stride=1, padding="SAME"),
            
            *conv(int(256 * scale), (1, 1), stride=1, padding="SAME"),
            concat_start3,

            Upsample(2),

        residual_start1,
        concat_end1,

        *csp_block(int(256 * scale), R(3), residual=False),
        *conv(int(256 * scale), (1, 1), stride=1, padding="SAME"), # ROUTE 1 (23 layers)
         
        *conv(int(256 * scale), (3, 3), stride=2, padding="SAME"), # 22 layers

        residual_start3, # 21 layers
        concat_end3, # 20 layers

        *csp_block(int(256 * scale), R(3), residual=False), # 19 layers (7 layers)
        *conv(int(512 * scale), (1, 1), stride=1, padding="SAME"), # ROUTE 2 (12 layers)

        *conv(int(512 * scale), (3, 3), stride=2, padding="SAME"), # 11 layers

        residual_start4, # 10
        concat_end4, # 9

        *csp_block(int(512 * scale), 3, residual=False), # 8 layers (10 layers)
        *conv(int(1024 * scale), (1, 1), stride=1, padding="SAME") # ROUTE 3 (1 layers)
    ]

    backprop_layer_indices = [
        -23,
        -12,
        -1,
    ]

    addon_layers = [
        [
            Conv2d(anchors * (5 + classes), (1, 1), padding="VALID", weight_initializer=weight_initializer, bias_initializer=bias_initializer),
            Activation(YoloActivation(classes=classes, dtype=dtype)),
            Reshape((-1, grid_size * 4, grid_size * 4, anchors, 5 + classes))
        ],
        [
            Conv2d(anchors * (5 + classes), (1, 1), padding="VALID", weight_initializer=weight_initializer, bias_initializer=bias_initializer),
            Activation(YoloActivation(classes=classes, dtype=dtype)),
            Reshape((-1, grid_size * 2, grid_size * 2, anchors, 5 + classes))
        ],
        [
            Conv2d(anchors * (5 + classes), (1, 1), padding="VALID", weight_initializer=weight_initializer, bias_initializer=bias_initializer),
            Activation(YoloActivation(classes=classes, dtype=dtype)),
            Reshape((-1, grid_size * 1, grid_size * 1, anchors, 5 + classes))
        ],
    ]

    cooridnate_weight = 5
    no_object_weight = 5
    object_weight = 1
    
    # Fundamental theory:
    # Both no_object_loss and object_loss will initially have the same goal of low presence scores
    # But as the coordinate_loss optimizes the object_loss will eventually want high presence scores to accomodate for coordinate loss
    # To give the no_obj_loss time to converge we lower coordinate weight so that the object_loss doesn't immediately conflict with no_object_loss
    # Then object_loss can gradually take over as coordinate_loss converges
    # Contraction loss also starts to conflict with coordinate_loss when it's too high
    # So we keep it low to allow coordinate_loss to converge first
    
    inv_freqs = 1.0 / (class_occurences / min(class_occurences))
    alpha = inv_freqs / inv_freqs.sum()

    print("Weights:", alpha)
    FL = FocalLoss(gamma=2.0, alpha=alpha)

    print("Objects Per Scale:", objects_per_scale)


    network = Network(
        model=model,
        addon_layers=addon_layers,
        backprop_layer_indices=backprop_layer_indices,
        loss_function = [
            YoloLoss(contraction_weight=3e-6, coordinate_loss_function=DIoU, objectness_loss_function=BCE, class_loss_function=FL, grid_size=grid_size, anchors=anchors, classes=classes, coordinate_weight=cooridnate_weight, no_object_weight=no_object_weight, object_weight=object_weight, anchor_dimensions=anchor_dimensions, dtype=dtype),
            YoloLoss(contraction_weight=3e-6, coordinate_loss_function=DIoU, objectness_loss_function=BCE, class_loss_function=FL, grid_size=grid_size, anchors=anchors, classes=classes, coordinate_weight=cooridnate_weight, no_object_weight=no_object_weight , object_weight=object_weight, anchor_dimensions=anchor_dimensions, dtype=dtype),
            YoloLoss(contraction_weight=3e-6, coordinate_loss_function=DIoU, objectness_loss_function=BCE, class_loss_function=FL, grid_size=grid_size, anchors=anchors, classes=classes, coordinate_weight=cooridnate_weight, no_object_weight=no_object_weight, object_weight=object_weight, anchor_dimensions=anchor_dimensions, dtype=dtype),
        ],
        # loss_function = YoloLoss(grid_size=grid_size, anchors=anchors, coordinate_weight=5, no_object_weight=no_object_weight, object_weight=1),
        optimizer = Adam(momentum = 0.8,  beta_constant = 0.9, weight_decay=2e-5), 
        # optimizer = RMSProp(beta_constant = 0.9),
        # optimizer = Momentum(momentum=0.9),
        scheduler = StepLR(initial_learning_rate=0.0001, decay_rate=0.6, decay_interval=50), 
        optimize_concats=optimize_concats,
        # scheduler=CosineAnnealingDecay(initial_learning_rate=0.001, min_learning_rate=0.00003, initial_cycle_size=15, cycle_mult=2),
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

    titles = ['object_loss', 'no_object_loss', 'coordinate_loss', 'class_loss', 'contraction_loss']
    colors = ['C0', 'C1', 'C2', 'C3', 'C4']

    generator = Generate(batch_size, anchor_dimensions, (image_width, image_height), grid_size, anchors, classes, choices, 16, data_augmentation=True)
    
    plt.ion()
    fig = plt.figure(figsize=(16, 6))

    lines = []   # line objects
    axes = []    # axes for each subplot

    init_plot(yolo_head_count, titles)

    config = {
        "image_width": int(image_width),
        "image_height": int(image_height),
        "grid_size": int(grid_size),
        "anchors": int(anchors),
        "anchor_dimensions": anchor_dimensions.tolist(),
        "classes": int(classes)
    }

    with open("model-config.json", "w") as json_file:
        json.dump(config, json_file)


    for idx, cost in enumerate(network.fit(generator=generator, batch_size=batch_size, accumulate=accumulate, epochs = 2000000000, gradient_transformer=AutoClipper(5))):

        print(cost)

        try:
            cost = np.stack(cost, axis=0)

            if not len(costs):
                costs = cost[None, ...]
            else:
                costs = np.vstack([costs, cost[None, ...]])

            if not idx % 100:
                steps = costs.shape[0]
                x_values = np.arange(steps) * (batch_size / (dataset_size * accumulate))
                print("PREPLOT")
                live_plot(costs, x_values, yolo_head_count, titles, colors, grid_size)
                print("POSTPLOT")

                plt.legend()
                fig.canvas.draw()
                fig.canvas.flush_events()
                fig.legend()
                plt.pause(0.001)
                print("POST DRAW")

            if not idx % 30 and not np.isnan(cost).any():
                save()

                print("PASSED SAVE")
        except Exception as e:
            now = datetime.now()
            date_string = now.strftime("%H:%M:%S")

            print(f"[LOG {date_string}] Iteration: {idx} Error: {e} ")

    else:
        print("LOOP EXITED")
