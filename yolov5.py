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
    def __init__(self, batch_size, anchor_dimensions, dimensions, grid_size, anchors, classes, choices, iou_ignore_threshold = 0.72, data_augmentation=True):
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
        self.classes = classes

        self.iou_ignore_threshold = iou_ignore_threshold

        multiprocessing.Process(target=self.fill_buffer).start()

    def fill_buffer(self):
        augmentor = A.Compose([
            A.HorizontalFlip(p=0.5),
            
            A.OneOf([
                A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.75),
                A.RandomBrightnessContrast(
                    brightness_limit=0.5,
                    contrast_limit=0.5,
                    p=0.75
                )
            ], p=0.3),

            A.Perspective(scale=(0.05, 0.1), p=0.3),
            A.Affine( 
                rotate=(-2, 2),
                translate_percent=(-0.03, 0.03),
                rotate_method="largest_box",
                keep_ratio=True,
                balanced_scale=True,
                p=0.3,
                border_mode=cv2.BORDER_CONSTANT
            ),

            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.25),
            
            A.RandomResizedCrop(
                size=[self.image_width, self.image_height],
                scale=[0.4, 0.85],
                ratio=[1, 1],
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                p=1
            ),

        ], bbox_params=A.BboxParams(format='yolo', min_visibility=0.25, label_fields=['class_labels']))
        all_filenames = np.array(os.listdir('Training'))
        
        while True:
            if len(self.buffer) >= self.buffer_size:
                continue

            choices = np.random.choice(self.choices, size=self.buffer_size - len(self.buffer), replace=False)
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
                        ]), [[0, 0, 0, 0 ,0 ]], [[self.classes-1, 1, 1, 1, 1]])


                else:
                    location_data = np.array([])

                try:
                    root_image = cv2.resize(
                                    cv2.cvtColor(
                                            cv2.imread(f'Training\\{filename}'), 
                                            cv2.COLOR_BGR2RGB
                                        ), (self.image_width, self.image_height)
                                    )
                except Exception as e:
                    print(filename, e, "[LOG] BROKEN")
                    continue
                
                with open("annotations\\classes.txt", "r+") as file:
                    class_names = file.read().splitlines()

                class_labels = [class_names[int(bbox[0])] for bbox in location_data]

                # Properly define class_names for every bbox then intepret new classs from augmented_result['class_labels']

                if self.data_augmentation:
                    try:
                        augmented_result = augmentor(image=root_image, bboxes=[np.clip(data[1:5], 0, 1) for data in location_data], class_labels=class_labels)
                        bboxes = np.array(augmented_result['bboxes'])
                        class_labels = augmented_result['class_labels']
                        image = augmented_result['image']
                    except ValueError as e:
                        print(location_data, filename, "[LOG] ValueError in augmentation")
                        print(f"[LOG] Error in augmentation for {filename}: {e}")
                        continue

                else:
                    bboxes = np.array(location_data)
                    image = root_image

                class_ints = np.array([class_names.index(label) for label in class_labels])

                image = image / 255

                ydata = [np.zeros((self.grid_size * 2 ** (2 - i), self.grid_size * 2 ** (2 - i), self.anchors, 5 + self.classes)) for i in range(3)]

                for class_int, (true_center_x, true_center_y, width, height) in zip(class_ints, bboxes):

                    formatted_anchor_dimensions = np.concatenate(
                        (np.full((self.anchor_dimensions.shape[0], 2), 0), 
                        self.anchor_dimensions
                    ), axis=-1), # [[0, 0, w, h], ...]

                    iou_values = Processing.iou(
                        np.array(formatted_anchor_dimensions),
                        np.array([0, 0, width, height]),
                        api=np
                    )[0, :, 0]

                    anchor_indices = iou_values.argsort()[::-1]

                    has_anchor = [False, False, False]

                    classes = np.zeros((self.classes))
                    classes[int(class_int)] = 1

                    for anchor_index in anchor_indices:
                        scale_idx = anchor_index // self.anchors
                        anchor_on_scale = anchor_index % self.anchors


                        _grid_size = self.grid_size * 2 ** (2-scale_idx)

                        grid_x_index = int(true_center_x * _grid_size)
                        grid_y_index = int(true_center_y * _grid_size)

                        relative_center_x = true_center_x * _grid_size - grid_x_index
                        relative_center_y = true_center_y * _grid_size - grid_y_index

                        # Adjust for grid sensitivity
                        relative_center_x = (relative_center_x + 0.5) / 2
                        relative_center_y = (relative_center_y + 0.5) / 2
                
                        # t_w and t_h calculation:
                        _width = 0.5 * np.sqrt(width / self.anchor_dimensions[anchor_index, 0])
                        _height = 0.5 * np.sqrt(height / self.anchor_dimensions[anchor_index, 1])

                        occupied = ydata[scale_idx][grid_y_index, grid_x_index, anchor_on_scale, 0]
                        if not occupied and not has_anchor[scale_idx]:
                            has_anchor[scale_idx] = True
                            ydata[scale_idx][grid_y_index, grid_x_index, anchor_on_scale] = np.concatenate(([1, relative_center_x, relative_center_y, _width, _height], classes))

                        elif not occupied and iou_values[anchor_index] > self.iou_ignore_threshold:
                            ydata[scale_idx][grid_y_index, grid_x_index, anchor_on_scale, 0] = -1
                    
                ydata = [scale.reshape(self.grid_size * 2 ** (2 - i), self.grid_size * 2 ** (2 - i), self.anchors, (5 + self.classes)) for i, scale in enumerate(ydata)]
                buffer_extension.append((image, ydata))
            try:
                self.buffer += buffer_extension
            except:
                continue

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
    dimensions = dimensions[dimensions[:, 0] > 0.005] 
    dimensions = dimensions[dimensions[:, 1] < 0.9]
    dimensions = dimensions[dimensions[:, 0] * dimensions[:, 1] < 0.5] 
    dimensions = dimensions[dimensions[:, 0] * dimensions[:, 1] > 0.001]
    

    # kmeans = KMeans(n_clusters=dimensions_count)
    # kmeans.fit(dimensions)



    clusters = [np.empty((2)) for _ in range(dimensions_count)]

    plt.figure(figsize=(8, 8))

    anchor_dimensions, labels = iou_kmeans(dimensions, dimensions_count)
    print(anchor_dimensions, labels)

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
    
    plt.title("Bounding Box Dimensions vs. Anchor Boxes", fontsize=14)
    plt.xlabel("Width", fontsize=12)
    plt.ylabel("Height", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()

    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.plot()
    plt.show()

    print("[LOG] Anchor data:\n", anchor_dimensions)

    return anchor_dimensions, class_occurrences, objects_per_scale

def save():
    
    save_data = network.save()
    
    with open(save_file, 'wb') as file:
        file.write(pickle.dumps(save_data))

    with open("cost-overtime.json", "w+") as file:
        file.write(json.dumps(costs))

def plot_worker(queue, yolo_head_count, grid_size, batch_size, accumulate, dataset_size, titles, colors):
    plt.ion()
    fig = plt.figure(figsize=(16, 6))
    costs = []

    while True:
        try:
            idx, new_cost = queue.get(timeout=1)
            costs.append(new_cost)
            
            plt.clf()

            for i in range(yolo_head_count):
                for j in range(len(titles)):
                    plt.subplot(yolo_head_count, len(titles), (i * len(titles)) + j + 1)
                    data = np.array(costs)[:, i, j]
                    plt.plot(np.arange(len(costs)) * (batch_size / (dataset_size * accumulate)), data, colors[j], label=titles[j])
                    plt.xscale('linear')
                    plt.yscale('linear')

                    ax = plt.gca()
                    ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
                    ax.yaxis.set_major_locator(MaxNLocator(nbins=10))
                    plt.title(f"{grid_size * 2 ** (yolo_head_count - i - 1)}x{grid_size * 2 ** (yolo_head_count - i - 1)} ({titles[j]})")

            plt.tight_layout()
            plt.draw()
            plt.pause(0.001)
        except multiprocessing.queues.Empty:
            continue
        except Exception as e:
            print("Plot error:", e)

def conv(depth, kernel_shape, stride=1, padding="SAME"):
    return [
        Conv2d(depth=depth, kernel_shape=kernel_shape, stride=stride, padding=padding),
        BatchNorm(momentum=0.99),
        Activation(activation_function)
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
    concat_start, residual_start, concat_end = Concat().generate_layers()
    return [
        concat_start,
            *conv(filters, (1, 1), padding="SAME"),
            *(long_res_block(filters, repeats) if residual else [
                    *conv(filters, (1, 1), padding="SAME"), 
                    *conv(filters, (3, 3), padding="SAME")
                ]),

        residual_start,
            *conv(filters, (1, 1), padding="SAME"),

        concat_end,
    ] # 15 layers

def sppf():
    concat_start1, residual_start1, concat_end1 = Concat().generate_layers()
    concat_start2, residual_start2, concat_end2 = Concat().generate_layers()
    concat_start3, residual_start3, concat_end3 = Concat().generate_layers()

    return [
        
        *conv(512, (1, 1), padding="SAME"),
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

        *conv(1024, (1,1), padding="SAME")
    ]

if __name__ == "__main__":
    training_percent = 1700/1804
    batch_size = 16
    accumulate = 1

    image_width, image_height = [512, 512]
    yolo_head_count = 3
    
    grid_size = int(image_width / 32)
    grid_count = grid_size ** 2

    anchors = 3
    classes = 3

    dropout_rate = 0
    activation_function = Silu()
    variance = "He"
    dtype = np.float16

    save_file = 'model-training-data.json'
    dataset_size = len(os.listdir('Training'))
    choices = np.random.choice(dataset_size, size=int(dataset_size * training_percent), replace=False)

    with open('training-files.json', 'w+') as file:
        file.write(json.dumps(choices.tolist()))

    augmentor = A.Compose([
        A.HorizontalFlip(p=0.5),
        
        A.OneOf([
            A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.75),
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.75
            ),
            A.ToGray(p=0.5),
        ], p=0.3),
        
        A.OneOf([
            A.MedianBlur(blur_limit=3, p=0.75),
            A.Blur(blur_limit=3, p=0.75),
        ], p=0.3),

        A.Perspective(scale=(0.05, 0.1), p=0.3),
        A.Affine( 
            rotate=(-3, 3),
            translate_percent=(-0.03, 0.03),
            rotate_method="largest_box",
            keep_ratio=True,
            balanced_scale=True,
            p=0.3,
            border_mode=cv2.BORDER_CONSTANT
        ),

        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
        
        A.RandomResizedCrop(
            size=[image_width, image_height],
            scale=[0.6, 1],
            ratio=[1, 1],
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST,
            p=0.3
        ),

        A.CLAHE(clip_limit=(1, 4), tile_grid_size=(8, 8), p=0.05)

    ], bbox_params=A.BboxParams(format='yolo', min_visibility=0.25, label_fields=['class_labels']))
    
    bboxes = get_bboxes(choices)
    anchor_dimensions, class_occurences, objects_per_scale = get_anchor_data(grid_size, bboxes)
    del bboxes

    median_dimension = np.mean(anchor_dimensions[anchor_dimensions.shape[0] // 2])

    concat_start1, residual_start1, concat_end1 = Concat().generate_layers()
    concat_start2, residual_start2, concat_end2 = Concat().generate_layers()
    concat_start3, residual_start3, concat_end3 = Concat().generate_layers()
    concat_start4, residual_start4, concat_end4 = Concat().generate_layers()

    weight_initializer = YoloSplit(presence_initializer=HeNormal(), xy_initializer=HeNormal(), dimensions_initializer=HeNormal(), class_initializer=HeNormal(), classes=classes, anchors=3)
    bias_initializer = YoloSplit(presence_initializer=Fill(np.log(0.05)), xy_initializer=Fill(0), dimensions_initializer=Fill(np.log(0.75)), class_initializer=Fill(0), classes=classes, anchors=3)

    model = [
        Input((image_height, image_width, 3)),

        *conv(64, (6, 6), stride=2, padding="SAME"),
        *conv(128, (3, 3), stride=2, padding="SAME"),

        *csp_block(64, 3),
        *conv(128, (1, 1), stride=1, padding="SAME"),

        *conv(256, (3, 3), stride=2, padding="SAME"),

        *csp_block(128, 6),
        *conv(256, (1, 1), stride=1, padding="SAME"),

        concat_start1,

            *conv(512, (3, 3), stride=2, padding="SAME"),

            *csp_block(256, 9),
            *conv(512, (1, 1), stride=1, padding="SAME"),

            concat_start2,

                *conv(1024, (3, 3), stride=2, padding="SAME"),

                *csp_block(512, 3),
                *conv(1024, (1, 1), stride=1, padding="SAME"),

                *sppf(),

                *conv(512, (1, 1), padding="SAME"),

                concat_start4,
                Upsample(2),

            residual_start2,
            concat_end2,

            *csp_block(512, 3, residual=False),
            *conv(512, (1, 1), stride=1, padding="SAME"),
            
            *conv(256, (1, 1), stride=1, padding="SAME"),
            concat_start3,

            Upsample(2),

        residual_start1,
        concat_end1,

        *csp_block(256, 3, residual=False),
        *conv(256, (1, 1), stride=1, padding="SAME"), # ROUTE 1
         
        *conv(256, (3, 3), stride=2, padding="SAME"), # 46 layers

        residual_start3, # 43 layers
        concat_end3, # 42 layers

        *csp_block(256, 3, residual=False), # 41 layers
        *conv(512, (1, 1), stride=1, padding="SAME"), # ROUTE 2 (26 layers)

        *conv(512, (3, 3), stride=2, padding="SAME"), # 3 layers (23 layers)

        residual_start4, # 20
        concat_end4, # 19

        *csp_block(512, 3, residual=False), # 18 layers
        *conv(1024, (1, 1), stride=1, padding="SAME") # ROUTE 3 (3 layers)
    ]

    backprop_layer_indices = [
        -47,
        -24,
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

    cooridnate_weight = 10
    no_object_weight = 5
    object_weight = 1

    
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
            YoloLoss(contraction_weight=1e-5, coordinate_loss_function=CIoU, objectness_loss_function=BCE, class_loss_function=FL, grid_size=grid_size, anchors=anchors, classes=classes, coordinate_weight=cooridnate_weight * 1, no_object_weight=no_object_weight / 1, object_weight=object_weight, anchor_dimensions=anchor_dimensions, dtype=dtype),
            YoloLoss(contraction_weight=1e-5, coordinate_loss_function=CIoU, objectness_loss_function=BCE, class_loss_function=FL, grid_size=grid_size, anchors=anchors, classes=classes, coordinate_weight=cooridnate_weight * 2, no_object_weight=no_object_weight / 2, object_weight=object_weight, anchor_dimensions=anchor_dimensions, dtype=dtype),
            YoloLoss(contraction_weight=1e-5, coordinate_loss_function=CIoU, objectness_loss_function=BCE, class_loss_function=FL, grid_size=grid_size, anchors=anchors, classes=classes, coordinate_weight=cooridnate_weight * 2, no_object_weight=no_object_weight / 100, object_weight=object_weight * 4, anchor_dimensions=anchor_dimensions, dtype=dtype),
        ],
        # loss_function = YoloLoss(grid_size=grid_size, anchors=anchors, coordinate_weight=5, no_object_weight=no_object_weight, object_weight=1),
        optimizer = Adam(momentum = 0.8, beta_constant = 0.9, weight_decay=0), 
        # optimizer = RMSProp(beta_constant = 0.9),
        # optimizer = Momentum(momentum=0.9),
        scheduler = StepLR(initial_learning_rate=0.0001, decay_rate=0.5, decay_interval=120), 
        # scheduler=CosineAnnealingDecay(initial_learning_rate=0.001, min_learning_rate=0.00003, initial_cycle_size=40, cycle_mult=2),
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

    generator = Generate(batch_size, anchor_dimensions, (image_width, image_height), grid_size, anchors, classes, choices, data_augmentation=True)
    dataset_size = generator.dataset_size
    
    queue = multiprocessing.Queue()

    plot_proc = multiprocessing.Process(
        target=plot_worker,
        args=(queue, yolo_head_count, grid_size, batch_size, accumulate, dataset_size, titles, colors)
    )
    plot_proc.start()

    for idx, cost in enumerate(network.fit(generator=generator, batch_size=batch_size, learning_rate=0.05, accumulate=accumulate, epochs = 20000000)):

        print(cost)

        costs.append(cost)
        plt.clf()

        try:
           queue.put((idx, cost))

        except Exception as e:
            print(e)

        if not idx % 5 and not np.isnan(cost).any():
            threading.Thread(target=save).start()
