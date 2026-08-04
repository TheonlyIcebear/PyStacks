from utils.layers import *
from utils.schedulers import *
from utils.network import Network
from utils.initializers import *
from utils.loss import *
from utils.activations import *
from datetime import datetime
from itertools import islice
from sklearn.cluster import KMeans
from sklearn.metrics import average_precision_score
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
import onnx
from onnxsim import simplify


class Generate:
    def __init__(self, batch_size, anchor_dimensions, dimensions, grid_size, anchors, classes, choices, buffer, buffer_size, iou_ignore_threshold=0.8, augmentor=None, mosaic_augmentor=None, data_augmentation=True):
        self.augmentor = augmentor
        self.mosaic_augmentor = mosaic_augmentor
        self.batch_size = batch_size

        anchor_dimensions = anchor_dimensions

        self.data_augmentation = data_augmentation
        self.image_width, self.image_height = dimensions
        self.dataset_size = len(choices)
        self.choices = choices
        self.buffer = buffer
        self.buffer_size = buffer_size
        self.grid_size = grid_size
        self.anchors = anchors
        self.classes = classes
        self.iou_ignore_threshold = iou_ignore_threshold

        self.local_buffer = Queue(maxsize=buffer_size)

        with open("annotations\\classes.txt", "r+") as file:
            self.class_names = file.read().splitlines()
        
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}

        self.workers = 2

        for _ in range(self.workers):
            threading.Thread(target=self.prefetch_to_local).start()

        for _ in range(self.workers):
            multiprocessing.Process(target=fill_buffer, args=(
                buffer, 
                choices, 
                augmentor, 
                mosaic_augmentor,
                anchor_dimensions,
                image_width, 
                image_height, 
                batch_size, 
                grid_size, 
                anchors,
                classes, 
                self.class_names, 
                self.class_to_idx,
                iou_ignore_threshold, 
                data_augmentation
            )).start()

    def prefetch_to_local(self):
        while True:
            batch_xdata, batch_ydata = [], []
            for _ in range(self.batch_size):
                image, ydata = self.buffer.get()
                batch_xdata.append(image)
                batch_ydata.append(ydata)

            stacked_x = np.array(batch_xdata, dtype=np.float32)
            stacked_y = tuple(
                np.stack([y[i] for y in batch_ydata], axis=0)
                for i in range(3)
            )
            self.local_buffer.put([stacked_x, stacked_y])

    def __call__(self):
        start_time = time.perf_counter()
        batch = self.local_buffer.get()
        print(f"[LOG] Buffer wait time: {time.perf_counter() - start_time:.4f}s")
        return batch

def mosaic(filenames, image_width, image_height, class_names, class_to_idx, mosaic_augmentor):
    chosen = np.random.choice(len(filenames), size=4, replace=False)
    mosaic_filenames = filenames[chosen]

    canvas_w, canvas_h = image_width * 2, image_height * 2
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    all_bboxes, all_class_labels = [], []
    positions = [(0, 0), (image_width, 0), (0, image_height), (image_width, image_height)]

    for filename, (px, py) in zip(mosaic_filenames, positions):
        img = cv2.imread(f'Training\\{filename}')
        if img is None:
            continue
        img = img[..., ::-1]

        locations_filename = f'annotations\\{filename.replace(".png", ".txt").replace(".jpg", ".txt")}'
        if os.path.exists(locations_filename):
            with open(locations_filename, 'r') as f:
                lines = f.read().splitlines()
            if lines:
                location_data = np.array([[float(x) for x in l.split()] for l in lines])
                bboxes = [bbox[1:5] for bbox in location_data]
                labels = [class_names[int(bbox[0])] for bbox in location_data]
            else:
                bboxes, labels = [], []
        else:
            bboxes, labels = [], []

        bboxes = np.clip(bboxes, 0, 1)

        try:
            augmented = mosaic_augmentor(image=img, bboxes=bboxes, class_labels=labels)
            img = cv2.resize(augmented['image'], (image_width, image_height))
            bboxes = augmented['bboxes']
            labels = augmented['class_labels']
        except Exception as e:
            img = cv2.resize(img, (image_width, image_height))
            print(f"[LOG] Mosaic augmentation error: {e}")

        canvas[py:py+image_height, px:px+image_width] = img

        for bbox, label in zip(bboxes, labels):
            cx, cy, w, h = bbox
            all_bboxes.append([
                (px + cx * image_width) / canvas_w,
                (py + cy * image_height) / canvas_h,
                w / 2,
                h / 2,
            ])
            all_class_labels.append(label)

    crop_factor = 1.5

    cx_c = canvas_w // 2 + np.random.randint(-image_width // 4, image_width // 4)
    cy_c = canvas_h // 2 + np.random.randint(-image_height // 4, image_height // 4)
    crop_w = int(image_width * crop_factor)
    crop_h = int(image_height * crop_factor)

    x1 = cx_c - crop_w // 2
    y1 = cy_c - crop_h // 2

    x1 = np.clip(x1, 0, canvas_w - crop_w)
    y1 = np.clip(y1, 0, canvas_h - crop_h)

    x2 = x1 + crop_w
    y2 = y1 + crop_h

    canvas = canvas[y1:y2, x1:x2]
    canvas = cv2.resize(canvas, (image_width, image_height))

    adjusted_bboxes, adjusted_labels = [], []

    for bbox, label in zip(all_bboxes, all_class_labels):
        cx_n, cy_n, w_n, h_n = bbox

        cx_px = cx_n * canvas_w - x1
        cy_px = cy_n * canvas_h - y1

        w_px = w_n * canvas_w
        h_px = h_n * canvas_h

        bx1 = max(0, cx_px - w_px / 2)
        by1 = max(0, cy_px - h_px / 2)

        bx2 = min(crop_w, cx_px + w_px / 2)
        by2 = min(crop_h, cy_px + h_px / 2)

        if bx2 - bx1 < 2 or by2 - by1 < 2:
            continue

        new_cx = ((bx1 + bx2) / 2) / crop_w
        new_cy = ((by1 + by2) / 2) / crop_h

        new_w = (bx2 - bx1) / crop_w
        new_h = (by2 - by1) / crop_h

        if (
            0 < new_cx < 1
            and 0 < new_cy < 1
            and new_w > 0.01
            and new_h > 0.01
        ):
            adjusted_bboxes.append([new_cx, new_cy, new_w, new_h])
            adjusted_labels.append(label)

    return canvas, np.array(adjusted_bboxes), adjusted_labels

def fill_buffer(
        buffer,
        choices,
        augmentor,
        mosaic_augmentor,
        anchor_dimensions,
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
    formatted_anchor_dimensions = np.concatenate(
        (np.zeros((anchor_dimensions.shape[0], 2)), anchor_dimensions),
        axis=-1
    )  # [[0, 0, w, h], ...] per anchor

    while True:
        try:
            batch_indices = np.random.choice(choices, size=batch_size, replace=False)
            batch_filenames = all_filenames[batch_indices]
            buffer_extension = []

            for idx, filename in enumerate(batch_filenames):
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

                if random.random() < 0.2 and data_augmentation:
                    root_image, bboxes, class_labels = mosaic(
                        all_filenames[choices], image_width, image_height, class_names, class_to_idx, mosaic_augmentor
                    )
                    bboxes = np.array(bboxes) if len(bboxes) else np.empty((0, 4))
                else:
                    bboxes = np.array(location_data[:, 1:5]) if len(location_data) else np.empty((0, 4))

                bboxes = np.clip(bboxes, 0, 1)


                try:
                    augmented_result = augmentor(image=root_image, bboxes=bboxes, class_labels=class_labels)
                    bboxes = np.array(augmented_result['bboxes'])
                    class_labels = augmented_result['class_labels']
                    image = cv2.resize(augmented_result['image'], (image_width, image_height))
                    class_ints = np.array([class_to_idx[label] for label in class_labels])
                except ValueError as e:
                    print(f"[LOG] Error in augmentation for {filename}: {e}")
                    continue

                
                image = (image / 255).astype(np.float32)

                ydata = [np.zeros((grid_size * 2 ** (2 - i), grid_size * 2 ** (2 - i), anchors, 5 + classes)) for i in range(3)]
                for class_int, (true_center_x, true_center_y, width, height) in zip(class_ints, bboxes):

                    iou_values = Processing.iou(
                        np.array(formatted_anchor_dimensions),
                        np.array([0, 0, width, height]),
                        api=np
                    )[:, 0]

                    anchor_indices = iou_values.argsort()[::-1]

                    has_anchor = [False, False, False]

                    _classes = np.zeros((classes))
                    _classes[int(class_int)] = 1

                    for anchor_index in anchor_indices:
                        scale_idx = anchor_index // anchors
                        anchor_on_scale = anchor_index % anchors

                        _grid_size = grid_size * 2 ** (2 - scale_idx)

                        grid_x_index = int(true_center_x * _grid_size)
                        grid_y_index = int(true_center_y * _grid_size)

                        raw_offset_x = true_center_x * _grid_size - grid_x_index
                        raw_offset_y = true_center_y * _grid_size - grid_y_index

                        _width = 0.5 * np.sqrt(width / anchor_dimensions[anchor_index, 0])
                        _height = 0.5 * np.sqrt(height / anchor_dimensions[anchor_index, 1])

                        occupied = ydata[scale_idx][grid_y_index, grid_x_index, anchor_on_scale, 0]

                        if not occupied and not has_anchor[scale_idx]:
                            has_anchor[scale_idx] = True

                            candidate_cells = [(grid_x_index, grid_y_index, raw_offset_x, raw_offset_y)]

                            if raw_offset_x < 0.5 and grid_x_index - 1 >= 0:
                                candidate_cells.append((grid_x_index - 1, grid_y_index, raw_offset_x + 1, raw_offset_y))
                            elif raw_offset_x >= 0.5 and grid_x_index + 1 < _grid_size:
                                candidate_cells.append((grid_x_index + 1, grid_y_index, raw_offset_x - 1, raw_offset_y))

                            if raw_offset_y < 0.5 and grid_y_index - 1 >= 0:
                                candidate_cells.append((grid_x_index, grid_y_index - 1, raw_offset_x, raw_offset_y + 1))
                            elif raw_offset_y >= 0.5 and grid_y_index + 1 < _grid_size:
                                candidate_cells.append((grid_x_index, grid_y_index + 1, raw_offset_x, raw_offset_y - 1))

                            for gx, gy, ox, oy in candidate_cells:
                                if ydata[scale_idx][gy, gx, anchor_on_scale, 0]:
                                    continue

                                t_x = (ox + 0.5) / 2
                                t_y = (oy + 0.5) / 2

                                ydata[scale_idx][gy, gx, anchor_on_scale] = np.concatenate(
                                    ([1, t_x, t_y, _width, _height], _classes)
                                )

                        elif not occupied and iou_values[anchor_index] > iou_ignore_threshold:
                            ydata[scale_idx][grid_y_index, grid_x_index, anchor_on_scale, 0] = -1
                
                buffer.put((image, ydata))
        except Exception as e:
            print(f"[LOG] Error in fill_buffer: {e}")

            

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
                if lines:
                    raw = np.array([[float(x) for x in line.split()] for line in lines if (line[0] == "2") or not head_only])
                    if raw.shape[0] == 0:
                        continue
                    class_ids, boxes = raw[:, 0], raw[:, 1:]  # x_c, y_c, w, h

                    x_min = np.clip(boxes[:, 0] - boxes[:, 2] / 2, 0.0, 1.0)
                    y_min = np.clip(boxes[:, 1] - boxes[:, 3] / 2, 0.0, 1.0)
                    x_max = np.clip(boxes[:, 0] + boxes[:, 2] / 2, 0.0, 1.0)
                    y_max = np.clip(boxes[:, 1] + boxes[:, 3] / 2, 0.0, 1.0)

                    new_w, new_h = x_max - x_min, y_max - y_min
                    keep = (new_w > 1e-6) & (new_h > 1e-6)

                    new_x_c = x_min[keep] + new_w[keep] / 2
                    new_y_c = y_min[keep] + new_h[keep] / 2
                    class_ids = np.clip(class_ids[keep], 0, len(class_names) - 1)

                    location_data = np.stack([class_ids, new_x_c, new_y_c, new_w[keep], new_h[keep]], axis=1)
                else:
                    continue

        else:
            continue

        if (location_data.shape[0] != 0):
            class_labels = [class_names[int(bbox[0])] for bbox in location_data]
            _bboxes = np.array(location_data)
            dummy_image = np.zeros((64, 64, 3), dtype=np.uint8)
            augmented_result = augmentor(image=dummy_image, bboxes=location_data[:, 1:5], class_labels=class_labels)
            _bboxes = np.array(augmented_result['bboxes'])

            class_labels = augmented_result['class_labels']
            class_idx = np.array([class_names.index(label) for label in class_labels]).reshape(-1, 1)

            _bboxes = np.hstack((class_idx, _bboxes.reshape((-1, 4))))

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

    classes_idx = bboxes[:, 0]
    class_occurrences = np.zeros(classes, dtype=np.int64)
    unique_classes, counts = np.unique(classes_idx, return_counts=True)

    for cls, count in zip(unique_classes, counts):
        class_occurrences[int(cls)] = count

    dimensions_count = anchors * yolo_head_count
    
    dimensions = bboxes[..., 3:5]

    aspect_ratio = np.maximum(
        dimensions[:, 0] / dimensions[:, 1],
        dimensions[:, 1] / dimensions[:, 0]
    )

    # Remove outliers
    
    dimensions = dimensions[aspect_ratio < 8]
    dimensions = dimensions[dimensions[:, 0] * dimensions[:, 1] > 0.001]


    clusters = [np.empty((2)) for _ in range(dimensions_count)]

    plt.figure(figsize=(8, 8))


    anchor_dimensions, labels = iou_kmeans(dimensions, dimensions_count)


    # YOLOv5 style anchor evolution

    def anchor_fitness(anchor_dimensions):

        ratios = dimensions[:, None, :] / anchor_dimensions[None, :, :]
        ratio_values = np.maximum(ratios, 1 / ratios).max(axis=2)

        return (ratio_values.min(axis=1) < 4).mean()


    best_score = anchor_fitness(anchor_dimensions)

    for _ in tqdm(range(10000)):

        mutated_anchor_dimensions = anchor_dimensions * np.random.uniform(0.8, 1.2, anchor_dimensions.shape)

        mutated_anchor_dimensions = np.clip(mutated_anchor_dimensions, 0.001, 1)

        score = anchor_fitness(mutated_anchor_dimensions)

        if score > best_score:
            anchor_dimensions = mutated_anchor_dimensions
            best_score = score


    print("[LOG] Anchor fitness:", best_score)


    # Reassign clusters after evolution

    ratios = dimensions[:, None, :] / anchor_dimensions[None, :, :]
    labels = np.argmin(np.maximum(ratios, 1 / ratios).max(axis=2), axis=1)


    for idx, (label, wh) in enumerate(zip(labels, dimensions)):
        clusters[label] = np.vstack((clusters[label], wh))

    clusters = [cluster[1:] for cluster in clusters]


    order = np.argsort(anchor_dimensions[:, 0] * anchor_dimensions[:, 1])

    anchor_dimensions = anchor_dimensions[order]
    clusters = [clusters[idx] for idx in order]


    num_images = int(len(os.listdir('Training')) * training_percent)
    objects_per_scale = np.zeros(yolo_head_count)

    for idx, (anchor_dimension, cluster) in enumerate(zip(anchor_dimensions, clusters)):

        objects_per_scale[idx // anchors] += cluster.shape[0] / num_images

        formmated_anchor_dimension = np.concatenate(
            (np.zeros(2), anchor_dimension),
            axis=-1
        )

        formatted_cluster = np.concatenate(
            (np.zeros((cluster.shape[0], 2)), cluster),
            axis=-1
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
            label=f"Group {idx} : {iou_values[0].mean()}"
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

    with open("cost-overtime.json", "w+") as file:
        file.write(json.dumps(costs.tolist()))


# ─── Validation loss + mAP helpers ───────────────────────────────────────────

def parse_output_for_map(outputs, anchor_dimensions, anchors, classes, cells, min_presence_score=0.3, max_iou=0.45):
    """Lightweight parse_output used by compute_quick_map during training."""
    out_list = []
    for scale_index, output in enumerate(outputs):
        output = tf.cast(output, dtype=tf.float32)
        anchor_dims_for_head = tf.convert_to_tensor(
            anchor_dimensions[scale_index * anchors:(scale_index + 1) * anchors],
            dtype=output.dtype
        )
        grid_size = (2 ** (2 - scale_index)) * cells
        output = tf.reshape(output, [-1, 5 + classes])

        idxs = tf.range(tf.shape(output)[0])
        grid_y_index = (idxs // anchors) // grid_size
        grid_x_index = (idxs // anchors) % grid_size

        center_x = ((2 * output[:, 1] - 0.5) + tf.cast(grid_x_index, output.dtype)) / tf.cast(grid_size, output.dtype)
        center_y = ((2 * output[:, 2] - 0.5) + tf.cast(grid_y_index, output.dtype)) / tf.cast(grid_size, output.dtype)

        output = tf.concat([
            tf.expand_dims(output[:, 0], -1),
            tf.expand_dims(center_x, -1),
            tf.expand_dims(center_y, -1),
            output[:, 3:5],
            output[:, 5:]
        ], axis=-1)

        wh = (2 * output[:, 3:5]) ** 2
        wh = tf.reshape(wh, [-1, anchors, 2]) * anchor_dims_for_head
        wh = tf.reshape(wh, [-1, 2])

        output = tf.concat([output[:, :3], wh, output[:, 5:]], axis=-1)

        class_ids = tf.argmax(output[:, 5:], axis=-1, output_type=tf.int32)
        class_scores = tf.reduce_max(output[:, 5:], axis=-1)
        combined_score = output[:, 0] * class_scores

        output = tf.concat([
            tf.expand_dims(combined_score, -1),
            output[:, 1:5],
            tf.cast(tf.expand_dims(class_ids, -1), output.dtype)
        ], axis=-1)

        mask = output[:, 0] >= min_presence_score  # now filtering on combined_score
        output = tf.boolean_mask(output, mask)
        out_list.append(output)

    if not out_list:
        return np.array([]), np.zeros((0, 4)), np.array([])

    output = tf.concat(out_list, axis=0)
    if tf.shape(output)[0] == 0:
        return np.array([]), np.zeros((0, 4)), np.array([])

    cx, cy, w, h = output[:, 1], output[:, 2], output[:, 3], output[:, 4]
    y1 = cy - h / 2
    x1 = cx - w / 2
    y2 = cy + h / 2
    x2 = cx + w / 2
    nms_boxes = tf.stack([y1, x1, y2, x2], axis=-1)

    selected_indices = tf.image.non_max_suppression(
        nms_boxes, output[:, 0], 1000,
        iou_threshold=max_iou, score_threshold=min_presence_score
    )
    output = tf.gather(output, selected_indices)

    return output[:, 0].numpy(), output[:, 1:5].numpy(), output[:, 5].numpy().astype(int)


def compute_quick_map(network, files, anchor_dimensions, anchors, classes, cells,
                      image_width, image_height, class_names, n_samples=75, iou_threshold=0.5):
    """
    Compute mAP on a random subset of files during training.
    Keep n_samples low to minimize interruption to the training loop.
    """
    subset = files[np.random.choice(len(files), size=min(n_samples, len(files)), replace=False)]
    num_classes = len(class_names)
    all_true_labels  = [[] for _ in range(num_classes)]
    all_pred_scores  = [[] for _ in range(num_classes)]
    ground_truth_count = np.zeros(num_classes, dtype=np.int32)

    for filename in subset:
        label_path = f'annotations/{filename.replace(".jpg", ".txt").replace(".png", ".txt")}'
        image_path = f'Training/{filename}'
        if not os.path.exists(label_path) or not os.path.exists(image_path):
            continue
        with open(label_path) as f:
            lines = f.read().splitlines()
        if not lines:
            continue
        gt_data = np.array([[float(x) for x in l.split()] for l in lines])
        if gt_data.size == 0:
            continue

        gt_classes = gt_data[:, 0].astype(int)
        gt_boxes   = gt_data[:, 1:]

        img = cv2.imread(image_path)
        if img is None:
            continue

        for gt_class in gt_classes:
            ground_truth_count[gt_class] += 1

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (image_width, image_height))
        input_data = img.astype(network.dtype) / 255.0

        outputs = network.forward(tf.constant(input_data[None, ...]), training=False)
        confidence_scores, box_data, pred_classes = parse_output_for_map(
            outputs, anchor_dimensions, anchors, classes, cells
        )

        if len(confidence_scores) == 0:
            continue

        gt_boxes_abs = gt_boxes.copy()
        gt_boxes_abs[:, [0, 2]] *= image_width
        gt_boxes_abs[:, [1, 3]] *= image_height
        pred_boxes_abs = box_data.copy()
        pred_boxes_abs[:, [0, 2]] *= image_width
        pred_boxes_abs[:, [1, 3]] *= image_height

        matched_gt = set()
        for confidence, pred_box, pred_class in zip(confidence_scores, pred_boxes_abs, pred_classes):
            best_iou    = 0
            best_gt_idx = -1
            for gt_idx, (gt_box, gt_class) in enumerate(zip(gt_boxes_abs, gt_classes)):
                if gt_class != pred_class or gt_idx in matched_gt:
                    continue
                iou = Processing.iou(pred_box, gt_box, api=np)
                if iou > best_iou:
                    best_iou    = iou
                    best_gt_idx = gt_idx
            if best_iou >= iou_threshold:
                matched_gt.add(best_gt_idx)
                all_true_labels[pred_class].append(1)
            else:
                all_true_labels[pred_class].append(0)
            all_pred_scores[pred_class].append(float(confidence))

    ap_per_class = []
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)

    for class_idx in range(num_classes):
        y_true   = np.array(all_true_labels[class_idx])
        y_scores = np.array(all_pred_scores[class_idx])

        if len(y_true) == 0 or y_true.sum() == 0:
            ap_per_class.append(0.0)
            continue

        # ── AP (unchanged) ──────────────────────────────────────────────
        try:
            num_missed = max(ground_truth_count[class_idx] - int(y_true.sum()), 0)
            y_true_padded  = np.concatenate([y_true, np.ones(num_missed)])
            y_score_padded = np.concatenate([y_scores, np.full(num_missed, -1e9)])
            ap_per_class.append(average_precision_score(y_true_padded, y_score_padded))
        except Exception:
            ap_per_class.append(0.0)

        # ── Precision/Recall at best-F1 threshold, not fixed floor ───────
        order = np.argsort(y_scores)[::-1]
        y_true_sorted = y_true[order]

        cum_tp = np.cumsum(y_true_sorted)
        cum_fp = np.cumsum(1 - y_true_sorted)

        cum_precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)
        cum_recall    = cum_tp / max(ground_truth_count[class_idx], 1)

        f1_values = 2 * cum_precision * cum_recall / np.maximum(cum_precision + cum_recall, 1e-9)
        best_index = np.argmax(f1_values)

        precision[class_idx] = cum_precision[best_index]
        recall[class_idx]    = cum_recall[best_index]

    ap_per_class = np.array(ap_per_class)
    mean_ap = float(np.mean(ap_per_class)) if len(ap_per_class) else 0.0

    return ap_per_class, mean_ap, precision, recall, float(np.mean(precision)), float(np.mean(recall))

# ─────────────────────────────────────────────────────────────────────────────


def init_plot(yolo_head_count, titles):
    for i in range(yolo_head_count):
        row     = []
        val_row = []
        ax_row  = []
        for j in range(len(titles)):
            ax = fig.add_subplot(yolo_head_count, len(titles),
                                 (i * len(titles)) + j + 1)

            # Only label the very first subplot — keeps the figure legend clean
            train_label = 'train' if (i == 0 and j == 0) else '_nolegend_'
            val_label   = 'val'   if (i == 0 and j == 0) else '_nolegend_'

            (line,)     = ax.plot([], [], label=train_label)
            (val_line,) = ax.plot([], [], linestyle='--', alpha=0.75, label=val_label)
            ax_row.append(ax)
            row.append(line)
            val_row.append(val_line)
        lines.append(row)
        val_lines.append(val_row)
        axes.append(ax_row)

def live_plot(costs_np, x_values, yolo_head_count, titles, colors, grid_size,
              val_costs_np=None, val_x_values=None):
    for i in range(yolo_head_count):
        for j in range(len(titles)):
            ax       = axes[i][j]
            line     = lines[i][j]
            val_line = val_lines[i][j]

            line.set_data(x_values, costs_np[:, i, j])
            line.set_color(colors[j])

            if val_costs_np is not None and len(val_costs_np) > 3:
                val_line.set_data(val_x_values[2:], val_costs_np[2:, i, j])
            else:
                val_line.set_data(val_x_values, val_costs_np[:, i, j]) if val_costs_np is not None else None

            val_line.set_color(colors[j])

            ax.relim()
            ax.autoscale_view()

            grid = grid_size * (2 ** (yolo_head_count - i - 1))
            ax.set_title(f"{grid}x{grid} ({titles[j]})")

    
def conv(depth, kernel_shape, stride=1, padding="SAME"):
    return [
        Conv2d(depth=depth, kernel_shape=kernel_shape, stride=stride, padding=padding, batch_norm=BatchNorm(
            momentum=0.99,
            baked=True
        ), 
        activation_function=activation_function,
        weight_initializer=LecunNormal(), 
        bias_initializer=Fill(0)),
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
            MaxPool((3,3), pooling_stride=(1, 1), padding="SAME"),

            concat_start2,
                MaxPool((5,5), pooling_stride=(1, 1), padding="SAME"),

                concat_start3,
                    MaxPool((7,7), pooling_stride=(1, 1), padding="SAME"),

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
    batch_size = 16
    accumulate = 1

    image_width, image_height = [352, 352]
    yolo_head_count = 3

    head_only = False

    grid_size = int(image_width / 32)
    grid_count = grid_size ** 2

    anchors = 3
    classes = 2

    yolo_size = 5 # 1: small, 2: medium, 3: large, 4: extra-large, 5: extra-extra-large

    scale = 0.5 + (0.25) * (yolo_size - 1)
    depth_mult = yolo_size / 3

    def R(x):
        return max(1, np.int32(np.ceil(x * depth_mult)))

    dropout_rate = 0
    activation_function = Silu()
    optimize_concats = True
    dtype = np.float16

    save_file = 'model-training-data.json'
    dataset_size = int(len(os.listdir('Training')) * training_percent)
    choices = np.random.choice(len(os.listdir('Training')), size=dataset_size, replace=False)

    with open('training-files.json', 'w+') as file:
        file.write(json.dumps(choices.tolist()))

    # ── Val split ────────────────────────────────────────────────────────────
    all_training_files = np.array(os.listdir('Training'))
    val_mask = np.ones(len(all_training_files), dtype=bool)
    val_mask[choices] = False
    val_choices  = np.where(val_mask)[0]
    val_files    = all_training_files[val_mask]
    train_files  = all_training_files[choices]
    print(f"[LOG] Train: {len(choices)} | Val: {len(val_choices)}")
    # ─────────────────────────────────────────────────────────────────────────

    with open("annotations\\classes.txt", "r+") as file:
        class_names = file.read().splitlines()
    
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    mosaic_augmentor = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.HueSaturationValue(
            hue_shift_limit=30,
            sat_shift_limit=30,
            val_shift_limit=20,
            p=1
        ),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
        A.RandomShadow(p=0.65),
        RandomScaledCenterCrop(min_scale=0.35, max_scale=0.55)
    ], bbox_params=A.BboxParams(format='yolo', min_visibility=0.1, label_fields=['class_labels']))

    augmentor = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.HueSaturationValue(
            hue_shift_limit=25,
            sat_shift_limit=25,
            val_shift_limit=25,
            p=0.9
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.1,
            contrast_limit=0.15,
            p=0.5
        ),
        A.Affine(
            translate_percent=(-0.02, 0.02),
            rotate=(-2.5, 2.5),
            shear=(-5, 5),
            p=0.5
        ),
        A.OneOf([
            A.RandomShadow(p=1.0),
            A.GaussNoise(var_limit=(5.0, 15.0), p=1.0),
            A.MotionBlur(blur_limit=3, p=1.0)
        ], p=0.15),

        RandomScaledCenterCrop(
            min_scale=0.3,
            max_scale=0.65,
            p=1.0
        ),
    ], bbox_params=A.BboxParams(format='yolo', min_visibility=0.15, label_fields=['class_labels']))


    bboxes = get_bboxes(choices)
    anchor_dimensions, class_occurences, objects_per_scale = get_anchor_data(grid_size, bboxes)
    del bboxes
    
    num_classes = len(class_occurences)
    median_dimension = np.mean(anchor_dimensions[anchor_dimensions.shape[0] // 2])

    concat_start1, residual_start1, concat_end1 = Concat(external_concat=optimize_concats).generate_layers()
    concat_start2, residual_start2, concat_end2 = Concat(external_concat=optimize_concats).generate_layers()
    concat_start3, residual_start3, concat_end3 = Concat(external_concat=optimize_concats).generate_layers()
    concat_start4, residual_start4, concat_end4 = Concat(external_concat=optimize_concats).generate_layers()

    weight_initializer = YoloSplit(presence_initializer=HeNormal(), xy_initializer=HeNormal(), dimensions_initializer=ScaledNormal(gain=3.0), class_initializer=HeNormal(), classes=classes, anchors=anchors)
    bias_initializer = YoloSplit(presence_initializer=Fill(-1), xy_initializer=Fill(0), dimensions_initializer=Fill(1.5), class_initializer=Fill(0), classes=classes, anchors=anchors)

    model = [
        Resize((image_height, image_width)),

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

        *csp_block(int(512 * scale), R(3), residual=False), # 8 layers (7 layers)
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
            Activation(Sigmoid),
            Reshape((-1, grid_size * 4, grid_size * 4, anchors, 5 + classes))
        ],
        [
            Conv2d(anchors * (5 + classes), (1, 1), padding="VALID", weight_initializer=weight_initializer, bias_initializer=bias_initializer),
            Activation(Sigmoid),
            Reshape((-1, grid_size * 2, grid_size * 2, anchors, 5 + classes))
        ],
        [
            Conv2d(anchors * (5 + classes), (1, 1), padding="VALID", weight_initializer=weight_initializer, bias_initializer=bias_initializer),
            Activation(Sigmoid),
            Reshape((-1, grid_size * 1, grid_size * 1, anchors, 5 + classes))
        ],
    ]

    cooridnate_weight = 5
    no_object_weight = 10
    object_weight = 1.0
    cls_weight = 0.5

    # Fundamental theory:
    # Both no_object_loss and object_loss will initially have the same goal of low presence scores
    # But as the coordinate_loss optimizes the object_loss will eventually want high presence scores to accomodate for coordinate loss
    # So the model doesn't over-emphasize low objectness scores we use a smaller weight for no_object_loss, to minimize the signal.
    # And increase coordinate weight so that the object_loss doesn't immediately agree with no_object_loss
    # Then object_loss can gradually take over as coordinate_loss converges
    # Contraction loss also starts to conflict with coordinate_loss when it's too high
    # So we keep it low to allow coordinate_loss to converge first
    # Also since no_obj_loss decreases so quickly in the beginning it accumulates a lot of momentum so to combat this we use a smaller momentum term for adam

    # OR 

    # Just use iou_annealing like I did, so no_obj and objectness dont agree at first.
    
    inv_freqs = 1.0 / (class_occurences / min(class_occurences))
    alpha = inv_freqs / inv_freqs.sum()
    alpha = np.sqrt(inv_freqs) # Cap values
    alpha /= alpha.sum()

    print("Weights:", alpha)
    class_loss_function = FocalLoss(alpha=alpha, gamma=2.0)

    print("Objects Per Scale:", objects_per_scale)


    network = Network(
        model=model,
        addon_layers=addon_layers,
        backprop_layer_indices=backprop_layer_indices,
        loss_function = [
            YoloLoss(contraction_weight=0.001, iou_anneal_iterations=30_000, contraction_decay_rate=0.999, coordinate_loss_function=EIoU, objectness_loss_function=BCE, no_objectness_loss_function=BCE, class_loss_function=class_loss_function, grid_size=grid_size, anchors=anchors, classes=classes, class_weight=cls_weight, coordinate_weight=cooridnate_weight, no_object_weight=no_object_weight / objects_per_scale[0], object_weight=object_weight * 1, anchor_dimensions=anchor_dimensions, dtype=dtype),
            YoloLoss(contraction_weight=0.001, iou_anneal_iterations=30_000, contraction_decay_rate=0.999, coordinate_loss_function=EIoU, objectness_loss_function=BCE, no_objectness_loss_function=BCE, class_loss_function=class_loss_function, grid_size=grid_size, anchors=anchors, classes=classes, class_weight=cls_weight, coordinate_weight=cooridnate_weight, no_object_weight=no_object_weight / objects_per_scale[1], object_weight=object_weight * 1, anchor_dimensions=anchor_dimensions, dtype=dtype),
            YoloLoss(contraction_weight=0.001, iou_anneal_iterations=30_000, contraction_decay_rate=0.999, coordinate_loss_function=EIoU, objectness_loss_function=BCE, no_objectness_loss_function=BCE, class_loss_function=class_loss_function, grid_size=grid_size, anchors=anchors, classes=classes, class_weight=cls_weight, coordinate_weight=cooridnate_weight, no_object_weight=no_object_weight / objects_per_scale[2], object_weight=object_weight * 1, anchor_dimensions=anchor_dimensions, dtype=dtype),
        ],
        # loss_function = YoloLoss(grid_size=grid_size, anchors=anchors, coordinate_weight=5, no_object_weight=no_object_weight, object_weight=1),
        optimizer = Adam(momentum = 0.95,  beta_constant = 0.98, weight_decay = 5e-4), 
        # optimizer = RMSProp(beta_constant = 0.9),
        # optimizer = Momentum(momentum=0.937),
        # scheduler = WarmupStepLR(warmup_epochs=10, target_lr=0.001, decay_rate=0.5, decay_interval=50),
        scheduler = StepLR(initial_learning_rate=0.0002, decay_rate=0.45, decay_interval=90),
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
        costs = np.array([])
        starting_idx = 0
        network.compile()

    titles = ['object_loss', 'no_object_loss', 'coordinate_loss', 'class_loss', 'contraction_loss']
    colors = ['C0', 'C1', 'C2', 'C3', 'C4']

    buffer_size = 100
    iou_ignore_threshold = 0.7

    buffer = multiprocessing.Queue(maxsize=buffer_size)
    generator = Generate(batch_size, anchor_dimensions, (image_width, image_height), grid_size, anchors, classes, choices, buffer, buffer_size, iou_ignore_threshold, mosaic_augmentor, augmentor, data_augmentation=True)

    # ── Fixed validation batch ────────────────────────────────────────────────
    # fill_buffer puts individual (image, ydata) items into the queue.
    # Pull batch_size of them and stack exactly as prefetch_to_local does,
    # producing a proper 4D input tensor (batch_size, H, W, C).
    # data_augmentation=False disables mosaic on validation data.
    validation_buffer = multiprocessing.Queue(maxsize=buffer_size)
    for _ in range(1):
        multiprocessing.Process(target=fill_buffer, args=(
            validation_buffer,
            val_choices,
            augmentor,
            mosaic_augmentor,
            anchor_dimensions,
            image_width,
            image_height,
            batch_size,
            grid_size,
            anchors,
            classes,
            class_names,
            class_to_idx,
            iou_ignore_threshold,
            False   # no mosaic augmentation for validation
        ), daemon=True).start()

    print("[LOG] Waiting for fixed validation batch...")
    batch_images    = []
    batch_label_sets = []
    for _ in range(batch_size):
        image, ydata = validation_buffer.get()
        batch_images.append(image)
        batch_label_sets.append(ydata)

    fixed_val_x = np.array(batch_images, dtype=np.float32)   # (batch_size, H, W, C)
    fixed_val_y = tuple(
        np.stack([batch_label_sets[batch_index][head_index] for batch_index in range(batch_size)], axis=0)
        for head_index in range(yolo_head_count)
    )                                                          # tuple of (batch_size, grid_h, grid_w, anchors, 5+classes)

    fixed_val_x_tensor   = tf.constant(fixed_val_x.astype(network.dtype))
    fixed_val_y_tensors  = tuple(
        tf.constant(fixed_val_y[head_index].astype(network.dtype))
        for head_index in range(yolo_head_count)
    )
    print("[LOG] Validation batch ready.")

    # History for val loss and mAP plots
    val_costs_history = []   # list of shape-(yolo_head_count, 5) arrays
    val_epoch_history = []   # epoch at each val measurement
    map_val_history   = []   # list of (ap_per_class_array, mean_ap) tuples
    map_train_history = []
    map_epoch_history = []
    precision_val_history = []
    precision_train_history = []

    recall_val_history = []
    recall_train_history = []
    f1_val_history = []
    f1_train_history = []

    MAP_INTERVAL         = 1000   # compute mAP every N iterations
    val_smoothing_window = 10      # moving average window for val loss, same logic as training
    # ─────────────────────────────────────────────────────────────────────────

    plt.ion()
    fig = plt.figure(figsize=(16, 6))

    lines     = []   # train loss line objects
    val_lines = []   # val loss line objects (dashed, same axes)
    axes      = []

    init_plot(yolo_head_count, titles)

    # ── Combined mAP / Precision / Recall / F1 figure ─────────────────────────
    metric_fig, metric_axes = plt.subplots(1, 4, figsize=(22, 5))
    metric_fig.suptitle("Detection Metrics Over Training", fontsize=12)

    map_ax, precision_ax, recall_ax, f1_ax = metric_axes

    map_class_lines_val = []
    map_class_lines_train = []
    map_colors = ['C0', 'C1', 'C2', 'C3', 'C4']

    for class_index, class_name in enumerate(class_names):
        val_class_line, = map_ax.plot([], [], color=map_colors[class_index % len(map_colors)], linestyle=":", lw=1.5, label=f"{class_name} val")
        train_class_line, = map_ax.plot([], [], color=map_colors[class_index % len(map_colors)], lw=1.5, label=f"{class_name} train")
        map_class_lines_val.append(val_class_line)
        map_class_lines_train.append(train_class_line)

    mean_ap_line_val, = map_ax.plot([], [], color='magenta', linestyle=":", lw=2.5, label="mAP val")
    mean_ap_line_train, = map_ax.plot([], [], color='magenta', lw=2.5, label="mAP train")

    pr_lines = {"precision": {}, "recall": {}, "f1": {}}

    for metric, ax in zip(["precision", "recall", "f1"], [precision_ax, recall_ax, f1_ax]):
        val_line, = ax.plot([], [], linestyle=":", alpha=0.8, label="val")
        train_line, = ax.plot([], [], linestyle="-", alpha=0.8, label="train")
        pr_lines[metric]["val"] = val_line
        pr_lines[metric]["train"] = train_line
        ax.set_title(metric.capitalize())
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric.capitalize())
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend()

    for ax, title in zip(metric_axes, ["mAP", "Precision", "Recall", "F1"]):
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

    map_ax.legend(fontsize=6)

    # ─────────────────────────────────────────────────────────────────────────

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


    for idx, cost in enumerate(network.fit(generator=generator, batch_size=batch_size, accumulate=accumulate, epochs = 2000000000, gradient_transformer=AutoClipper(15)), start=starting_idx):

        print(cost)

        if np.isnan(cost).any():
            os.system("PAUSE")

        try:
            cost = np.stack(cost, axis=0)

            if not len(costs):
                costs = cost[None, ...]
            else:
                costs = np.vstack([costs, cost[None, ...]])

            if not idx % 100:
                epoch = network.epoch
                epoch_steps = max(1, int(np.ceil(dataset_size / (batch_size * accumulate))))

                # ── Training loss smoothing ───────────────────────────────────
                if costs.shape[0] >= epoch_steps and epoch_steps > 1:
                    smoothed = np.apply_along_axis(
                        lambda v: np.convolve(v, np.ones(epoch_steps, dtype=np.float32) / epoch_steps, mode="valid"),
                        axis=0,
                        arr=costs,
                    )
                    pad = np.tile(smoothed[0:1], (costs.shape[0] - smoothed.shape[0], 1, 1))
                    costs_to_plot = np.concatenate([pad, smoothed], axis=0)
                else:
                    costs_to_plot = costs

                steps = costs_to_plot.shape[0]
                x_values = np.arange(steps) * (batch_size / (dataset_size * accumulate))
                # ─────────────────────────────────────────────────────────────

                # ── Validation loss ───────────────────────────────────────────
                try:
                    validation_outputs = network.forward(fixed_val_x_tensor, training=False)
                    validation_costs = []
                    for validation_output, validation_expected, validation_loss_function in zip(
                            validation_outputs, fixed_val_y_tensors, network.loss_functions):
                        result = validation_loss_function.forward(
                            tf.cast(validation_output, network.dtype), validation_expected
                        )
                        # YoloLoss returns (total_loss, stacked_component_losses) — take the components
                        validation_cost = result[1] if isinstance(result, tuple) else result
                        validation_costs.append(
                            validation_cost.numpy() if hasattr(validation_cost, 'numpy') else np.array(validation_cost)
                        )
                    val_costs_history.append(np.stack(validation_costs, axis=0))  # (yolo_head_count, 5)
                    val_epoch_history.append(epoch)
                except Exception as validation_error:
                    print(f"[LOG] Val loss error: {validation_error}")

                # ── Validation loss smoothing (same window logic as training) ─
                val_costs_to_plot = None
                val_x_to_plot     = None
                if val_costs_history:
                    val_costs_array = np.array(val_costs_history)
                    val_x_array     = np.array(val_epoch_history)
                    if val_costs_array.shape[0] >= val_smoothing_window and val_smoothing_window > 1:
                        val_smoothed = np.apply_along_axis(
                            lambda v: np.convolve(v, np.ones(val_smoothing_window, dtype=np.float32) / val_smoothing_window, mode="valid"),
                            axis=0,
                            arr=val_costs_array,
                        )
                        val_pad = np.tile(val_smoothed[0:1], (val_costs_array.shape[0] - val_smoothed.shape[0], 1, 1))
                        val_costs_to_plot = np.concatenate([val_pad, val_smoothed], axis=0)
                    else:
                        val_costs_to_plot = val_costs_array
                    val_x_to_plot = val_x_array
                # ─────────────────────────────────────────────────────────────

                print("PREPLOT")
                live_plot(costs_to_plot, x_values, yolo_head_count, titles, colors, grid_size,
                          val_costs_np=val_costs_to_plot, val_x_values=val_x_to_plot)
                print("POSTPLOT")

                fig.legend()
                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.pause(0.001)
                print("POST DRAW")

            # ── mAP (less frequent) ──────────────────────────────────────────
            if not idx % MAP_INTERVAL:
                epoch = network.epoch
                print(f"[mAP] Computing mAP subset at epoch {epoch:.2f}...")
                (val_ap_per_class, val_mean_ap, val_precision, val_recall, val_mean_precision, val_mean_recall) = compute_quick_map(
                    network, val_files, anchor_dimensions, anchors, classes,
                    grid_size, image_width, image_height, class_names
                )
                (train_ap_per_class, train_mean_ap, train_precision, train_recall, train_mean_precision, train_mean_recall) = compute_quick_map(
                    network, train_files, anchor_dimensions, anchors, classes,
                    grid_size, image_width, image_height, class_names
                )
                map_val_history.append((val_ap_per_class, val_mean_ap))
                map_train_history.append((train_ap_per_class, train_mean_ap))
                map_epoch_history.append(epoch)

                precision_val_history.append((val_precision, val_mean_precision))
                precision_train_history.append((train_precision, train_mean_precision))

                recall_val_history.append((val_recall, val_mean_recall))
                recall_train_history.append((train_recall, train_mean_recall))


                val_f1 = np.divide(
                    2 * val_precision * val_recall,
                    val_precision + val_recall,
                    out=np.zeros_like(val_precision),
                    where=(val_precision + val_recall) > 0
                )

                train_f1 = np.divide(
                    2 * train_precision * train_recall,
                    train_precision + train_recall,
                    out=np.zeros_like(train_precision),
                    where=(train_precision + train_recall) > 0
                )

                f1_val_history.append((val_f1, np.mean(val_f1)))
                f1_train_history.append((train_f1, np.mean(train_f1)))

                map_x_values = np.array(map_epoch_history)
                for class_index in range(len(class_names)):
                    map_class_lines_val[class_index].set_data(
                        map_x_values, [entry[0][class_index] for entry in map_val_history]
                    )
                    map_class_lines_train[class_index].set_data(
                        map_x_values, [entry[0][class_index] for entry in map_train_history]
                    )
                mean_ap_line_val.set_data(map_x_values,   [entry[1] for entry in map_val_history])
                mean_ap_line_train.set_data(map_x_values, [entry[1] for entry in map_train_history])

                # ── Update Precision Recall F1 plot ───────────────────────────────────────
                
                pr_x = np.array(map_epoch_history)

                pr_lines["precision"]["val"].set_data(pr_x, [x[1] for x in precision_val_history])
                pr_lines["precision"]["train"].set_data(pr_x, [x[1] for x in precision_train_history])
                pr_lines["recall"]["val"].set_data(pr_x, [x[1] for x in recall_val_history])
                pr_lines["recall"]["train"].set_data(pr_x, [x[1] for x in recall_train_history])
                pr_lines["f1"]["val"].set_data(pr_x, [x[1] for x in f1_val_history])
                pr_lines["f1"]["train"].set_data(pr_x, [x[1] for x in f1_train_history])

                for metric_ax in metric_axes:
                    metric_ax.relim()
                    metric_ax.autoscale_view()

                metric_fig.canvas.draw()
                metric_fig.canvas.flush_events()
                plt.pause(0.001)
                
                # ───────────────────────────────────────────────────────────────────────────

                print(f"[mAP] Val mAP: {val_mean_ap:.4f} | Train mAP: {train_mean_ap:.4f}")
                print(f"[PR]  Val Precision={val_mean_precision:.4f} Recall={val_mean_recall:.4f} | Train Precision={train_mean_precision:.4f} Recall={train_mean_recall:.4f}")
                for class_index, class_name in enumerate(class_names):
                    print(f"  {class_name}: val={val_ap_per_class[class_index]:.4f}  train={train_ap_per_class[class_index]:.4f}")
            # ─────────────────────────────────────────────────────────────────

            if not idx % 250 and not np.isnan(cost).any():
                save()

                print("PASSED SAVE")
                
        except Exception as e:
            now = datetime.now()
            date_string = now.strftime("%H:%M:%S")

            print(f"[LOG {date_string}] Iteration: {idx} Error: {e} ")

    else:
        print("LOOP EXITED")
