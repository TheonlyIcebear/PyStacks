from functools import partial
import tensorflow.compat.v1 as tf
import numpy as np, cupy as cp, time
from PIL import Image, ImageTk, ImageDraw
from tensorflow.experimental.dlpack import from_dlpack, to_dlpack

class CrossEntropy:
    def __init__(self, epsilon=1e-12):
        self.epsilon = epsilon

    def forward(self, outputs, target_outputs):
        outputs = tf.clip_by_value(outputs, self.epsilon, 1.0 - self.epsilon)
        return tf.reduce_mean(-(target_outputs * tf.log(outputs + self.epsilon)))

class MSE:
    def __init__(self):
        pass

    @staticmethod
    def forward(outputs, target_outputs, axis=None):
        if outputs.shape[0] == 0:
            return 0

        return tf.reduce_mean((outputs - target_outputs) ** 2, axis=axis)

class BCE:
    def __init__(self):
        pass

    @staticmethod
    def forward(outputs, target_outputs):
        if outputs.shape[0] == 0:
            return 0

        outputs = tf.clip_by_value(outputs, 1e-6, 1 - 1e-6)  # To prevent log(0)

        return -tf.reduce_mean(
            target_outputs * tf.math.log(outputs) + (1 - target_outputs) * tf.math.log(1 - outputs)
        )

class SmoothL1:
    def __init__(self, delta=1.0):
        self.delta = delta

    def forward(self, outputs, target_outputs):
        if outputs.shape[0] == 0:
            return 0

        diff = outputs - target_outputs 
        return tf.reduce_mean(tf.where(tf.abs(diff) <= self.delta, 0.5 * diff ** 2, self.delta * tf.abs(diff) - 0.5 * self.delta))

class CIoU:
    @staticmethod
    def forward(outputs, target_outputs):
        dtype = outputs.dtype
        outputs = tf.cast(tf.reshape(outputs, [-1, 4]), tf.float64)
        target_outputs = tf.cast(tf.reshape(target_outputs, [-1, 4]), tf.float64)

        outputs_wh = tf.exp(outputs[..., 2:])
        target_outputs_wh = tf.exp(target_outputs[..., 2:])

        outputs_top_left = outputs[..., :2] - outputs_wh / 2
        outputs_bottom_right = outputs[..., :2] + outputs_wh / 2
        target_outputs_top_left = target_outputs[..., :2] - target_outputs_wh / 2
        target_outputs_bottom_right = target_outputs[..., :2] + target_outputs_wh / 2

        inter_top_left = tf.maximum(outputs_top_left, target_outputs_top_left)
        inter_bottom_right = tf.minimum(outputs_bottom_right, target_outputs_bottom_right)
        inter_wh = tf.maximum(inter_bottom_right - inter_top_left, 0.0)
        inter_area = inter_wh[..., 0] * inter_wh[..., 1]

        outputs_area = outputs_wh[..., 0] * outputs_wh[..., 1]
        target_outputs_area = target_outputs_wh[..., 0] * target_outputs_wh[..., 1]
        union_area = outputs_area + target_outputs_area - inter_area

        iou = inter_area / (union_area + 1e-6)

        outputs_center = outputs[..., :2]
        target_outputs_center = target_outputs[..., :2]
        center_distance = tf.reduce_sum(tf.square(outputs_center - target_outputs_center), axis=1)

        enclosing_top_left = tf.minimum(outputs_top_left, target_outputs_top_left)
        enclosing_bottom_right = tf.maximum(outputs_bottom_right, target_outputs_bottom_right)
        enclosing_wh = tf.maximum(enclosing_bottom_right - enclosing_top_left, 0.0)
        enclosing_diagonal = tf.reduce_sum(tf.square(enclosing_wh), axis=1)

        outputs_aspect_ratio = outputs_wh[..., 0] / outputs_wh[..., 1]
        target_aspect_ratio = target_outputs_wh[..., 0] / target_outputs_wh[..., 1]
        v = 4.0 * tf.square(tf.atan(outputs_aspect_ratio) - tf.atan(target_aspect_ratio)) / (3.141592653589793 ** 2)
        alpha = v / (1 - iou + v)

        ciou_loss = 1 - iou + center_distance / enclosing_diagonal + alpha * v
        loss = tf.cast(ciou_loss, dtype)

        return loss

class DIoU:
    @staticmethod
    def forward(outputs, target_outputs):
        dtype = outputs.dtype
        outputs = tf.cast(tf.reshape(outputs, [-1, 4]), tf.float64)
        target_outputs = tf.cast(tf.reshape(target_outputs, [-1, 4]), tf.float64)

        outputs_wh = tf.exp(outputs[..., 2:])
        target_outputs_wh = tf.exp(target_outputs[..., 2:])

        outputs_top_left = outputs[..., :2] - outputs_wh / 2
        outputs_bottom_right = outputs[..., :2] + outputs_wh / 2
        target_outputs_top_left = target_outputs[..., :2] - target_outputs_wh / 2
        target_outputs_bottom_right = target_outputs[..., :2] + target_outputs_wh / 2

        inter_top_left = tf.maximum(outputs_top_left, target_outputs_top_left)
        inter_bottom_right = tf.minimum(outputs_bottom_right, target_outputs_bottom_right)
        inter_wh = tf.maximum(inter_bottom_right - inter_top_left, 0.0)
        inter_area = inter_wh[..., 0] * inter_wh[..., 1]

        outputs_area = (outputs_bottom_right[..., 0] - outputs_top_left[..., 0]) * \
                    (outputs_bottom_right[..., 1] - outputs_top_left[..., 1])
        target_outputs_area = (target_outputs_bottom_right[..., 0] - target_outputs_top_left[..., 0]) * \
                                (target_outputs_bottom_right[..., 1] - target_outputs_top_left[..., 1])
        union_area = outputs_area + target_outputs_area - inter_area

        iou = inter_area / (union_area + 1e-6)

        outputs_center = outputs[..., :2]
        target_outputs_center = target_outputs[..., :2]
        center_distance = tf.reduce_sum(tf.square(outputs_center - target_outputs_center), axis=1)

        enclosing_top_left = tf.minimum(outputs_top_left, target_outputs_top_left)
        enclosing_bottom_right = tf.maximum(outputs_bottom_right, target_outputs_bottom_right)
        enclosing_wh = tf.maximum(enclosing_bottom_right - enclosing_top_left, 0.0)
        enclosing_diagonal = tf.reduce_sum(tf.square(enclosing_wh), axis=1)

        diou_loss = 1 - iou + center_distance / enclosing_diagonal

        loss = tf.cast(diou_loss, dtype)

        return loss
        

class YoloLoss:
    def __init__(self, grid_size=13, anchors=5, coordinate_weight=5, no_object_weight=0.5, object_weight=1, coordinate_loss_function=MSE, objectness_loss_function=MSE, anchor_dimensions=None, dtype=tf.float64):
        self.objectness_loss_function = objectness_loss_function
        self.coordinate_loss_function = coordinate_loss_function
        self.anchor_dimensions = tf.constant(anchor_dimensions, dtype=dtype)
        self.grid_size = tf.constant(grid_size, dtype=dtype)
        self.coordinate_weight = coordinate_weight
        self.no_object_weight = no_object_weight
        self.object_weight = object_weight
        self.anchors = anchors
        self.dtype = dtype

    def forward(self, outputs, target_outputs):  # Fully TensorFlow-based implementation
        dtype = self.dtype
        coordinate_weight = tf.constant(self.coordinate_weight, dtype=dtype)
        no_object_weight = tf.constant(self.no_object_weight, dtype=dtype)
        object_weight = tf.constant(self.object_weight, dtype=dtype)
        anchors = tf.constant(self.anchors, dtype=dtype)

        batch_size = outputs.shape[0]
        grid_size = tf.sqrt(tf.reduce_prod(tf.cast(outputs.shape[1:], dtype)) / (anchors * 5))
        grid_size = tf.floor(grid_size)

        scale_idx = tf.math.log(grid_size / self.grid_size) / tf.math.log(tf.constant(2.0, dtype=dtype))

        presence_scores = outputs[:, ::5]
        target_presence_scores = target_outputs[:, ::5]

        inactive_mask = tf.equal(target_presence_scores, 0)
        active_mask = tf.equal(target_presence_scores, 1)

        inactive_presence = tf.boolean_mask(presence_scores, inactive_mask)
        target_inactive_presence = tf.boolean_mask(target_presence_scores, inactive_mask)

        active_presence = tf.boolean_mask(presence_scores, active_mask)
        target_active_presence = tf.boolean_mask(target_presence_scores, active_mask)

        # Remove presence score
        boxes = tf.reshape(outputs, (batch_size, -1, 5))[..., 1:]
        target_boxes = tf.reshape(target_outputs, (batch_size, -1, 5))[..., 1:]

        # Shape: (batch_size, grid_size ** 2, anchors, 2)
        reshaped_boxes = tf.reshape(boxes[..., 2:4], (batch_size, -1, anchors, 2))
        reshaped_target_boxes = tf.reshape(target_boxes[..., 2:4], (batch_size, -1, anchors, 2))

        # Shape: (anchors, 2)
        scale_anchor_dimensions = tf.constant(
            self.anchor_dimensions[int(scale_idx * self.anchors): int((scale_idx + 1) * self.anchors)],
            dtype=dtype,
        )

        # Convert raw width and height to bounding box width and height

        updated_boxes = tf.boolean_mask(tf.reshape(tf.exp(reshaped_boxes) * scale_anchor_dimensions, (batch_size, -1, 2)), active_mask)
        updated_target_boxes = tf.boolean_mask(tf.reshape(tf.exp(reshaped_target_boxes) * scale_anchor_dimensions, (batch_size, -1, 2)), active_mask)

        # Concatenate offsets with dimensions
        updated_boxes = tf.concat([tf.boolean_mask(boxes[..., :2], active_mask), updated_boxes], axis=-1)
        updated_target_boxes = tf.concat([tf.boolean_mask(target_boxes[..., :2], active_mask), updated_target_boxes], axis=-1)

        boxes = tf.boolean_mask(boxes, active_mask)
        target_boxes = tf.boolean_mask(target_boxes, active_mask)

        CIoU.forward(updated_boxes, updated_target_boxes)
        ious = tf.stop_gradient(tf.maximum((1-DIoU.forward(updated_boxes, updated_target_boxes)), 0.1))
        ious = tf.tensor_scatter_nd_update(
            ious,
            tf.where(active_presence > ious),
            tf.boolean_mask(active_presence, active_presence > ious)
        )

        print(ious)

        for iou, presence, box, target_box in zip(ious, active_presence, updated_boxes, updated_target_boxes):
            print(np.append(presence.numpy(), box.numpy()), iou)
            print(np.append(1, target_box.numpy()))

        # Compute losses
        coordinate_loss = tf.reduce_mean(self.coordinate_loss_function.forward(boxes, target_boxes)) * coordinate_weight
        no_object_loss = self.objectness_loss_function.forward(inactive_presence, target_inactive_presence) * no_object_weight
        object_loss = self.objectness_loss_function.forward(active_presence, ious) * object_weight

        loss = (object_loss, no_object_loss, coordinate_loss)
        total_loss = coordinate_loss + no_object_loss + object_loss

        return total_loss, tf.concat(loss, axis=0)