from functools import partial
import tensorflow.compat.v1 as tf
import numpy as np, cupy as cp, time
from PIL import Image, ImageTk, ImageDraw
from utils.functions import Processing
from tensorflow.experimental.dlpack import from_dlpack, to_dlpack

class CrossEntropy:
    def __init__(self, epsilon=1e-12):
        self.epsilon = epsilon

    def forward(self, outputs, target_outputs):
        outputs += epsilon
        per_sample_loss = -tf.reduce_sum(target_outputs * tf.math.log(outputs), axis=1)
        return tf.reduce_mean(per_sample_loss)

class MSE:
    def __init__(self):
        pass

    @staticmethod
    def forward(outputs, target_outputs, axis=None):
        if outputs.shape[0] == 0:
            return 0

        return tf.reduce_mean((outputs - target_outputs) ** 2, axis=axis)

class BCE:
    def __init__(self, epsilon=1e-12):
        self.epsilon = epsilon

    def forward(self, outputs, target_outputs):
        if outputs.shape[0] == 0:
            return 0

        outputs += self.epsilon  # To prevent log(0)

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

        outputs_wh = outputs[..., 2:]
        target_outputs_wh = target_outputs[..., 2:]

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
        alpha *= tf.cast(iou >= 0.5, tf.float64)

        ciou_loss = 1 - iou + center_distance / enclosing_diagonal + alpha * v
        loss = tf.cast(ciou_loss, dtype)

        return loss

class DIoU:
    @staticmethod
    def forward(outputs, target_outputs):
        dtype = outputs.dtype
        outputs = tf.cast(tf.reshape(outputs, [-1, 4]), tf.float64)
        target_outputs = tf.cast(tf.reshape(target_outputs, [-1, 4]), tf.float64)

        outputs_wh = outputs[..., 2:]
        target_outputs_wh = target_outputs[..., 2:]

        outputs_top_left = outputs[..., :2] - outputs_wh / 2
        outputs_bottom_right = outputs[..., :2] + outputs_wh / 2
        target_outputs_top_left = target_outputs[..., :2] - target_outputs_wh / 2
        target_outputs_bottom_right = target_outputs[..., :2] + target_outputs_wh / 2

        # Intersection area calculation
        inter_top_left = tf.maximum(outputs_top_left, target_outputs_top_left)
        inter_bottom_right = tf.minimum(outputs_bottom_right, target_outputs_bottom_right)
        inter_wh = tf.maximum(inter_bottom_right - inter_top_left, 0.0)
        inter_area = inter_wh[..., 0] * inter_wh[..., 1]

        # Area of the individual boxes
        outputs_area = (outputs_bottom_right[..., 0] - outputs_top_left[..., 0]) * \
                    (outputs_bottom_right[..., 1] - outputs_top_left[..., 1])
        target_outputs_area = (target_outputs_bottom_right[..., 0] - target_outputs_top_left[..., 0]) * \
                                (target_outputs_bottom_right[..., 1] - target_outputs_top_left[..., 1])

        # Union area
        union_area = outputs_area + target_outputs_area - inter_area

        # IoU calculation
        iou = inter_area / (union_area + 1e-6)

        # Center distance calculation (make sure to square the differences)
        outputs_center = outputs[..., :2]
        target_outputs_center = target_outputs[..., :2]
        center_distance = tf.reduce_sum(tf.square(outputs_center - target_outputs_center), axis=1)

        # Enclosing box diagonal calculation (correct the diagonal formula)
        enclosing_top_left = tf.minimum(outputs_top_left, target_outputs_top_left)
        enclosing_bottom_right = tf.maximum(outputs_bottom_right, target_outputs_bottom_right)
        enclosing_wh = enclosing_bottom_right - enclosing_top_left
        enclosing_diagonal = tf.reduce_sum(tf.square(enclosing_wh), axis=1)

        # DIoU loss
        diou_loss = 1 - iou + center_distance / (enclosing_diagonal + 1e-6)

        loss = tf.cast(diou_loss, dtype)

        return loss
        

class YoloLoss:
    def __init__(self, grid_size=13, anchors=5, coordinate_weight=5, no_object_weight=0.5, object_weight=1, contraction_weight=1e-6, contraction_decay_rate=0.9, coordinate_loss_function=MSE, objectness_loss_function=MSE, anchor_dimensions=None, dtype=tf.float64):
        self.objectness_loss_function = objectness_loss_function
        self.coordinate_loss_function = coordinate_loss_function

        self.grid_size = grid_size
        self.anchor_dimensions = tf.constant(anchor_dimensions, dtype=dtype)

        self.contraction_decay_rate = tf.constant(contraction_decay_rate, dtype=dtype)
        self.contraction_weight = tf.constant(contraction_weight, dtype=dtype)

        self.coordinate_weight = tf.constant(coordinate_weight, dtype=dtype)
        self.no_object_weight = tf.constant(no_object_weight, dtype=dtype)
        self.object_weight = tf.constant(object_weight, dtype=dtype)

        self.anchors = anchors
        self.dtype = dtype
        self.epoch = 0

    def forward(self, outputs, target_outputs):  # Fully TensorFlow-based implementation
        dtype = self.dtype
        contraction_weight = self.contraction_weight * self.contraction_decay_rate ** self.epoch
        coordinate_weight = tf.constant(self.coordinate_weight, dtype=dtype)
        no_object_weight = tf.constant(self.no_object_weight, dtype=dtype)
        object_weight = tf.constant(self.object_weight, dtype=dtype)
        anchors = self.anchors

        batch_size = outputs.shape[0]
        grid_size = tf.sqrt(tf.reduce_prod(outputs.shape[1:]) / (anchors * 5))
        grid_size = tf.floor(grid_size)

        scale_idx = tf.math.log(grid_size / self.grid_size) / tf.math.log(tf.constant(2.0, dtype=tf.float64))

        # Shape: (anchors, 2)
        scale_anchor_wh = tf.convert_to_tensor(
            self.anchor_dimensions[int(scale_idx * self.anchors): int((scale_idx + 1) * self.anchors)],
            dtype=dtype,
        )

        presence_scores = tf.reshape(outputs, (batch_size, -1))[..., ::5]
        target_presence_scores = tf.reshape(target_outputs, (batch_size, -1))[..., ::5]

        inactive_mask = tf.equal(target_presence_scores, 0)
        active_mask = tf.equal(target_presence_scores, 1)

        inactive_presence = tf.boolean_mask(presence_scores, inactive_mask)
        target_inactive_presence = tf.boolean_mask(target_presence_scores, inactive_mask)

        active_presence = tf.boolean_mask(presence_scores, active_mask)
        target_active_presence = tf.boolean_mask(target_presence_scores, active_mask)

        # Remove presence score
        boxes = tf.reshape(outputs, (batch_size, grid_size ** 2 * anchors, 5))[..., 1:]
        target_boxes = tf.reshape(target_outputs, (batch_size, grid_size ** 2 * anchors, 5))[..., 1:]

        # Shape: (batch_size, grid_size ** 2, anchors, 2)
        reshaped_boxes = tf.reshape(boxes[..., 2:4], (batch_size, grid_size ** 2, anchors, 2))
        reshaped_target_boxes = tf.reshape(target_boxes[..., 2:4], (batch_size, grid_size ** 2, anchors, 2))

        # Convert raw width and height to bounding box width and height

        scaled_wh = tf.reshape(tf.exp(reshaped_boxes), (batch_size, grid_size ** 2 * anchors, 2))
        scaled_target_wh = tf.reshape(tf.exp(reshaped_target_boxes), (batch_size, grid_size ** 2 * anchors, 2))

        active_boxes_wh = tf.boolean_mask(scaled_wh, active_mask)
        active_target_boxes_wh = tf.boolean_mask(scaled_target_wh, active_mask)

        # Concatenate offsets with wh
        active_boxes = tf.concat([tf.boolean_mask(boxes[..., :2], active_mask), active_boxes_wh], axis=-1)
        active_target_boxes = tf.concat([tf.boolean_mask(target_boxes[..., :2], active_mask), active_target_boxes_wh], axis=-1)


        reshaped_boxes = tf.reshape(boxes[..., :2], (batch_size * grid_size ** 2, anchors, 2))
        inactive_boxes_wh = tf.boolean_mask(scaled_wh, inactive_mask)
        
        inactive_boxes = tf.concat([tf.boolean_mask(boxes[..., :2], inactive_mask), inactive_boxes_wh], axis=-1)
        # Set target x and y offsets to what was predicted so there is only a penalty for the wh
        inactive_target_boxes = tf.concat([reshaped_boxes, tf.repeat(scale_anchor_wh[None, :, :], reshaped_boxes.shape[0], axis=0)], axis=-1)  # Join predicted offsets with anchor box dimensions

        inactive_target_boxes = tf.reshape(inactive_target_boxes, (batch_size, grid_size ** 2 * anchors, 4))
        inactive_target_boxes = tf.stop_gradient(
            tf.boolean_mask(inactive_target_boxes, inactive_mask), 
        )

        ious = tf.stop_gradient(tf.linalg.diag_part(Processing.iou(active_boxes, active_target_boxes, api=tf)))

        # Compute losses
        coordinate_loss = tf.reduce_mean(self.coordinate_loss_function.forward(active_boxes, active_target_boxes)) * coordinate_weight
        contraction_loss = tf.reduce_mean(self.coordinate_loss_function.forward(inactive_boxes, inactive_target_boxes)) * contraction_weight
        no_object_loss = self.objectness_loss_function.forward(inactive_presence, target_inactive_presence) * no_object_weight
        object_loss = self.objectness_loss_function.forward(active_presence, target_active_presence * ious) * object_weight

        loss = (object_loss, no_object_loss, coordinate_loss) # These are for collection metrics to return back
        total_loss = object_loss + coordinate_loss + no_object_loss + contraction_loss # Actual output used for backpropagation

        self.epoch += 1

        return (total_loss, tf.stack(loss, axis=0))