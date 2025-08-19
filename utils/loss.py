from functools import partial
import tensorflow.compat.v1 as tf
import numpy as np, cupy as cp, time
from PIL import Image, ImageTk, ImageDraw
from utils.functions import Processing
from utils.activations import Sigmoid
from tensorflow.experimental.dlpack import from_dlpack, to_dlpack

class CrossEntropy:
    def __init__(self):
        pass

    @staticmethod
    def forward(outputs, target_outputs):
        if outputs.shape[0] == 0:
            return 0

        outputs = tf.clip_by_value(outputs, 1e-6, 1.0)

        loss = -tf.reduce_mean(
            tf.reduce_sum(target_outputs * tf.math.log(outputs), axis=-1)
        )

        return loss


class MSE:
    def __init__(self):
        pass

    @staticmethod
    def forward(outputs, target_outputs, axis=None):
        if outputs.shape[0] == 0:
            return 0
        loss = tf.reduce_mean((outputs - target_outputs) ** 2, axis=axis)
        return loss

class BCE:
    def __init__(self):
        pass

    @staticmethod
    def forward(outputs, target_outputs):
        if outputs.shape[0] == 0:
            return 0
        outputs = tf.clip_by_value(outputs, 1e-6, 1 - 1e-6)  # To prevent log(0)
        loss = -tf.reduce_mean(
            target_outputs * tf.math.log(outputs) + (1 - target_outputs) * tf.math.log(1 - outputs)
        )
        return loss

class FocalLoss:
    def __init__(self, alpha=0.25, gamma=2.0):
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, outputs, target_outputs):
        if outputs.shape[0] == 0:
            return 0.0
        outputs = tf.clip_by_value(outputs, 1e-6, 1. - 1e-6)
        cross_entropy = -(
            target_outputs * tf.math.log(outputs) +
            (1 - target_outputs) * tf.math.log(1 - outputs)
        )
        p_t = tf.where(tf.equal(target_outputs, 1), outputs, 1 - outputs)
        p_t = tf.clip_by_value(p_t, 1e-6, 1. - 1e-6)
        loss = self.alpha * tf.pow(1 - p_t, self.gamma) * cross_entropy
        return tf.reduce_mean(loss)

class SmoothL1:
    def __init__(self, delta=1.0):
        self.delta = delta

    def forward(self, outputs, target_outputs):
        if outputs.shape[0] == 0:
            return 0

        diff = outputs - target_outputs 
        return tf.reduce_mean(tf.where(tf.abs(diff) <= self.delta, 0.5 * diff ** 2, self.delta * tf.abs(diff) - 0.5 * self.delta))


class SIoU:
    @staticmethod
    def forward(outputs, target_outputs):
        eps = 1e-6
        dtype = outputs.dtype
        outputs = tf.cast(tf.reshape(outputs, [-1, 4]), tf.float64)
        target_outputs = tf.cast(tf.reshape(target_outputs, [-1, 4]), tf.float64)

        outputs_wh = tf.maximum(outputs[..., 2:4], eps)
        target_outputs_wh = tf.maximum(target_outputs[..., 2:4], eps)

        outputs_top_left = outputs[..., :2] - outputs_wh / 2
        outputs_bottom_right = outputs[..., :2] + outputs_wh / 2
        target_outputs_top_left = target_outputs[..., :2] - target_outputs_wh / 2
        target_outputs_bottom_right = target_outputs[..., :2] + target_outputs_wh / 2

        inter_top_left = tf.maximum(outputs_top_left, target_outputs_top_left)
        inter_bottom_right = tf.minimum(outputs_bottom_right, target_outputs_bottom_right)
        inter_wh = tf.maximum(inter_bottom_right - inter_top_left, 0.0)
        inter_area = inter_wh[..., 0] * inter_wh[..., 1]

        outputs_area = tf.maximum((outputs_bottom_right[..., 0] - outputs_top_left[..., 0]) *
                                 (outputs_bottom_right[..., 1] - outputs_top_left[..., 1]), eps)
        target_outputs_area = tf.maximum((target_outputs_bottom_right[..., 0] - target_outputs_top_left[..., 0]) *
                                        (target_outputs_bottom_right[..., 1] - target_outputs_top_left[..., 1]), eps)
        union_area = outputs_area + target_outputs_area - inter_area
        union_area = tf.maximum(union_area, eps)

        iou = inter_area / union_area

        outputs_center = outputs[..., :2]
        target_outputs_center = target_outputs[..., :2]
        center_distance = tf.reduce_sum(tf.square(outputs_center - target_outputs_center), axis=1)

        enclosing_top_left = tf.minimum(outputs_top_left, target_outputs_top_left)
        enclosing_bottom_right = tf.maximum(outputs_bottom_right, target_outputs_bottom_right)
        enclosing_wh = tf.maximum(enclosing_bottom_right - enclosing_top_left, eps)
        enclosing_diagonal = tf.reduce_sum(tf.square(enclosing_wh), axis=1)
        enclosing_diagonal = tf.maximum(enclosing_diagonal, eps)

        # Angle cost
        delta_x = target_outputs_center[..., 0] - outputs_center[..., 0]
        delta_y = target_outputs_center[..., 1] - outputs_center[..., 1]
        angle = tf.abs(tf.atan2(delta_y, delta_x))
        angle_cost = 1 - tf.cos(angle)

        # Shape cost
        w_pred, h_pred = outputs_wh[..., 0], outputs_wh[..., 1]
        w_gt, h_gt = target_outputs_wh[..., 0], target_outputs_wh[..., 1]
        shape_cost = tf.abs(w_pred - w_gt) / (w_gt + eps) + tf.abs(h_pred - h_gt) / (h_gt + eps)

        # SIoU loss
        siou_loss = 1 - iou + center_distance / enclosing_diagonal + angle_cost + shape_cost

        loss = tf.reduce_mean(tf.cast(siou_loss, dtype))

        # Replace NaNs with zeros for stability
        loss = tf.where(tf.math.is_finite(loss), loss, tf.zeros_like(loss))

        return loss
    

class CIoU:
    @staticmethod
    def forward(outputs, target_outputs):
        eps = 1e-6
        dtype = outputs.dtype
        outputs = tf.cast(tf.reshape(outputs, [-1, 4]), tf.float64)
        target_outputs = tf.cast(tf.reshape(target_outputs, [-1, 4]), tf.float64)

        outputs_wh = tf.maximum(outputs[..., 2:4], eps)
        target_outputs_wh = tf.maximum(target_outputs[..., 2:4], eps)

        outputs_top_left = outputs[..., :2] - outputs_wh / 2
        outputs_bottom_right = outputs[..., :2] + outputs_wh / 2
        target_outputs_top_left = target_outputs[..., :2] - target_outputs_wh / 2
        target_outputs_bottom_right = target_outputs[..., :2] + target_outputs_wh / 2

        inter_top_left = tf.maximum(outputs_top_left, target_outputs_top_left)
        inter_bottom_right = tf.minimum(outputs_bottom_right, target_outputs_bottom_right)
        inter_wh = tf.maximum(inter_bottom_right - inter_top_left, 0.0)
        inter_area = inter_wh[..., 0] * inter_wh[..., 1]

        outputs_area = tf.maximum((outputs_bottom_right[..., 0] - outputs_top_left[..., 0]) *
                                 (outputs_bottom_right[..., 1] - outputs_top_left[..., 1]), eps)
        target_outputs_area = tf.maximum((target_outputs_bottom_right[..., 0] - target_outputs_top_left[..., 0]) *
                                        (target_outputs_bottom_right[..., 1] - target_outputs_top_left[..., 1]), eps)
        union_area = outputs_area + target_outputs_area - inter_area
        union_area = tf.maximum(union_area, eps)

        iou = inter_area / union_area

        outputs_center = outputs[..., :2]
        target_outputs_center = target_outputs[..., :2]
        center_distance = tf.reduce_sum(tf.square(outputs_center - target_outputs_center), axis=1)

        enclosing_top_left = tf.minimum(outputs_top_left, target_outputs_top_left)
        enclosing_bottom_right = tf.maximum(outputs_bottom_right, target_outputs_bottom_right)
        enclosing_wh = tf.maximum(enclosing_bottom_right - enclosing_top_left, eps)
        enclosing_diagonal = tf.reduce_sum(tf.square(enclosing_wh), axis=1)
        enclosing_diagonal = tf.maximum(enclosing_diagonal, eps)

        outputs_aspect_ratio = outputs_wh[..., 0] / (outputs_wh[..., 1] + eps)
        target_aspect_ratio = target_outputs_wh[..., 0] / (target_outputs_wh[..., 1] + eps)
        v = 4.0 * tf.square(tf.atan(outputs_aspect_ratio) - tf.atan(target_aspect_ratio)) / (np.pi ** 2 + eps)

        alpha = v / (1 - iou + v + eps)
        alpha *= tf.cast(iou >= 0.5, tf.float64)

        ciou_loss = 1 - iou + center_distance / enclosing_diagonal + alpha * v
        loss = tf.cast(ciou_loss, dtype)

        # Replace NaNs with zeros for stability
        loss = tf.where(tf.math.is_finite(loss), loss, tf.zeros_like(loss))

        return loss

class DIoU:
    @staticmethod
    def forward(outputs, target_outputs):
        dtype = outputs.dtype
        outputs = tf.cast(tf.reshape(outputs, [-1, 4]), tf.float64)
        target_outputs = tf.cast(tf.reshape(target_outputs, [-1, 4]), tf.float64)

        outputs_wh = outputs[..., 2:4]
        target_outputs_wh = target_outputs[..., 2:4]

        outputs_top_left = outputs[..., :2] - outputs_wh / 2
        outputs_bottom_right = outputs[..., :2] + outputs_wh / 2
        target_outputs_top_left = target_outputs[..., :2] - target_outputs_wh / 2
        target_outputs_bottom_right = target_outputs[..., :2] + target_outputs_wh / 2

        inter_top_left = tf.maximum(outputs_top_left, target_outputs_top_left)
        inter_bottom_right = tf.minimum(outputs_bottom_right, target_outputs_bottom_right)
        inter_wh = tf.maximum(inter_bottom_right - inter_top_left, 0.0)
        inter_area = inter_wh[..., 0] * inter_wh[..., 1]

        outputs_area = (outputs_bottom_right[..., 0] - outputs_top_left[..., 0]) * (outputs_bottom_right[..., 1] - outputs_top_left[..., 1])
        target_outputs_area = (target_outputs_bottom_right[..., 0] - target_outputs_top_left[..., 0]) * (target_outputs_bottom_right[..., 1] - target_outputs_top_left[..., 1])
        union_area = outputs_area + target_outputs_area - inter_area

        iou = inter_area / (union_area + 1e-6)

        outputs_center = outputs[..., :2]
        target_outputs_center = target_outputs[..., :2]
        center_distance = tf.reduce_sum(tf.square(outputs_center - target_outputs_center), axis=1)

        enclosing_top_left = tf.minimum(outputs_top_left, target_outputs_top_left)
        enclosing_bottom_right = tf.maximum(outputs_bottom_right, target_outputs_bottom_right)
        enclosing_wh = enclosing_bottom_right - enclosing_top_left
        enclosing_diagonal = tf.reduce_sum(tf.square(enclosing_wh), axis=1)

        # DIoU loss
        diou_loss = 1 - iou + center_distance / (enclosing_diagonal + 1e-6)

        loss = tf.cast(diou_loss, dtype)

        return loss
        

class YoloLoss:
    def __init__(self, grid_size=13, anchors=5, classes=2, coordinate_weight=5, class_weight=0.5, no_object_weight=0.5, object_weight=1, contraction_weight=1e-6, contraction_decay_rate=0.99, coordinate_loss_function=MSE, objectness_loss_function=MSE, class_loss_function=BCE, anchor_dimensions=None, dtype=tf.float64):
        self.objectness_loss_function = objectness_loss_function
        self.coordinate_loss_function = coordinate_loss_function
        self.class_loss_function = class_loss_function

        self.anchor_dimensions = tf.constant(anchor_dimensions, dtype=dtype)
        self.grid_size = tf.constant(grid_size, dtype=tf.float64)

        self.contraction_decay_rate = tf.constant(contraction_decay_rate, dtype=dtype)
        self.contraction_weight = tf.constant(contraction_weight, dtype=dtype)

        self.coordinate_weight = tf.constant(coordinate_weight, dtype=dtype)
        self.no_object_weight = tf.constant(no_object_weight, dtype=dtype)
        self.object_weight = tf.constant(object_weight, dtype=dtype)
        self.class_weight = tf.constant(class_weight, dtype=dtype)

        self.classes = classes
        self.anchors = anchors
        self.dtype = dtype
        self.epoch = 0

    def forward(self, outputs, target_outputs):  # Fully TensorFlow-based implementation
        orig_dtype = outputs.dtype
        outputs = tf.cast(outputs, tf.float64)
        target_outputs = tf.cast(target_outputs, tf.float64)

        dtype = tf.float64
        anchor_dimensions = tf.cast(self.anchor_dimensions, dtype)
        contraction_weight = tf.cast(self.contraction_weight * self.contraction_decay_rate ** self.epoch, dtype)
        coordinate_weight = tf.cast(self.coordinate_weight, dtype)
        no_object_weight = tf.cast(self.no_object_weight, dtype)
        object_weight = tf.cast(self.object_weight, dtype)
        class_weight = tf.cast(self.class_weight, dtype)
        anchors = tf.cast(self.anchors, dtype)
        classes = tf.cast(self.classes, dtype)

        batch_size = outputs.shape[0]
        grid_size = tf.sqrt(tf.reduce_prod(outputs.shape[1:]) / tf.cast(anchors * (5 + classes), tf.int32))
        grid_size = tf.cast(tf.floor(grid_size), dtype)

        scale_idx = tf.math.log(grid_size / tf.cast(self.grid_size, dtype)) / tf.math.log(tf.constant(2.0, dtype=dtype))

        # Shape: (anchors, 2)
        scale_anchor_wh = tf.convert_to_tensor(
            anchor_dimensions[int(scale_idx * anchors): int((scale_idx + 1) * anchors)],
            dtype=dtype,
        )

        presence_scores = tf.reshape(outputs, (batch_size, -1))[..., ::int(5 + classes)]
        target_presence_scores = tf.reshape(target_outputs, (batch_size, -1))[..., ::int(5 + classes)]

        inactive_mask = tf.equal(target_presence_scores, 0)
        active_mask = tf.equal(target_presence_scores, 1)

        inactive_presence = tf.boolean_mask(presence_scores, inactive_mask)
        target_inactive_presence = tf.boolean_mask(target_presence_scores, inactive_mask)

        active_presence = tf.boolean_mask(presence_scores, active_mask)
        target_active_presence = tf.boolean_mask(target_presence_scores, active_mask)

        grid_count = int(tf.ceil(grid_size ** 2))

        # Remove presence score
        boxes = tf.reshape(outputs, (batch_size, grid_count * anchors, 5 + classes))[..., 1:]
        target_boxes = tf.reshape(target_outputs, (batch_size, grid_count * anchors, 5 + classes))[..., 1:]

        idxs = tf.cast(tf.range(grid_count * anchors), dtype=tf.int32)

        grid_x = (idxs // tf.cast(anchors, dtype=tf.int32)) % tf.cast(grid_size, dtype=tf.int32)
        grid_y = (idxs // tf.cast(anchors, dtype=tf.int32)) // tf.cast(grid_size, dtype=tf.int32)

        grid_xy = tf.stack([grid_x, grid_y], axis=-1)
        grid_xy = tf.cast(grid_xy, dtype=boxes.dtype)
        grid_xy = tf.expand_dims(grid_xy, axis=0)  # (1, grid_count * anchors, 2)

        scaled_xy = (2 * boxes[..., :2] - 0.5 + grid_xy) / grid_size
        scaled_target_xy = (2 * target_boxes[..., :2] - 0.5 + grid_xy) / grid_size

        # Shape: (batch_size, grid_count, anchors, 2)
        reshaped_boxes = tf.reshape(boxes[..., 2:4], (batch_size, grid_count, anchors, 2))
        reshaped_target_boxes = tf.reshape(target_boxes[..., 2:4], (batch_size, grid_count, anchors, 2))

        # Convert raw width and height to bounding box width and height
        scaled_wh = tf.reshape((2 * reshaped_boxes)**2, (batch_size, grid_count, anchors, 2)) * scale_anchor_wh
        scaled_target_wh = tf.reshape((2 * reshaped_target_boxes)**2, (batch_size, grid_count, anchors, 2)) * scale_anchor_wh

        scaled_wh = tf.reshape(scaled_wh, (batch_size, grid_count * anchors, 2))
        scaled_target_wh = tf.reshape(scaled_target_wh, (batch_size, grid_count * anchors, 2))

        del reshaped_target_boxes


        # Concatenate offsets with wh
        active_boxes = tf.concat([
            tf.boolean_mask(scaled_xy, active_mask), 
            tf.boolean_mask(scaled_wh, active_mask)
            ], axis=-1
        )
        
        active_target_boxes = tf.concat([
            tf.boolean_mask(scaled_target_xy, active_mask), 
            tf.boolean_mask(scaled_target_wh, active_mask)
            ], axis=-1
       )


        reshaped_boxes = tf.reshape(scaled_xy, (batch_size * grid_count, anchors, 2))
        inactive_boxes_wh = tf.boolean_mask(scaled_wh, inactive_mask)
        inactive_boxes_xy = tf.boolean_mask(scaled_xy, inactive_mask)

        del scaled_xy, scaled_wh
        del scaled_target_xy, scaled_target_wh

        inactive_boxes = tf.concat([inactive_boxes_xy, inactive_boxes_wh], axis=-1)
        # Set target x and y offsets to what was predicted so there is only a penalty for the wh
        inactive_target_boxes = tf.concat([reshaped_boxes, tf.repeat(scale_anchor_wh[None, :, :], reshaped_boxes.shape[0], axis=0)], axis=-1)  # Join predicted offsets with anchor box dimensions

        del reshaped_boxes, boxes

        inactive_target_boxes = tf.reshape(inactive_target_boxes, (batch_size, grid_count * anchors, 4))
        inactive_target_boxes = tf.stop_gradient(
            tf.boolean_mask(inactive_target_boxes, inactive_mask), 
        )

        # ious = 1 - tf.stop_gradient(self.coordinate_loss_function.forward(active_boxes, active_target_boxes)) / 2
        ious = tf.stop_gradient(tf.linalg.diag_part(Processing.iou(active_boxes, active_target_boxes, api=tf)))

        print(tf.reduce_mean(active_boxes[..., 2:4], axis=tuple(range(active_boxes.shape.ndims - 1))), "predicted wh")
        print(tf.reduce_mean(active_target_boxes[..., 2:4], axis=tuple(range(active_boxes.shape.ndims - 1))), "target wh")


        print(float(tf.reduce_mean(active_presence)), "predicted ious")
        print(float(tf.reduce_mean(ious)), scale_idx, "target ious")


        # Define class predictions and target class predictions
        reshaped_classes = tf.reshape(outputs[..., 5:], (batch_size, grid_count * anchors, classes))
        reshaped_target_classes = tf.reshape(target_outputs[..., 5:], (batch_size, grid_count * anchors, classes))

        active_class = tf.boolean_mask(reshaped_classes, active_mask)
        target_active_class = tf.boolean_mask(reshaped_target_classes, active_mask)

        del reshaped_classes, reshaped_target_classes

        # Compute losses
        coordinate_loss = tf.reduce_mean(
            self.coordinate_loss_function.forward(
            active_boxes,
            active_target_boxes
            )
        ) * coordinate_weight
        del active_boxes, active_target_boxes

        contraction_loss = tf.reduce_mean(
            self.coordinate_loss_function.forward(
            inactive_boxes,
            inactive_target_boxes
            )
        ) * contraction_weight
        del inactive_boxes, inactive_target_boxes

        no_object_loss = self.objectness_loss_function.forward(
            inactive_presence,
            target_inactive_presence
        ) * no_object_weight
        del inactive_presence, target_inactive_presence

        object_loss = self.objectness_loss_function.forward(
            active_presence,
            target_active_presence * ious
        ) * object_weight
        del active_presence, target_active_presence, ious

        class_loss = self.class_loss_function.forward(
            active_class,
            target_active_class
        ) * class_weight
        del active_class, target_active_class

        # Cast losses back to original dtype for stability
        coordinate_loss = tf.cast(coordinate_loss, orig_dtype)
        contraction_loss = tf.cast(contraction_loss, orig_dtype)
        no_object_loss = tf.cast(no_object_loss, orig_dtype)
        object_loss = tf.cast(object_loss, orig_dtype)
        class_loss = tf.cast(class_loss, orig_dtype)

        loss = (object_loss, no_object_loss, coordinate_loss, class_loss, contraction_loss) # These are for collection metrics to return back
        total_loss = object_loss + no_object_loss + coordinate_loss + class_loss + contraction_loss # Actual output used for backpropagation

        self.epoch += 1

        return (total_loss, tf.stack(loss, axis=0))
