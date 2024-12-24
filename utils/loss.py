from functools import partial
import tensorflow.compat.v1 as tf
import numpy as np, cupy as cp, time
from utils.functions import Processing
from PIL import Image, ImageTk, ImageDraw
from tensorflow.experimental.dlpack import from_dlpack, to_dlpack

class CrossEntropy:
    def __init__(self, epsilon=1e-12):
        self.epsilon = epsilon

    def forward(self, outputs, expected_outputs):
        outputs = tf.clip_by_value(outputs, self.epsilon, 1.0 - self.epsilon)
        return tf.reduce_mean(-(expected_outputs * tf.log(outputs + self.epsilon)))

    def backward(self, outputs, expected_outputs):
        return (outputs - expected_outputs)

class MSE:
    def __init__(self):
        pass

    @staticmethod
    def forward(outputs, expected_outputs):
        if outputs.shape[0] == 0:
            return 0

        return tf.reduce_mean((outputs - expected_outputs) ** 2)

    @staticmethod
    def backward(outputs, expected_outputs):
        if outputs.shape[0] == 0:
            return 0

        return 2 * (outputs - expected_outputs)

class BinaryCrossEntropy:
    def __init__(self):
        pass

    @staticmethod
    def forward(outputs, expected_outputs):
        if outputs.shape[0] == 0:
            return 0

        outputs = tf.clip_by_value(outputs, 1e-6, 1 - 1e-6)  # To prevent log(0)

        return -tf.reduce_mean(
            expected_outputs * tf.math.log(outputs) + (1 - expected_outputs) * tf.math.log(1 - outputs)
        )

    @staticmethod
    def backward(outputs, expected_outputs):
        if outputs.shape[0] == 0:
            return 0

        outputs = tf.clip_by_value(outputs, 1e-6, 1 - 1e-6)  # To prevent division by zero

        return (outputs - expected_outputs) / (outputs * (1 - outputs) + 1e-6)

class SmoothL1:
    def __init__(self, delta=1.0):
        self.delta = delta

    def forward(self, outputs, expected_outputs):
        if outputs.shape[0] == 0:
            return 0

        diff = outputs - expected_outputs 
        return tf.reduce_mean(tf.where(tf.abs(diff) <= self.delta, 0.5 * diff ** 2, self.delta * tf.abs(diff) - 0.5 * self.delta))

    def backward(self, outputs, expected_outputs):
        if outputs.shape[0] == 0:
            return 0

        diff = outputs - expected_outputs
        mask = tf.abs(diff) <= self.delta
        return tf.where(mask, diff, tf.sign(diff) * self.delta)

class DIoU:
    @staticmethod
    def forward(outputs, expected_outputs):
        outputs = cp.array(outputs).reshape(-1, 4)
        expected_outputs = cp.array(expected_outputs).reshape(-1, 4)
        
        outputs_top_left = outputs[..., :2] - outputs[..., 2:] / 2
        outputs_bottom_right = outputs[..., :2] + outputs[..., 2:] / 2
        expected_outputs_top_left = expected_outputs[..., :2] - expected_outputs[..., 2:] / 2
        expected_outputs_bottom_right = expected_outputs[..., :2] + expected_outputs[..., 2:] / 2
        
        inter_top_left = cp.maximum(outputs_top_left, expected_outputs_top_left)
        inter_bottom_right = cp.minimum(outputs_bottom_right, expected_outputs_bottom_right)
        inter_wh = cp.maximum(inter_bottom_right - inter_top_left, 0)
        inter_area = inter_wh[..., 0] * inter_wh[..., 1]
        
        outputs_area = (outputs_bottom_right[..., 0] - outputs_top_left[..., 0]) * (outputs_bottom_right[..., 1] - outputs_top_left[..., 1])
        expected_outputs_area = (expected_outputs_bottom_right[..., 0] - expected_outputs_top_left[..., 0]) * (expected_outputs_bottom_right[..., 1] - expected_outputs_top_left[..., 1])
        union_area = outputs_area + expected_outputs_area - inter_area
        
        iou = inter_area / (union_area + 1e-6)
        
        outputs_center = outputs[..., :2]
        expected_outputs_center = expected_outputs[..., :2]
        center_distance = cp.sum((outputs_center - expected_outputs_center) ** 2, axis=1)
        
        enclosing_top_left = cp.minimum(outputs_top_left, expected_outputs_top_left)
        enclosing_bottom_right = cp.maximum(outputs_bottom_right, expected_outputs_bottom_right)
        enclosing_wh = enclosing_bottom_right - enclosing_top_left
        enclosing_diagonal = cp.sum(enclosing_wh ** 2, axis=1)
        
        diou_loss = 1 - iou + center_distance / (enclosing_diagonal + 1e-6)
        
        return cp.mean(diou_loss)

    @staticmethod
    def compute_iou_gradient(outputs, expected_outputs):
        outputs_top_left = outputs[..., :2] - outputs[..., 2:] / 2
        outputs_bottom_right = outputs[..., :2] + outputs[..., 2:] / 2
        expected_outputs_top_left = expected_outputs[..., :2] - expected_outputs[..., 2:] / 2
        expected_outputs_bottom_right = expected_outputs[..., :2] + expected_outputs[..., 2:] / 2
        
        inter_top_left = cp.maximum(outputs_top_left, expected_outputs_top_left)
        inter_bottom_right = cp.minimum(outputs_bottom_right, expected_outputs_bottom_right)
        inter_wh = cp.maximum(inter_bottom_right - inter_top_left, 0)
        inter_area = inter_wh[..., 0] * inter_wh[..., 1]
        
        outputs_area = (outputs_bottom_right[..., 0] - outputs_top_left[..., 0]) * \
                      (outputs_bottom_right[..., 1] - outputs_top_left[..., 1])
        expected_outputs_area = (expected_outputs_bottom_right[..., 0] - expected_outputs_top_left[..., 0]) * \
                               (expected_outputs_bottom_right[..., 1] - expected_outputs_top_left[..., 1])
        union_area = outputs_area + expected_outputs_area - inter_area
        
        diou_grad_xy = cp.zeros_like(outputs[..., :2])
        mask = (inter_wh > 0).all(axis=1)[..., None]
        diou_grad_xy += mask * (1 / (union_area + 1e-6))[..., None]
        diou_grad_xy -= (outputs_area / (union_area + 1e-6))[..., None]
        
        diou_grad_wh = cp.zeros_like(outputs[..., 2:])
        diou_grad_wh += mask * (1 / (union_area + 1e-6))[..., None]
        diou_grad_wh -= (outputs_area / (union_area + 1e-6))[..., None]
        
        return cp.hstack((diou_grad_xy, diou_grad_wh))
    
    @staticmethod
    def compute_center_distance_gradient(outputs, expected_outputs):
        outputs_center = outputs[..., :2]
        expected_outputs_center = expected_outputs[..., :2]
        
        outputs_top_left = outputs[..., :2] - outputs[..., 2:] / 2
        outputs_bottom_right = outputs[..., :2] + outputs[..., 2:] / 2
        expected_outputs_top_left = expected_outputs[..., :2] - expected_outputs[..., 2:] / 2
        expected_outputs_bottom_right = expected_outputs[..., :2] + expected_outputs[..., 2:] / 2
        
        enclosing_top_left = cp.minimum(outputs_top_left, expected_outputs_top_left)
        enclosing_bottom_right = cp.maximum(outputs_bottom_right, expected_outputs_bottom_right)
        enclosing_diagonal = cp.sum((enclosing_bottom_right - enclosing_top_left) ** 2, axis=1)
        
        center_grad = 2 * (outputs_center - expected_outputs_center) / (enclosing_diagonal[..., None] + 1e-7)

        diag_grad = -cp.sum((outputs_center - expected_outputs_center) ** 2, axis=1)[..., None] * \
                   (2 * (outputs[..., 2:] / 2)) / (enclosing_diagonal[..., None] ** 2 + 1e-7)
        
        return cp.hstack((center_grad, diag_grad))
    
    @staticmethod
    def backward(outputs, expected_outputs):
        outputs = cp.array(outputs).reshape(-1, 4)
        expected_outputs = cp.array(expected_outputs).reshape(-1, 4)
        
        iou_grad = DIoU.compute_iou_gradient(outputs, expected_outputs)
        center_grad = DIoU.compute_center_distance_gradient(outputs, expected_outputs)

        diou_grad = -iou_grad + center_grad

        diou_grad_safe = cp.where(cp.isnan(diou_grad), cp.zeros_like(diou_grad), diou_grad)
        diou_grad_safe = cp.where(cp.isinf(diou_grad_safe), cp.zeros_like(diou_grad), diou_grad_safe)

        diou_grad = cp.clip(diou_grad_safe, -2, 2)

        return diou_grad.flatten()

class YoloLoss:
    def __init__(self, grid_size=13, anchors=5, coordinate_weight=5, no_object_weight=0.5, object_weight=1, coordinate_loss_function=MSE, objectness_loss_function=MSE, anchor_dimensions=None):
        self.objectness_loss_function = objectness_loss_function
        self.coordinate_loss_function = coordinate_loss_function
        self.anchor_dimensions = anchor_dimensions
        self.coordinate_weight = coordinate_weight
        self.no_object_weight = no_object_weight
        self.object_weight = object_weight
        self.grid_size = grid_size

        self.anchors = anchors

    def forward(self, outputs, expected_outputs):
        coordinate_weight = self.coordinate_weight
        no_object_weight = self.no_object_weight
        object_weight = self.object_weight

        anchors = self.anchors
        batch_size = outputs.shape[0]

        outputs = cp.array(outputs)
        expected_outputs = cp.array(expected_outputs)

        grid_count = int(outputs.size / (anchors * 5 * batch_size))

        new_expected_outputs = cp.zeros((batch_size, grid_count, anchors, 5), dtype=expected_outputs.dtype)

        # for grid_idx, (grid, true_grid) in enumerate(zip(outputs.reshape(batch_size, grid_count, anchors, 5), expected_outputs.reshape(batch_size, grid_count, anchors, 5))):
        #     active_anchors = true_grid[true_grid[:, 0] == 1]
        #     predicted_anchors = grid[:, 1:]

        #     if not active_anchors.shape[0]:
        #         continue

        #     anchor_indices = cp.argmax(Processing.iou(active_anchors[:, 1:], predicted_anchors), axis=1)
        #     new_expected_outputs[grid_idx, anchor_indices] = active_anchors

            # for true_anchor in active_anchors:
            #     anchor_idx = cp.argmax(Processing.iou(true_anchor[1:], predicted_anchors))
            #     new_expected_outputs[grid_idx][anchor_idx] = true_anchor

        # Seperation

        outputs_reshaped = outputs.reshape(batch_size, grid_count, anchors, 5)
        expected_outputs_reshaped = expected_outputs.reshape(batch_size, grid_count, anchors, 5)

        active_anchors_mask = expected_outputs_reshaped[..., 0] == 1
        print(batch_size, grid_count, anchors, 5, active_anchors_mask.shape)
        active_anchors = expected_outputs_reshaped[active_anchors_mask]
        print(active_anchors.shape)

        grid_indices = cp.nonzero(active_anchors_mask)[0]
        anchor_indices_per_grid = cp.nonzero(active_anchors_mask)[1]

        predicted_anchors = outputs_reshaped[..., 1:].reshape(-1, anchors, 4)

        predicted_anchors_for_active_grids = predicted_anchors[grid_indices]
        iou_values = Processing.iou(active_anchors[:, None, 1:], predicted_anchors_for_active_grids)

        best_anchor_indices = cp.argmax(iou_values, axis=-1)

        # Seperation

        # best_anchor_ious = cp.max(iou_values, axis=-1)

        # for i, (grid_idx, best_anchor_idx, best_anchor_iou) in enumerate(zip(grid_indices, best_anchor_indices, best_anchor_ious)):
        #     for j in cp.argsort(best_anchor_iou)[::-1]:
        #         target_anchor = best_anchor_idx[j]

        #         if new_expected_outputs[grid_idx, target_anchor, 0] == 1:
        #             available_anchors = cp.where(new_expected_outputs[grid_idx, :, 0] == 0)[0]
        #             if available_anchors.size != 0:
        #                 target_anchor = available_anchors[0]

        #             else:
        #                 continue # Dont display anchor if ran out of space

        #         new_expected_outputs[grid_idx, target_anchor] = active_anchors[j]

        # del best_anchor_ious

        # Seperation

        new_expected_outputs[:, grid_indices, best_anchor_indices] = active_anchors

        del best_anchor_indices, active_anchors

        expected_outputs = new_expected_outputs.reshape(batch_size, -1)
        
        self.expected_outputs = expected_outputs

        presence_scores = Processing.to_tensorflow(outputs[:, ::5])
        expected_presence_scores = Processing.to_tensorflow(expected_outputs[:, ::5])

        coordinate_mask = cp.arange(0, outputs.shape[1])
        coordinate_mask = (coordinate_mask % 5 != 0) & (expected_outputs[:, (coordinate_mask // 5) * 5] == 1)

        if outputs[coordinate_mask].size != 0:
            predicted_coords = outputs[coordinate_mask].reshape(-1, 4)
            expected_coords = expected_outputs[coordinate_mask].reshape(-1, 4)
            
            predicted_coords[..., 2:4] = cp.sign(predicted_coords[..., 2:4]) * cp.sqrt(cp.abs(predicted_coords[..., 2:4]) + 1e-6)
            expected_coords[..., 2:4] = cp.sign(expected_coords[..., 2:4]) * cp.sqrt(cp.abs(expected_coords[..., 2:4]) + 1e-6)

            predicted_coords = Processing.to_tensorflow(predicted_coords.flatten())
            expected_coords = Processing.to_tensorflow(expected_coords.flatten())

        else:
            predicted_coords = cp.array([])
            expected_coords = cp.array([])

        no_object_loss = self.objectness_loss_function.forward(presence_scores[expected_presence_scores == 0], expected_presence_scores[expected_presence_scores == 0]) * no_object_weight
        object_loss = self.objectness_loss_function.forward(presence_scores[expected_presence_scores == 1], expected_presence_scores[expected_presence_scores == 1]) * object_weight

        if len(predicted_coords):
            coordinate_loss = self.coordinate_loss_function.forward(predicted_coords, expected_coords) * coordinate_weight
            # if cp.isnan(cp.array(object_loss)).any():
            #     print(presence_scores[expected_presence_scores == 1], expected_presence_scores[expected_presence_scores == 1])

        else:
            coordinate_loss = 0

        return Processing.to_tensorflow(cp.hstack((object_loss, no_object_loss, coordinate_loss), dtype=outputs.dtype))

    def backward(self, outputs, expected_outputs):
        coordinate_weight = self.coordinate_weight
        no_object_weight = self.no_object_weight
        object_weight = self.object_weight

        anchors = self.anchors

        expected_outputs = self.expected_outputs

        outputs = cp.array(outputs)

        presence_scores = Processing.to_tensorflow(outputs[:, ::5])
        expected_presence_scores = Processing.to_tensorflow(expected_outputs[:, ::5])

        coordinate_mask = cp.arange(0, outputs.shape[1])
        coordinate_mask = (coordinate_mask % 5 != 0) & (expected_outputs[:, (coordinate_mask // 5) * 5] == 1)

        if outputs[coordinate_mask].size != 0:
            predicted_coords = outputs[coordinate_mask].reshape(-1, 4)
            expected_coords = expected_outputs[coordinate_mask].reshape(-1, 4)
            
            predicted_coords[..., 2:4] = cp.sign(predicted_coords[..., 2:4]) * cp.sqrt(cp.abs(predicted_coords[..., 2:4]) + 1e-6)
            expected_coords[..., 2:4] = cp.sign(expected_coords[..., 2:4]) * cp.sqrt(cp.abs(expected_coords[..., 2:4]) + 1e-6)

            predicted_coords = Processing.to_tensorflow(predicted_coords.flatten())
            expected_coords = Processing.to_tensorflow(expected_coords.flatten())
        else:
            predicted_coords = cp.array([])
            expected_coords = cp.array([])

        del self.expected_outputs
        deriv_values = cp.zeros(outputs.shape)

        deriv_values[:, ::5][expected_presence_scores == 0] = self.objectness_loss_function.backward(presence_scores[expected_presence_scores == 0], expected_presence_scores[expected_presence_scores == 0]) * no_object_weight
        deriv_values[:, ::5][expected_presence_scores == 1] = self.objectness_loss_function.backward(presence_scores[expected_presence_scores == 1], expected_presence_scores[expected_presence_scores == 1]) * object_weight
        
        if len(predicted_coords) != 0:
            deriv_values[coordinate_mask] = self.coordinate_loss_function.backward(predicted_coords, expected_coords) * coordinate_weight

        return Processing.to_tensorflow(deriv_values)