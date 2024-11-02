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
        return tf.reduce_mean(-(expected_outputs * tf.log(outputs + self.epsilon)), axis=0)

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

class YoloLoss:
    def __init__(self, grid_size=13, anchors=5, coordinate_weight=5, no_object_weight=0.5, object_weight=1):
        self.coordinate_weight = coordinate_weight
        self.no_object_weight = no_object_weight
        self.object_weight = object_weight

        self.grid_size = grid_size
        self.anchors = anchors

    def forward(self, outputs, expected_outputs):
        coordinate_weight = self.coordinate_weight
        no_object_weight = self.no_object_weight
        object_weight = self.object_weight

        grid_size = self.grid_size
        anchors = self.anchors

        outputs = cp.array(outputs)
        expected_outputs = cp.array(expected_outputs)

        new_expected_outputs = cp.zeros((grid_size ** 2, anchors, 5), dtype=expected_outputs.dtype)

        # for grid_idx, (grid, true_grid) in enumerate(zip(outputs.reshape(grid_size ** 2, anchors, 5), expected_outputs.reshape(grid_size ** 2, anchors, 5))):
        #     active_anchors = true_grid[true_grid[:, 0] == 1]
        #     predicted_anchors = grid[:, 1:]

        #     if not active_anchors.shape[0]:
        #         continue

        #     anchor_indices = cp.argmax(Processing.iou(active_anchors[:, 1:], predicted_anchors), axis=1)
        #     new_expected_outputs[grid_idx][anchor_indices] = active_anchors

        #     # for true_anchor in active_anchors:
        #     #     anchor_idx = cp.argmax(Processing.iou(true_anchor[1:], predicted_anchors))
        #     #     new_expected_outputs[grid_idx][anchor_idx] = true_anchor

        outputs_reshaped = outputs.reshape(grid_size ** 2, anchors, 5)
        expected_outputs_reshaped = expected_outputs.reshape(grid_size ** 2, anchors, 5)

        active_anchors_mask = expected_outputs_reshaped[:, :, 0] == 1
        active_anchors = expected_outputs_reshaped[active_anchors_mask]

        grid_indices = cp.nonzero(active_anchors_mask)[0]
        anchor_indices_per_grid = cp.nonzero(active_anchors_mask)[1]

        predicted_anchors = outputs_reshaped[:, :, 1:].reshape(grid_size ** 2, anchors, 4)

        predicted_anchors_for_active_grids = predicted_anchors[grid_indices]
        iou_values = Processing.iou(active_anchors[:, None, 1:], predicted_anchors_for_active_grids)

        best_anchor_indices = cp.argmax(iou_values, axis=-1)

        # Seperation

        best_anchor_ious = cp.max(iou_values, axis=-1)

        for i, (grid_idx, best_anchor_idx, best_anchor_iou) in enumerate(zip(grid_indices, best_anchor_indices, best_anchor_ious)):
            for j in cp.argsort(best_anchor_iou)[::-1]:
                target_anchor = best_anchor_idx[j]

                if new_expected_outputs[grid_idx, target_anchor, 0] == 1:
                    available_anchors = cp.where(new_expected_outputs[grid_idx, :, 0] == 0)[0]
                    if available_anchors.size != 0:
                        target_anchor = available_anchors[0]

                    else:
                        continue # Dont display anchor if ran out of space

                new_expected_outputs[grid_idx, target_anchor] = active_anchors[j]

        # new_expected_outputs[grid_indices, best_anchor_indices] = active_anchors

        expected_outputs = new_expected_outputs.flatten()
        
        self.expected_outputs = expected_outputs

        presence_scores = Processing.to_tensorflow(outputs[::5])
        expected_presence_scores = Processing.to_tensorflow(expected_outputs[::5])

        coordinate_mask = cp.arange(0, outputs.shape[0])
        coordinate_mask = (coordinate_mask % 5 != 0) & (expected_outputs[(coordinate_mask // 5) * 5] == 1)

        if outputs[coordinate_mask].size != 0:
            predicted_coords = outputs[coordinate_mask].reshape((-1, 4))
            expected_coords = expected_outputs[coordinate_mask].reshape((-1, 4))
            
            predicted_coords[..., 2:4] = cp.sqrt(predicted_coords[..., 2:4])
            expected_coords[..., 2:4] = cp.sqrt(expected_coords[..., 2:4])

            predicted_coords = Processing.to_tensorflow(predicted_coords.flatten())
            expected_coords = Processing.to_tensorflow(expected_coords.flatten())

        else:
            predicted_coords = cp.array([])
            expected_coords = cp.array([])

        no_object_loss = MSE.forward(presence_scores[expected_presence_scores == 0], expected_presence_scores[expected_presence_scores == 0]) * no_object_weight
        object_loss = MSE.forward(presence_scores[expected_presence_scores == 1], expected_presence_scores[expected_presence_scores == 1]) * object_weight
        coordinate_loss = MSE.forward(predicted_coords, expected_coords) * coordinate_weight

        return Processing.to_tensorflow(cp.hstack((object_loss, no_object_loss, coordinate_loss), dtype=outputs.dtype))

    def backward(self, outputs, expected_outputs):
        coordinate_weight = self.coordinate_weight
        no_object_weight = self.no_object_weight
        object_weight = self.object_weight

        grid_size = self.grid_size
        anchors = self.anchors

        expected_outputs = self.expected_outputs

        outputs = cp.array(outputs)

        presence_scores = Processing.to_tensorflow(outputs[::5])
        expected_presence_scores = Processing.to_tensorflow(expected_outputs[::5])

        coordinate_mask = cp.arange(0, outputs.shape[0])
        coordinate_mask = (coordinate_mask % 5 != 0) & (expected_outputs[(coordinate_mask // 5) * 5] == 1)

        if outputs[coordinate_mask].size != 0:
            predicted_coords = outputs[coordinate_mask].reshape((-1, 4))
            expected_coords = expected_outputs[coordinate_mask].reshape((-1, 4))
            
            predicted_coords[..., 2:4] = cp.sqrt(predicted_coords[..., 2:4])
            expected_coords[..., 2:4] = cp.sqrt(expected_coords[..., 2:4])

            predicted_coords = Processing.to_tensorflow(predicted_coords.flatten())
            expected_coords = Processing.to_tensorflow(expected_coords.flatten())
        else:
            predicted_coords = cp.array([])
            expected_coords = cp.array([])

        del self.expected_outputs
        deriv_values = cp.zeros(outputs.shape)

        deriv_values[::5][expected_presence_scores == 0] = MSE.backward(presence_scores[expected_presence_scores == 0], expected_presence_scores[expected_presence_scores == 0]) * no_object_weight
        deriv_values[::5][expected_presence_scores == 1] = MSE.backward(presence_scores[expected_presence_scores == 1], expected_presence_scores[expected_presence_scores == 1]) * object_weight
        deriv_values[coordinate_mask] = MSE.backward(predicted_coords, expected_coords) * coordinate_weight

        return Processing.to_tensorflow(deriv_values)