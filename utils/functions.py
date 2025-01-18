from functools import partial
import numpy as np, cupy as cp, time
import tensorflow as tf
from PIL import Image, ImageTk, ImageDraw

class Processing:
    def iou(boxes1, boxes2, api=cp):
        boxes1_top_left = (boxes1[..., :2] - (boxes1[..., 2:] / 2))
        boxes1_bottom_right = (boxes1[..., :2] + (boxes1[..., 2:] / 2))

        boxes2_top_left = (boxes2[..., :2] - (boxes2[..., 2:] / 2))
        boxes2_bottom_right = (boxes2[..., :2] + (boxes2[..., 2:] / 2))


        top_left_x = api.maximum(boxes1_top_left[..., 0:1], boxes2_top_left[..., 0])  # Broadcasting along batch size
        top_left_y = api.maximum(boxes1_top_left[..., 1:2], boxes2_top_left[..., 1])

        bottom_right_x = api.minimum(boxes1_bottom_right[..., 0:1], boxes2_bottom_right[..., 0])
        bottom_right_y = api.minimum(boxes1_bottom_right[..., 1:2], boxes2_bottom_right[..., 1])

        intersection_width = api.maximum(0.0, bottom_right_x - top_left_x)
        intersection_height = api.maximum(0.0, bottom_right_y - top_left_y)
        intersection_area = intersection_width * intersection_height

        boxes1_area = (boxes1_bottom_right[..., 0] - boxes1_top_left[..., 0]) * (boxes1_bottom_right[..., 1] - boxes1_top_left[..., 1])
        boxes2_area = (boxes2_bottom_right[..., 0] - boxes2_top_left[..., 0]) * (boxes2_bottom_right[..., 1] - boxes2_top_left[..., 1])

        union_area = boxes1_area[..., None] + boxes2_area - intersection_area

        iou = intersection_area / api.maximum(union_area, 1e-6)  # Avoid division by zero

        return iou

class ClipGradient:
    def __init__(self, clip_norm):
        self.clip_norm = clip_norm

    def _clip_gradient(self, gradient):
        for idx, layer in enumerate(gradient):
            if isinstance(layer, cp.ndarray):
                if layer.size == 0:
                    continue

                # Convert to float64 then back since it's more numerically stable
                gradient[idx] = tf.clip_by_norm(gradient[idx] + 1.0e-8, self.clip_norm)

            else:
                gradient[idx] = self._clip_gradient(layer)

        return gradient
        
    def _get_norm(self, gradient):
        dtype = gradient.dtype
        gradient = tf.cast(gradient, tf.float64)

        l2sum = tf.math.reduce_sum(gradient * gradient, keepdims=True)
        pred = l2sum > 0
        l2sum_safe = tf.where(pred, l2sum, tf.ones_like(l2sum))

        return tf.cast(l2sum_safe, dtype)

    def forward(self, gradient):
        return self._clip_gradient(gradient)


# Original Paper: https://github.com/pseeth/autoclip
class AutoClipper:
    def __init__(self, percentile, history_size=10000):
        self.percentile = percentile
        self.norm_history = cp.zeros(history_size)
        self.history_size = history_size
        self.iteration = 0

    def _get_total_norm(self, gradient):
        norms = cp.array([])
        for idx, layer in enumerate(gradient):
            if isinstance(layer, cp.ndarray):
                if layer.size == 0:
                    continue

                if norms.dtype != layer.dtype:
                    norms = norms.astype(layer.dtype)

                norms = cp.append(norms, self._get_norm(layer.flatten()))
            else:
                norms = cp.append(norms, self._get_total_norm(layer))

        return self._get_norm(norms.flatten())

    def _get_norm(self, gradient):
        dtype = gradient.dtype
        gradient = tf.cast(gradient, tf.float64)

        l2sum = tf.math.reduce_sum(gradient * gradient, keepdims=True)
        pred = l2sum > 0
        l2sum_safe = tf.where(pred, l2sum, tf.ones_like(l2sum))

        return tf.cast(tf.squeeze(tf.where(pred, tf.math.sqrt(l2sum_safe), l2sum)), dtype)

    def _clip_gradient(self, gradient, clip_norm):
        for idx, layer in enumerate(gradient):
            if isinstance(layer, cp.ndarray):
                if layer.size == 0:
                    continue

                # Convert to float64 then back since it's more numerically stable
                gradient[idx] = tf.clip_by_norm(gradient[idx] + 1.0e-8, clip_norm)

            else:
                gradient[idx] = self._clip_gradient(layer, clip_norm)

        return gradient

    def forward(self, gradient):
        total_norm = self._get_total_norm(gradient)

        self.norm_history[self.iteration % self.history_size] = total_norm
        self.iteration += 1

        clip_norm = Processing.to_tensorflow(cp.percentile(cp.array(self.norm_history[:self.iteration]), q=self.percentile))

        print("[LOG] Clip Value:", clip_norm)
        
        return self._clip_gradient(gradient, clip_norm)