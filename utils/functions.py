from functools import partial
import numpy as np, cupy as cp, time
import tensorflow as tf
from tensorflow.experimental.dlpack import from_dlpack, to_dlpack
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

    def to_tensorflow(array):
        array = from_dlpack(cp.ascontiguousarray(array).toDlpack())
        return array

    def to_cupy(array):
        array = cp.from_dlpack(to_dlpack(array))
        return array

class ClipGradient:
    def __init__(self, clip_norm):
        self.clip_norm = clip_norm

    def _clip_gradient(self, gradient):
        for idx, layer in enumerate(gradient):
            if not isinstance(layer, (list, tuple)):
                if len(layer) == 0:
                    continue

                # Convert to float64 then back since it's more numerically stable
                gradient[idx] = tf.clip_by_norm(gradient[idx] + 1.0e-8, self.clip_norm)

            else:
                gradient[idx] = self._clip_gradient(layer)

        return gradient

    def forward(self, gradient):
        return self._clip_gradient(gradient)


# Original Paper: https://github.com/pseeth/autoclip
class AutoClipper:
    def __init__(self, percentile, history_size=10000):
        self.percentile = percentile
        self.norm_history = tf.Variable(tf.zeros(history_size, dtype=tf.float32))
        self.history_size = history_size
        self.iteration = tf.Variable(0, dtype=tf.int32)

    def _get_total_norm(self, gradient):
        norms = tf.constant([], dtype=tf.float32)
        
        for layer in gradient:
            if not isinstance(layer, (list, tuple)):
                if tf.size(layer) == 0:
                    continue
                
                norms = tf.concat([norms, [self._get_norm(tf.reshape(layer, [-1]))]], axis=0)
            else:
                norms = tf.concat([norms, [self._get_total_norm(layer)]], axis=0)

        return self._get_norm(tf.reshape(norms, [-1]))

    def _get_norm(self, gradient):
        gradient = tf.cast(gradient, tf.float64)
        l2sum = tf.reduce_sum(gradient * gradient)
        
        pred = l2sum > 0
        l2sum_safe = tf.where(pred, l2sum, tf.constant(1.0, dtype=tf.float64))
        
        return tf.cast(tf.where(pred, tf.sqrt(l2sum_safe), l2sum), gradient.dtype)

    def _clip_gradient(self, gradient, clip_norm):
        clipped_gradient = []
        
        for layer in gradient:
            if not isinstance(layer, (list, tuple)):
                if tf.size(layer) == 0:
                    clipped_gradient.append(layer)
                    continue

                clipped_layer = tf.clip_by_norm(layer + 1.0e-8, tf.cast(clip_norm, layer.dtype))
                clipped_gradient.append(clipped_layer)
            else:
                clipped_gradient.append(self._clip_gradient(layer, clip_norm))
        
        return clipped_gradient

    def forward(self, gradient):
        total_norm = self._get_total_norm(gradient)

        update_index = tf.math.mod(self.iteration, self.history_size)
        self.norm_history = tf.tensor_scatter_nd_update(
            self.norm_history, 
            [[update_index]], 
            [total_norm]
        )
        self.iteration.assign_add(1)

        sorted_norms = tf.sort(self.norm_history[:self.iteration])
        clip_norm = Processing.to_tensorflow(cp.percentile(cp.array(self.norm_history[:self.iteration]), q=self.percentile))

        print("[LOG] Clip Value:", clip_norm)

        return self._clip_gradient(gradient, clip_norm)