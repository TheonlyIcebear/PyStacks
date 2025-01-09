from functools import partial
import numpy as np, cupy as cp, time
import tensorflow.compat.v1 as tf
from PIL import Image, ImageTk, ImageDraw
from tensorflow.experimental.dlpack import from_dlpack, to_dlpack

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

        intersection_width = api.maximum(0, bottom_right_x - top_left_x)
        intersection_height = api.maximum(0, bottom_right_y - top_left_y)
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
            if isinstance(layer, cp.ndarray):
                if layer.size == 0:
                    continue

                # Convert to float64 then back since it's more numerically stable
                dtype = gradient[idx].dtype
                gradient[idx] = gradient[idx].astype(cp.float64)
                gradient[idx] = self._clip_by_norm(gradient[idx] + 1.0e-8, self.clip_norm)
                gradient[idx] = cp.nan_to_num(gradient[idx].astype(dtype), 0, 0, 0)

            else:
                gradient[idx] = self._clip_gradient(layer)

        return gradient

    def _clip_by_norm(self, gradient, clip_norm):
        return gradient * cp.minimum(1.0, clip_norm / (self._get_norm(gradient) + 1e-8))
        
    def _get_norm(self, gradient):
        dtype = gradient.dtype
        gradient = gradient.astype(cp.float64)

        l2sum = cp.sum(gradient * gradient)
        l2sum = cp.nan_to_num(l2sum, 1, 1, 1)
        l2norm = cp.sqrt(l2sum)

        return l2norm.astype(dtype)

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
        gradient = gradient.astype(cp.float64)
        
        l2sum = cp.sum(gradient * gradient)
        l2sum = cp.nan_to_num(l2sum, 1, 1, 1)
        l2norm = cp.sqrt(l2sum)

        return l2norm.astype(dtype)

    def _clip_gradient(self, gradient, clip_norm):
        for idx, layer in enumerate(gradient):
            if isinstance(layer, cp.ndarray):
                if layer.size == 0:
                    continue

                # Convert to float64 then back since it's more numerically stable
                dtype = gradient[idx].dtype
                gradient[idx] = gradient[idx].astype(cp.float64)
                gradient[idx] = self._clip_by_norm(gradient[idx] + 1.0e-8, clip_norm)
                gradient[idx] = cp.nan_to_num(gradient[idx].astype(dtype), 0, 0, 0)

            else:
                gradient[idx] = self._clip_gradient(layer, clip_norm)

        return gradient

    def _clip_by_norm(self, gradient, clip_norm):
        l2norm = self._get_norm(gradient)
        return gradient * cp.minimum(1.0, clip_norm / (l2norm + 1e-8))

    def forward(self, gradient):
        total_norm = self._get_total_norm(gradient)

        self.norm_history[self.iteration % self.history_size] = total_norm
        self.iteration += 1

        clip_norm = cp.percentile(self.norm_history[:self.iteration], q=self.percentile)

        print("[LOG] Clip Value:", clip_norm)
        
        return self._clip_gradient(gradient, clip_norm)