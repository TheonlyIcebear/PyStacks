from functools import partial
import numpy as np, cupy as cp, time
import tensorflow.compat.v1 as tf
from PIL import Image, ImageTk, ImageDraw
from tensorflow.experimental.dlpack import from_dlpack, to_dlpack

class Processing:
    def iou(boxes1, boxes2):
        boxes1_top_left = (boxes1[..., :2] - (boxes1[..., 2:] / 2))
        boxes1_bottom_right = (boxes1[..., :2] + (boxes1[..., 2:] / 2))

        boxes2_top_left = (boxes2[..., :2] - (boxes2[..., 2:] / 2))
        boxes2_bottom_right = (boxes2[..., :2] + (boxes2[..., 2:] / 2))

        top_left_x = cp.maximum(boxes1_top_left[..., 0:1], boxes2_top_left[..., 0])  # Broadcasting along batch size
        top_left_y = cp.maximum(boxes1_top_left[..., 1:2], boxes2_top_left[..., 1])

        bottom_right_x = cp.minimum(boxes1_bottom_right[..., 0:1], boxes2_bottom_right[..., 0])
        bottom_right_y = cp.minimum(boxes1_bottom_right[..., 1:2], boxes2_bottom_right[..., 1])

        intersection_width = cp.maximum(0, bottom_right_x - top_left_x)
        intersection_height = cp.maximum(0, bottom_right_y - top_left_y)
        intersection_area = intersection_width * intersection_height

        boxes1_area = (boxes1_bottom_right[..., 0] - boxes1_top_left[..., 0]) * (boxes1_bottom_right[..., 1] - boxes1_top_left[..., 1])
        boxes2_area = (boxes2_bottom_right[..., 0] - boxes2_top_left[..., 0]) * (boxes2_bottom_right[..., 1] - boxes2_top_left[..., 1])

        union_area = boxes1_area[..., None] + boxes2_area - intersection_area

        iou = intersection_area / cp.maximum(union_area, 1e-6)  # Avoid division by zero

        return iou

    def to_tensorflow(array):
        array = from_dlpack(cp.ascontiguousarray(array).toDlpack())
        return array

    def to_cupy(array):
        array = cp.from_dlpack(to_dlpack(array))
        return array