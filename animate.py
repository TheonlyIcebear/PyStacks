import albumentations as A, multiprocessing, pickle, random, numpy as np, cv2, os
from PIL import ImageDraw, ImageTk, Image
from tkinter import Tk, Button, Label

from utils.layers import *
from utils.network import Network
from utils.optimizers import Adam
from utils.loss import *

from typing import Annotated, Any, Literal, Union, cast
import onnxruntime as ort, json


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

class Animate:
    def __init__(self, min_presence_score=0.87, max_iou=0.5):
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        session = ort.InferenceSession(
            "model-training-data.onnx",
            sess_options=so,
            providers=["DmlExecutionProvider", "CPUExecutionProvider"]
        )

        so.intra_op_num_threads = 4

        self.input_name = session.get_inputs()[0].name
        dummy = np.random.randn(1, 352, 352, 3).astype(np.float16)

        for _ in range(50):
            session.run(None, {
                self.input_name: dummy,
            })

        config = json.load(open("model-config.json", "rb"))

        self.session = session
        self.image_height, self.image_width = config["image_height"], config["image_width"]

        self.anchors = config["anchors"]
        self.classes = config["classes"]
        self.cells = config["grid_size"]
        self.dtype = np.float16

        self.anchor_dimensions = config["anchor_dimensions"]

        print(self.anchor_dimensions, "Anchor Dimensions")

        self.min_presence_score = min_presence_score
        self.max_iou = max_iou

        filename = os.listdir('Training')[0]

        root = Tk()
        root.geometry("550x300+300+150")
        root.resizable(width=True, height=True)
        image = ImageTk.PhotoImage(Image.open(f'Training\\{filename}').resize((self.image_width // 2, self.image_height // 2)))
        panel = Label(root, image=image)
        panel.pack()

        self.root = root
        self.panel = panel

    def draw_boxes(self, image, points, labels, color, confidence_scores=None):
        predicted_points = np.array(points.reshape((-1, 2, 2)))

        draw = ImageDraw.Draw(image)

        dimensions = np.array(image.size)

        for idx, (center, distances) in enumerate(predicted_points):
            top_left = (center - (distances / 2)) * dimensions
            bottom_right = (center + (distances / 2)) * dimensions

            draw.line([(top_left[0], top_left[1]), (bottom_right[0], top_left[1])], fill=color, width=5)
            draw.line([(top_left[0], bottom_right[1]), (bottom_right[0], bottom_right[1])], fill=color, width=5)

            draw.line([(top_left[0], top_left[1]), (top_left[0], bottom_right[1])], fill=color, width=5)
            draw.line([(bottom_right[0], top_left[1]), (bottom_right[0], bottom_right[1])], fill=color, width=5)

            if confidence_scores is not None:
                confidence_score = round(confidence_scores[idx], 2)
                label = labels[idx]
                text = f"{label}: {confidence_score}"
                draw.text((top_left[0], top_left[1] - 20), text, fill="white")

            elif labels is not None:
                label = labels[idx]
                text = f"{label}: True"
                draw.text((top_left[0], top_left[1] - 30), text, fill="blue")

    def parse_output(self, outputs):
        out_list = []
        for idx, output in enumerate(outputs):
            output = tf.cast(output, dtype=tf.float32)
            anchor_dimensions = tf.convert_to_tensor(self.anchor_dimensions[idx * self.anchors: (idx + 1) * self.anchors], dtype=output.dtype)

            grid_size = (2 ** (2 - idx)) * self.cells
            output = tf.reshape(output, [-1, 5 + self.classes])


            idxs = tf.range(tf.shape(output)[0])
            grid_y_index = (idxs // self.anchors) // grid_size
            grid_x_index = (idxs // self.anchors) % grid_size

            center_x = ((2 * output[:, 1] - 0.5) + tf.cast(grid_x_index, output.dtype)) / tf.cast(grid_size, output.dtype)
            center_y = ((2 * output[:, 2] - 0.5) + tf.cast(grid_y_index, output.dtype)) / tf.cast(grid_size, output.dtype)

            output = tf.concat([
                tf.expand_dims(output[:, 0], -1),
                tf.expand_dims(center_x, -1),
                tf.expand_dims(center_y, -1),
                output[:, 3:5],
                output[:, 5:]
            ], axis=-1)


            wh = (2 * output[:, 3:5])**2

            wh = tf.reshape(wh, [-1, self.anchors, 2]) * anchor_dimensions
            wh = tf.reshape(wh, [-1, 2])

            output = tf.concat([
                output[:, :3],
                wh,
                output[:, 5:]
            ], axis=-1)

            class_ids = tf.argmax(output[:, 5:], axis=-1, output_type=tf.int32)
            output = tf.concat([
                output[:, :5],
                tf.cast(tf.expand_dims(class_ids, -1), output.dtype)
            ], axis=-1)

            mask = output[:, 0] >= self.min_presence_score
            output = tf.boolean_mask(output, mask)

            out_list.append(output)


        output = tf.concat(out_list, axis=0)


        if tf.shape(output)[0] == 0:
            return tf.zeros([0], dtype=tf.float32), tf.zeros([0], dtype=tf.float32), tf.zeros([0], dtype=tf.float32)

        outputs = output

        print(output)

        selected_indices = tf.image.non_max_suppression(
            outputs[:, 1:5], outputs[:, 0], 10,
            iou_threshold=self.max_iou, score_threshold=self.min_presence_score
        )

        print(selected_indices, len(selected_indices))

        outputs = tf.gather(outputs, selected_indices)

        print(outputs)

        confidence_scores = outputs[:, 0]
        box_data = outputs[:, 1:5]
        classes = outputs[:, 5]

        return confidence_scores, box_data, classes

    def start(self):
        self.file_idx = 0
        self.update()
        self.root.mainloop()

    def update(self):
        files = os.listdir('Training')
        filename = files[self.file_idx]

        image = Image.open(f'Training\\{filename}')

        # image_width, image_height = image.size
        # image_width //= 2
        # image_height //= 2

        # image = image.resize((image_width, image_height))

        color_limit = 20

        locations_filename = f'annotations\\{filename.replace(".png", ".txt").replace(".jpg", ".txt")}'

        if os.path.exists(locations_filename):
            with open(locations_filename, "r+") as file:
                lines = file.read().splitlines()
                location_data = np.array([
                    [
                        float(value) for value in line.split(' ')
                    ] for line in lines
                ])
            
                
            objects_present = location_data.shape[0]


        else:
            objects_present = 0
            location_data = np.array([])

        with open("annotations\\classes.txt", "r+") as file:
            class_names = file.read().splitlines()

        location_data[..., 1:] += 10e-8
        location_data[..., :1] = np.clip(location_data[..., :1], 0, len(class_names)-1)
        
        if location_data.shape[0]:
            print(location_data)
            class_labels = [class_names[int(bbox[0])] for bbox in location_data]

        pil_image = Image.open(f'Training\\{filename}')

        try:
            image = cv2.resize(
                            cv2.cvtColor(
                                    cv2.imread(f'Training\\{filename}'), 
                                    cv2.COLOR_BGR2RGB
                                ), (self.image_width, self.image_height)
                            )
        except Exception as e:
            print(filename, e, "[LOG] BROKEN")
            self.file_idx += 1
            self.file_idx %= len(files)

            self.root.after(1, self.update)
            return
        
        print(location_data)
        
        classes = location_data[..., :1]

        augmentor = A.Compose([
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.75),
            A.RandomBrightnessContrast(
                brightness_limit=[-0.07, 0.07],
                contrast_limit=[-0.07, 0.07],
                p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=5,
                sat_shift_limit=5,
                val_shift_limit=5,
                p=0.5
            ),

            RandomScaledCenterCrop(
                min_scale=0.1, max_scale=0.45,
            ),
        ], bbox_params=A.BboxParams(format='yolo', min_visibility=0.1, label_fields=['class_labels']))

        augmented_result = augmentor(image=np.array(pil_image), bboxes=location_data[..., 1:], class_labels=classes)

        location_data = np.array(augmented_result['bboxes'])
        image = Image.fromarray(augmented_result['image']).resize((self.image_width, self.image_height))

        input_data = np.asarray(image).astype(self.dtype) / 255
        outputs = self.session.run(None, {
                self.input_name: input_data[None, ...],
            })
        confidence_scores, box_data, classes = self.parse_output(outputs)

        confidence_scores = confidence_scores.numpy()
        box_data = box_data.numpy()
        classes = classes.numpy()

        predicted_class_labels = [class_names[int(class_label)] for class_label in classes]

        print(box_data)
        print(location_data)

        print("==========================")
        print(predicted_class_labels)

        if location_data.shape[0]:
            print(class_labels)

        if location_data.shape[0]:
            self.draw_boxes(image, location_data, class_labels, "red")

        if box_data.shape[0]:
            print(box_data, predicted_class_labels, "green", confidence_scores)
            self.draw_boxes(image, box_data, predicted_class_labels, "green", confidence_scores)

        tk_image = ImageTk.PhotoImage(image)
        self.panel.configure(image=tk_image)
        self.panel.image = tk_image

        self.file_idx += 100 // np.random.randint(1, 100)
        self.file_idx %= len(files)

        self.root.after(1, self.update)

if __name__ == "__main__":

    animate = Animate()
    animate.start()