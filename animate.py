import albumentations as A, multiprocessing, pickle, numpy as np, cv2, os
from PIL import ImageDraw, ImageTk, Image
from tkinter import Tk, Button, Label

from utils.layers import *
from utils.network import Network
from utils.optimizers import Adam
from utils.loss import *

class Animate:
    def __init__(self, min_presence_score=0.65, max_iou=0.3):
        save_data = pickle.load(open('model-training-data.json', 'rb'))

        network = Network(dtype=np.float16)
        network.load(save_data)

        self.network = network
        self.image_height, self.image_width = network.model[0].output_shape[:-1]

        self.objects = 4
        self.anchors = network.loss_functions[-1].anchors
        self.cells = network.loss_functions[-1].grid_size.numpy()

        self.anchor_dimensions = cp.array(network.loss_functions[0].anchor_dimensions)

        self.min_presence_score = min_presence_score
        self.max_iou = max_iou

        filename = os.listdir('gameplay')[0]

        root = Tk()
        root.geometry("550x300+300+150")
        root.resizable(width=True, height=True)
        image = ImageTk.PhotoImage(Image.open(f'gameplay\\{filename}').resize((self.image_width // 2, self.image_height // 2)))
        panel = Label(root, image=image)
        panel.pack()

        self.root = root
        self.panel = panel

    def draw_boxes(self, image, points, color, confidence_scores=None):
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
                draw.text((top_left[0], top_left[1] - 20), str(confidence_score), fill="white")

    def parse_output(self, outputs):
        _outputs = cp.empty((0, 5))
        for idx, output in enumerate(outputs):
            
            anchor_dimensions = self.anchor_dimensions[idx * self.anchors: (idx + 1) * self.anchors]

            grid_size = (2 ** idx) * self.cells
            output = cp.array(output).reshape(-1, 5)
            
            idx = cp.arange(output.shape[0])

            grid_y_index = (idx // self.anchors) % grid_size
            grid_x_index = (idx // self.anchors) // grid_size

            relative_center_x = output[:, 1]
            relative_center_y = output[:, 2]

            center_x = (grid_x_index + relative_center_x) / grid_size
            center_y = (grid_y_index + relative_center_y) / grid_size

            output[:, 1] = center_x
            output[:, 2] = center_y
            output[:, 3:] = (cp.exp(output[:, 3:]).reshape(-1, self.anchors, 2) * anchor_dimensions).reshape(-1, 2)

            _outputs = cp.concatenate((_outputs, output))

        outputs = _outputs

        object_presence_scores = outputs[:, 0]
        present_boxes_indices = object_presence_scores >= self.min_presence_score
        object_presence_scores = object_presence_scores[present_boxes_indices]
        unprocessed_box_data = outputs[present_boxes_indices][object_presence_scores.argsort()][::-1]

        box_data = []
        confidence_scores = []
        while len(unprocessed_box_data) > 0:
            current = unprocessed_box_data[0]
            unprocessed_box_data = unprocessed_box_data[1:]
            
            iou = Processing.iou(current[1:], unprocessed_box_data[:, 1:])
            box_data.append(current[1:].get())
            confidence_scores.append(current[0].get())
            
            unprocessed_box_data = unprocessed_box_data[iou < self.max_iou]

        box_data = np.array(box_data)
        confidence_scores = np.array(confidence_scores)
        return confidence_scores, box_data

    def start(self):
        self.file_idx = 0
        self.update()
        self.root.mainloop()

    def update(self):
        files = os.listdir('gameplay')
        filename = files[self.file_idx]

        image = Image.open(f'gameplay\\{filename}')

        # image_width, image_height = image.size
        # image_width //= 2
        # image_height //= 2

        # image = image.resize((image_width, image_height))

        color_limit = 20

        locations_filename = f'annotations\\{filename.replace(".png", ".txt")}'

        if os.path.exists(locations_filename):
            with open(locations_filename, "r+") as file:
                lines = file.read().splitlines()
                location_data = np.array([
                    [
                        np.float64(value) for value in line.split(' ')[1:]
                    ] for line in lines
                ], dtype=np.float16)

            midpoints = location_data[:, 2] * location_data[:, 3]
                
            location_data = location_data[midpoints.argsort()[::-1]]
            objects_present = location_data.shape[0]

        else:
            location_data = np.array([])
            objects_present = 0

        location_data += 10e-8

        pil_image = Image.open(f'gameplay\\{filename}')

        image = cv2.resize(
            cv2.cvtColor(
                    cv2.imread(f'gameplay\\{filename}'), 
                    cv2.COLOR_BGR2RGB
                ), (self.image_width, self.image_height)
            )

        input_data = np.asarray(image).astype(self.network.dtype) / 255
        outputs = self.network.forward(input_data[None, ...], training=False)
        confidence_scores, box_data = self.parse_output(outputs)

        self.draw_boxes(pil_image, location_data, "red")
        self.draw_boxes(pil_image, box_data, "green", confidence_scores)

        pil_image = ImageTk.PhotoImage(pil_image)
        self.panel.configure(image=pil_image)
        self.panel.image = pil_image

        self.file_idx += 1
        self.file_idx %= len(files)

        self.root.after(1, self.update)

if __name__ == "__main__":

    animate = Animate()
    animate.start()