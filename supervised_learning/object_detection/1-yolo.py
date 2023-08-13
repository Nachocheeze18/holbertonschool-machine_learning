#!/usr/bin/env python3
"""imports"""
import numpy as np
import tensorflow.keras as K
from tensorflow.keras.models import load_model


class Yolo:
    """yolo class"""
    def __init__(self, model_path, classes_path,
                 class_threshold, nms_threshold, anchors):
        """It is designed to load a pre-trained YOLO model,
        preprocess input images, run object detection on the
        images, and post-process the model's outputs to obtain
        the detected objects and their associated information."""
        self.model = self._load_yolo_model(model_path)
        self.class_names = self._load_classes(classes_path)
        self.class_t = class_threshold
        self.nms_t = nms_threshold
        self.anchors = anchors

    def _load_yolo_model(self, model_path):
        """Loads the YOLO model from a file"""
        return load_model(model_path)

    def _load_classes(self, classes_path):
        """Loads the classes from a file"""
        with open(classes_path, 'r') as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def process_outputs(self, outputs, image_size):
        """Processes YOLO model outputs and returns processed information"""
        boxes = []
        box_confidences = []
        box_class_probs = []

        for output in outputs:
            grid_height, grid_width, anchor_boxes, _ = output.shape
            processed_boxes = np.zeros_like(output[..., :4])
            processed_box_confidences = output[..., 4:5]
            processed_box_class_probs = output[..., 5:]

            for y in range(grid_height):
                for x in range(grid_width):
                    for b in range(anchor_boxes):
                        t_x, t_y, t_w, t_h = output[y, x, b, :4]
                        box_x = (x + self._sigmoid(t_x)) / grid_width
                        box_y = (y + self._sigmoid(t_y)) / grid_height
                        box_w = self.anchors[b][0] * np.exp(t_w) / self.model.input.shape[1].value
                        box_h = self.anchors[b][1] * np.exp(t_h) / self.model.input.shape[2].value
                        
                        x1 = (box_x - box_w / 2) * image_size[1]
                        y1 = (box_y - box_h / 2) * image_size[0]
                        x2 = x1 + box_w * image_size[1]
                        y2 = y1 + box_h * image_size[0]

                        processed_boxes[y, x, b, :] = [x1, y1, x2, y2]

            boxes.append(processed_boxes)
            box_confidences.append(processed_box_confidences)
            box_class_probs.append(processed_box_class_probs)

        return boxes, box_confidences, box_class_probs

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))