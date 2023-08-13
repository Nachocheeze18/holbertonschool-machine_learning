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
        boxes = []
        box_confidences = []
        box_class_probs = []

        for output in outputs:
            grid_height, grid_width, anchor_boxes, _ = output.shape

            # Extract values from the output
            box_data = output[:, :, :, :4]
            box_confidence = output[:, :, :, 4:5]
            box_class_probs_raw = output[:, :, :, 5:]

            # Calculate boundary box coordinates relative to the original image
            grid_x = np.arange(grid_width)
            grid_y = np.arange(grid_height)
            grid_x, grid_y = np.meshgrid(grid_x, grid_y)
            x_offset = (grid_x + 0.5) / grid_width
            y_offset = (grid_y + 0.5) / grid_height
            width_scale = self.anchors[:, 0]
            height_scale = self.anchors[:, 1]
            box_data[..., :2] = (box_data[..., :2] + np.stack(
                (x_offset,
                 y_offset),
                 axis=-1)) / (grid_width,
                              grid_height)
            box_data[..., 2:4] = np.exp(box_data
                                        [..., 2:4]) * np.expand_dims(np.stack(
                (width_scale, height_scale),
                      axis=-1), axis=(0, 1, 2))

            # Calculate box probabilities
            box_class_probs = box_confidence * box_class_probs_raw

            # Append processed values to lists
            boxes.append(box_data)
            box_confidences.append(box_confidence)
            box_class_probs.append(box_class_probs)

        return boxes, box_confidences, box_class_probs
