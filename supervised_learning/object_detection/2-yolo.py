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
        """Process Darknet outputs"""
        boxes = []
        box_confidences = []
        box_class_probs = []

        for i in range(len(outputs)):
            boxes.append(outputs[i][..., :4])
            box_confidences.append(1 / (1 + np.exp(-outputs[i][..., 4:5])))
            box_class_probs.append(1 / (1 + np.exp(-outputs[i][..., 5:])))

        image_height, image_width = image_size

        for i in range(len(boxes)):
            grid_width = outputs[i].shape[1]
            grid_height = outputs[i].shape[0]
            anchor_boxes = outputs[i].shape[2]
            for j in range(grid_height):
                for k in range(grid_width):
                    for l in range(anchor_boxes):
                        tx, ty, tw, th = boxes[i][j, k, l]
                        pw, ph = self.anchors[i][l]
                        bx = (1 / (1 + np.exp(-tx))) + k
                        by = (1 / (1 + np.exp(-ty))) + j
                        bw = pw * np.exp(tw)
                        bh = ph * np.exp(th)
                        bx /= grid_width
                        by /= grid_height
                        bw /= self.model.input.shape[1].value
                        bh /= self.model.input.shape[2].value
                        x1 = (bx - (bw / 2)) * image_width
                        y1 = (by - (bh / 2)) * image_height
                        x2 = (bx + (bw / 2)) * image_width
                        y2 = (by + (bh / 2)) * image_height
                        boxes[i][j, k, l] = [x1, y1, x2, y2]

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """ YOLO model outputs by discarding low-confidence detections,
        then applies non-maximum suppression to retain only the most
        confident and non-overlapping bounding boxes, returning the
        filtered bounding boxes, their corresponding class indices,
        and confidence scores."""
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for i in range(len(boxes)):
            # Flatten the arrays for easier manipulation
            flat_boxes = boxes[i].reshape(-1, 4)
            flat_confidences = box_confidences[i].reshape(-1)
            flat_class_probs = box_class_probs[i].reshape(-1, len(self.class_names))

            # Apply confidence threshold
            box_mask = flat_confidences >= self.class_t

            filtered_boxes.extend(flat_boxes[box_mask])
            box_scores.extend(flat_confidences[box_mask])
            box_classes.extend(np.argmax(flat_class_probs[box_mask], axis=1))

        filtered_boxes = np.array(filtered_boxes)
        box_classes = np.array(box_classes)
        box_scores = np.array(box_scores)

        # Apply non-maximum suppression to remove overlapping boxes
        selected_indices = K.backend.image.non_max_suppression(
            filtered_boxes, box_scores, max_output_size=50,
            iou_threshold=self.nms_t
        )

        filtered_boxes = filtered_boxes[selected_indices]
        box_classes = box_classes[selected_indices]
        box_scores = box_scores[selected_indices]

        return filtered_boxes, box_classes, box_scores
