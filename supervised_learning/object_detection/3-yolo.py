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
        filtered_boxes = None
        box_classes_list = []
        box_scores_list = []
        for i in range(len(boxes)):
            new_box_score = box_confidences[i] * box_class_probs[i]
            new_box_class = np.argmax(new_box_score, axis=-1)
            new_box_score = np.max(new_box_score, axis=-1)

            box_classes_list.append(new_box_class.reshape(-1))
            box_scores_list.append(new_box_score.reshape(-1))

        box_scores_all = np.concatenate(box_scores_list)
        box_classes_all = np.concatenate(box_classes_list)
        box_mask = box_scores_all >= self.class_t

        filtered_boxes = np.concatenate(
            [box.reshape(-1, 4) for box in boxes], axis=0)
        filtered_boxes = filtered_boxes[box_mask]

        box_classes = box_classes_all[box_mask]
        box_scores = box_scores_all[box_mask]

        return filtered_boxes, box_classes, box_scores

    def non_max_suppression(self, filtered_boxes,
                            box_classes, box_scores):
        """Applies non-maximum suppression to the filtered boxes"""
        unique_classes = np.unique(box_classes)
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        for cls in unique_classes:
            idx = np.where(box_classes == cls)
            class_boxes = filtered_boxes[idx]
            class_box_scores = box_scores[idx]

            while len(class_boxes) > 0:
                max_score_idx = np.argmax(class_box_scores)
                box_predictions.append(class_boxes[max_score_idx])
                predicted_box_classes.append(cls)
                predicted_box_scores.append(class_box_scores[max_score_idx])

                iou = [self.intersection_over_union(class_boxes[max_score_idx],
                                                    box)for box in class_boxes]
                to_remove = np.where(np.array(iou) > self.nms_t)
                cls_boxes = np.delete(cls_boxes, to_remove, axis=0)
                cls_box_scores = np.delete(cls_box_scores, to_remove, axis=0)

        return (np.array(box_predictions),
                np.array(predicted_box_classes),
                np.array(predicted_box_scores))
