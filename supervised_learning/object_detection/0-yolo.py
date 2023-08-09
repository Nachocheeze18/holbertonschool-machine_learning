#!/usr/bin/env python3
"""imports"""
import numpy as np
import tensorflow.keras as K


class Yolo:
    """yolo class"""
    def __init__(self, model_path, classes_path,
                 class_threshold, nms_threshold, anchors):
        """It is designed to load a pre-trained YOLO model,
        preprocess input images, run object detection on the
        images, and post-process the model's outputs to obtain
        the detected objects and their associated information."""
        self.model = K.models.load_model(model_path)
        with open(classes_path) as file:
            class_names = file.read()
        self.class_names = class_names.replace("\n", "|").split("|")[:-1]
        self.class_threshold = class_threshold
        self.nms_threshold = nms_threshold
        self.anchors = anchors

    def _load_classes(self, classes_path):
        with open(classes_path, 'r') as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names
    
    def _load_model(self, model_path):
        model = tf.compat.v2.saved_model.load(model_path)
        return model