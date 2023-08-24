#!/usr/bin/env python3
"""imports"""
import numpy as np
import os
import cv2
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

    def calculate_iou(self, box1, box2):
        """calculate the Intersection over Union (IoU)
        between two bounding boxes, which measures the
        extent of overlap between the two boxes in relation
        to their total area, returning a value indicating
        the degree of overlap."""
        x1_overlap = max(box1[0], box2[0])
        y1_overlap = max(box1[1], box2[1])
        x2_overlap = min(box1[2], box2[2])
        y2_overlap = min(box1[3], box2[3])

        if x2_overlap > x1_overlap and y2_overlap > y1_overlap:
            intersection_area = (x2_overlap - x1_overlap) * (y2_overlap
                                                             - y1_overlap)
            box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
            box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

            union_area = box1_area + box2_area - intersection_area
            iou = intersection_area / union_area
            return iou
        else:
            return 0.0

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """Applies non-maximum suppression to the filtered boxes."""
        unique_classes = np.unique(box_classes)
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        for cls in unique_classes:
            cls_indices = np.where(box_classes == cls)[0]
            cls_boxes = filtered_boxes[cls_indices]
            cls_box_scores = box_scores[cls_indices]

            while len(cls_boxes) > 0:
                max_score_idx = np.argmax(cls_box_scores)
                box_predictions.append(cls_boxes[max_score_idx])
                predicted_box_classes.append(cls)
                predicted_box_scores.append(cls_box_scores[max_score_idx])

                iou_scores = [self.calculate_iou(cls_boxes[max_score_idx], box)
                              for box in cls_boxes]
                to_remove = np.where(np.array(iou_scores) > self.nms_t)[0]
                cls_boxes = np.delete(cls_boxes, to_remove, axis=0)
                cls_box_scores = np.delete(cls_box_scores, to_remove)

        return (np.array(box_predictions),
                np.array(predicted_box_classes),
                np.array(predicted_box_scores))

    @staticmethod
    def load_images(folder_path):
        """Loads images from a folder and
        returns a tuple of images and their paths"""
        image_paths = [os.path.join(folder_path, img) for
                       img in os.listdir(folder_path) if
                       img.lower().endswith(('.jpg', '.jpeg', '.png'))]
        images = [cv2.imread(image_path) for image_path in image_paths]
        return images, image_paths

    def preprocess_images(self, images):
        """preprocesses a list of input images by resizing them
        to a desired size while maintaining their aspect ratios,
        normalizing pixel values, and returning the processed images
        along with their original shapes."""
        processed_images = []
        image_shapes = []

        i = 0  # Initialize loop counter
        while i < len(images):
            image = images[i]
            original_shape = image.shape[:2]
            image_shapes.append(original_shape)

            resized_image = cv2.resize(image, self.target_shape[:2], interpolation=cv2.INTER_CUBIC)
            scaled_image = resized_image / 255.0

            processed_images.append(scaled_image)
            
            i += 1  # Increment loop counter

        processed_images = np.array(processed_images)
        image_shapes = np.array(image_shapes)

        return processed_images, image_shapes

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            class_idx = box_classes[i]
            class_name = self.class_names[class_idx]
            score = box_scores[i]

            color = (255, 0, 0)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=2)

            text = f"{class_name} {score:.2f}"
            text_coords = (x1, y1 - 5)

            font_scale = 0.5
            font_color = (0, 0, 255)
            line_thickness = 1
            line_type = cv2.LINE_AA

            cv2.putText(image, text, text_coords, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, line_thickness, line_type)

        cv2.imshow(file_name, image)
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()

        if key == ord('s'):
            detections_dir = 'detections'
            os.makedirs(detections_dir, exist_ok=True)
            save_path = os.path.join(detections_dir, file_name)
            cv2.imwrite(save_path, image)

    def predict(self, folder_path):
        """folder path containing images, processes these images
        through a machine learning model, performs object detection,
        filters and selects the most confident predictions, displays
        the detected boxes on the images, and finally returns the
        predictions along with the image file paths.
        """
        predictions = []
        images, img_pth = self.load_images(folder_path)
        pimages, image_shapes = self.preprocess_images(images)
        out = self.model.predict(pimages)

        for i, img in enumerate(images):
            mat = [out[j][i] for j in range(len(out))]

            boxes, conf, prob = self.process_outputs(mat,
                                                     image_shapes[i])

            filtered_boxes, cls, box_scores = self.filter_boxes(boxes,
                                                                conf, prob)

            box_pred, pred_classes, pred_scores = self.non_max_suppression
            (filtered_boxes, cls, box_scores)

            predictions.append((box_pred,
                                pred_classes, pred_scores))

            self.show_boxes(img, box_pred, pred_classes,
                            pred_scores, img_pth[i].split('/')[-1])

        return predictions, img_pth