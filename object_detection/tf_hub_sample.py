"""
Very quick and very simple example of using pretrained efficient_det detector.
We're gonna to find cars on a sample image.
"""

import cv2
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import numpy as np


class TfHubSampleDetector:
    """
    Tensorflow Hub Detector sample class.
    """

    def __init__(self, coco_labels_path, det_type=0):
        """
        :param det_type: type of an efficient detector 0-7
        :param coco_labels_path: path to csv data of a coco labels
        """
        if isinstance(det_type, int) and 0 <= det_type <= 7:
            self.det_type = det_type
        else:
            print("Wrong type of model, choose int number within range 0-7.")
            print("Set default type = 0")
            self.det_type = 0

        # loading detector
        self.det_source = f'https://tfhub.dev/tensorflow/efficientdet/d{self.det_type}/1'
        print(f"Downloading model from {self.det_source}")
        self.detector = hub.load(self.det_source)

        # reading coco labels
        labels_coco = pd.read_csv(coco_labels_path, sep=';', index_col='ID')

        # take only 2017 categories
        self.labels_coco = np.array(labels_coco['OBJECT (2017 REL.)'])

    def load_image(self, path_to_image):
        """
        Loading image by given path
        :param path_to_image: place where image is stored
        """

        return cv2.imread(path_to_image)

    def prepare_image(self, image, target_size):
        """
        Preparation for detection.
        :param image: numpy array type of an image, 3 dims, HWC convention
        :param target_size: tuple containing width and height of an input image to resize
        :return img_tensor: image as 4D tensor
        """

        # resize
        image_resized = cv2.resize(image, target_size)
        # convert bgr to rgb
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        # creating tensor and convert to unsigned int 8
        img_tensor = tf.convert_to_tensor(image_rgb, dtype=tf.uint8)
        # adding extra dimension
        img_tensor = tf.expand_dims(img_tensor, 0)

        return img_tensor

    def detect_boxes(self, img_tensor, score_threshold):
        """
        Detection with effdet.
        :param img_tensor: ONE image as 4D tensor
        :param score_threshold: threshold value above which boxes are chosen
        :return (pred_scores, pred_boxes, pred_labels): detected objects, tuple
        """

        # detection
        detector_output = self.detector(img_tensor)

        # extract specific outputs
        pred_boxes = detector_output['detection_boxes'].numpy()[0]
        pred_scores = detector_output['detection_scores'].numpy()[0]
        pred_labels = detector_output['detection_classes'].numpy().astype('int')[0]

        # create chosen matrix based on score
        chosen_matrix = pred_scores > score_threshold

        # get data only for best scores
        pred_scores = pred_scores[chosen_matrix]
        pred_boxes = pred_boxes[chosen_matrix]
        pred_labels = pred_labels[chosen_matrix]

        # get classes names from coco
        pred_labels = self.labels_coco[pred_labels - 1]

        return pred_scores, pred_boxes, pred_labels
