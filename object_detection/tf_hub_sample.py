"""
Very quick and very simple example of using pretrained efficient_det detector.
We're gonna to find cars on a sample image.
"""

import cv2
import tensorflow as tf
import numpy as np
import tensorflow_hub as hub


class TfHubSampleDetector:
    """
    Tensorflow Hub Detector sample class.
    """
    def __init__(self, det_type=0):
        """
        :param det_type: type of an efficient detector 0-7
        """
        if isinstance(det_type, int) and det_type >= 0 and det_type <= 7:
            self.det_type = det_type
        else:
            print("Wrong type of model, choose int number within range 0-7.")
            print("Set default type = 0")
            self.det_type = 0

        # loading detector
        self.det_source = f'https://tfhub.dev/tensorflow/efficientdet/d{self.det_type}/1'
        print(f"Downloading model from {self.det_source}")
        self.detector = hub.load(self.det_source)

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
        """

        # resize
        image_resized = cv2.resize(image, target_size)
        # convert bgr to rgb
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        # creating tensor and convert to uint8
        img_tensor = tf.convert_to_tensor(image_rgb, dtype=tf.uint8)
        # adding extra dimension
        img_tensor = tf.expand_dims(img_tensor, 0)

        return img_tensor