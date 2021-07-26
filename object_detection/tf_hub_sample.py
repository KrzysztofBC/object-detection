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
