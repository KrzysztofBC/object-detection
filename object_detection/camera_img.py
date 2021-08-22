"""
using builtin camera to capture image and process it
"""


import cv2
import tensorflow as tf
from object_detection import tf_hub_sample

def process_img_from_camera():
    # create processing object
    tf_hub_det = tf_hub_sample.TfHubSampleDetector()

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, img = cap.read()
        if success:
            img = cv2.flip(img, 1)
            cv2.imshow("output", img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                image_tensor = tf_hub_det.prepare_image(img, (1024, 1024))
                output = tf_hub_det.detector(image_tensor)
                detection_scores = output['detection_scores'].numpy()
                print(f"detection scores: {detection_scores}")
                break


if __name__ == '__main__':
    process_img_from_camera()
