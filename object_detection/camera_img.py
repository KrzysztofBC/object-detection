"""
using builtin camera to capture image and process it
"""


import cv2
import matplotlib.pyplot as plt
from object_detection import tf_hub_sample
import numpy as np


def draw_detections(img, predictions):
    # unpack tuple
    pred_scores, pred_boxes, pred_labels = predictions

    # inverse transformations for boxes coordinates
    height = img.shape[0]
    width = img.shape[1]
    pred_boxes[:, 0] = pred_boxes[:, 0] * height
    pred_boxes[:, 1] = pred_boxes[:, 1] * width
    pred_boxes[:, 2] = pred_boxes[:, 2] * height
    pred_boxes[:, 3] = pred_boxes[:, 3] * width
    pred_boxes = pred_boxes.astype(np.int32)

    # draw boxes and scores on detected objects
    how_many_boxes = 0
    for score, (ymin, xmin, ymax, xmax), label in zip(pred_scores, pred_boxes, pred_labels):
        score_txt = f'{100 * score:.4}%'
        img_boxes = cv2.rectangle(img, (xmin, ymax), (xmax, ymin), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img_boxes, label, (xmin, ymin - 40), font, 2.0, (255, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img_boxes, score_txt, (xmin, ymin - 10), font, 1.0, (255, 0, 0), 2, cv2.LINE_AA)

    return img_boxes

def process_img_from_camera():
    # create processing object
    tf_hub_det = tf_hub_sample.TfHubSampleDetector(coco_labels_path='../samples/labels.csv')

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, img = cap.read()
        if success:
            cv2.imshow("output", img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                image_tensor = tf_hub_det.prepare_image(img, (1024, 1024))
                predictions = tf_hub_det.detect_boxes(image_tensor, 0.5)
                print("is there some prediction?:\n", predictions)
                img_boxes = draw_detections(img, predictions)
                cv2.imwrite("/home/nvidia/Projekty/object_detection/samples/output.jpg", img_boxes)
                break


if __name__ == '__main__':
    process_img_from_camera()
