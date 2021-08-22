"""
using builtin camera to capture image and process it
"""


import cv2


def process_img_from_camera():
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, img = cap.read()
        if success:
            img = cv2.flip(img, 1)
            cv2.imshow("output", img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break


if __name__ == '__main__':
    process_img_from_camera()
