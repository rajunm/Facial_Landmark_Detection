# Facial landmarks detection using face_alignment library

import face_alignment
import cv2
import numpy as np
import time
import keyboard
from pynput.keyboard import Key, Listener

# Facial Landmark Detector
detector = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cpu')

input = cv2.imread("face3.jpg")
preds = detector.get_landmarks(input)   # its a list
print(np.shape(preds))
print(preds[0])

for point in preds[0]:
    cv2.circle(input, (int(point[0]),int(point[1])), 0, (0, 0, 255), 2)

cv2.imshow('Output',input)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Landmark detection from a camera stream
def decode_camera_capture():
    camera = cv2.VideoCapture(0)
    camera.set(3,640)
    camera.set(4,480)

    while not keyboard.is_pressed('esc'):
        success, img = camera.read()
        #print(success)
        preds = detector.get_landmarks(img)  # its a list
        for point in preds[0]:
            cv2.circle(img, (int(point[0]), int(point[1])), 0, (0, 0, 255), 2)

        cv2.imshow("Result", img)
        cv2.waitKey(1)
        if success == False:
            print("Escape key pressed. Terminating the detection")
            break

decode_camera_capture()
cv2.destroyAllWindows()