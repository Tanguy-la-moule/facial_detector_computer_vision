from imutils import face_utils
import numpy as np
import imutils
import os
import dlib
import cv2

trained_predictor_path = './predictor/shape_predictor_68_face_landmarks.dat'
image_path = './img/tanguy.jpg'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(trained_predictor_path)

image = cv2.imread(image_path)
bnw_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
rects = detector(bnw_image, 1)

