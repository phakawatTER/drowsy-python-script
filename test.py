from scipy.spatial import distance as dist
# from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import math
from threading import Thread
from multiprocessing import Process, Manager
from getpass import getpass
import sys
import tkinter as tk

SHAPE_PREDICTOR = "shape_predictor_68_face_landmarks.dat"
IS_AUTH = False
CO = 0
LPG = 0
SMOKE = 0
LATLNG = (0, 0)
DIRECTION = 0
SPEED = 0
PROGRAM_FINISHED = False


def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])
    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # return the eye aspect ratio
    return ear


# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold
ear = 0.00
EYE_AR_THRESH = 0.275
EYE_AR_CONSEC_FRAMES = 3

# initialize the frame counters and the total number of blinks
COUNTER = 0
TOTAL = 0
# TIME WHEN EYES ARE CLOSED AND NEXR SECOND
EYES_CLOSED_TIME = 0
NEXT_SECOND = 0
EYES_CLOSED_TIMER = 0
# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(SHAPE_PREDICTOR)

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
(innmStart, innmEnd) = face_utils.FACIAL_LANDMARKS_IDXS["inner_mouth"]
# start the video stream thread
print("[INFO] starting video stream thread...")

START_TIME = time.time()
PREV_TIME = 0
TOTAL_FRAME = 0
AVG_FRAME = 10
FPS = 0
WIDTH = 240
HEIGHT = 160
# USE HUAWEI IP CAM
cap = cv2.VideoCapture("rtsp://admin:HuaWei123@192.168.1.38/LiveMedia/ch1/Media2")
# USE WEBCAM
# cap = cv2.VideoCapture(0)
while True:
    try:
        ret, frame = cap.read()
        original_frame = frame
        print("eiei")
        cv2.imshow("frame",frame)
        key = cv2.waitKey(1) & 0xff
        if key == 27:
            break

    except Exception as err:
        print(err)
        pass
    
    

cap.release()
cv2.destroyAllWindows()  # destroy all windows
PROGRAM_FINISHED = True
sys.exit(0)
