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
cap = cv2.VideoCapture("rtsp://admin:HuaWei123@192.168.2.3/LiveMedia/ch1/Media2")
# USE WEBCAM
# cap = cv2.VideoCapture(0)
while True:
    try:
        ret, frame = cap.read()
        original_frame = frame
        # frame = cv2.resize(frame, (WIDTH, HEIGHT),
        #                    interpolation=cv2.INTER_AREA)  # RESIZE IMAGE

        # CONVERT INTO GREYSCALE IMAGE
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # detect faces in the grayscale frame
        rects = detector(gray, 0)
        # loop over the face detections
        for (index, rect)in enumerate(rects, start=0):
            if(index > 0):
                break
             # Finding points for rectangle to draw on face
            x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()
            # Drawing simple rectangle around found faces
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # cv2.rectangle(frame,rect[0],rect[1],(0,255,0),3)
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # extract the left and right eye coordinates, then use the
            # coordinates to compute the eye aspect ratio for both eyes
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            mouth = shape[mStart:mEnd]
            inner_mouth = shape[innmStart:innmEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            # average the eye aspect ratio together for both eyes
            ear = (leftEAR + rightEAR) / 2.0

            # compute the convex hull for the left and right eye, then
            # visualize each of the eyes
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            mouthHull = cv2.convexHull(mouth)
            innerMouthHull = cv2.convexHull(inner_mouth)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [innerMouthHull], -1, (0, 255, 0), 1)

            # check to see if the eye aspect ratio is below the blink
            # threshold, and if so, increment the blink frame counter
            if ear < EYE_AR_THRESH:
                COUNTER += 1
                # print("EYES ARE CLOSED {}".format(COUNTER))
                EYES_CLOSED_TIME = COUNTER / AVG_FRAME

            # otherwise, the eye aspect ratio is not below the blink
            # threshold
            else:
                # if the eyes were closed for a sufficient number of
                # then increment the total number of blinks
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    TOTAL += 1

                # reset the eye frame counter
                COUNTER = 0
                EYES_CLOSED_TIME = 0
                NEXT_SECOND = 0

            # Draw text
            cv2.putText(frame, "Eyes Aspect Ratio: {:.2f}".format(ear), (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 255, 0), 1)
            cv2.putText(frame, "EYES CLOSED: {:.2f} S".format(EYES_CLOSED_TIME),
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 255, 0), 1)
        cv2.putText(frame, "EAR THRESHOLD: {:.2f}".format(EYE_AR_THRESH),
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 255, 0), 1)
        cv2.putText(frame, "Latitude:{:.3f} Longitude:{:.3f}".format(
            LATLNG[0], LATLNG[1]), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 255, 0), 1)
        cv2.putText(frame, "CO:{} LPG:{} SMOKE:{}".format(
            CO, LPG, SMOKE), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 255, 0), 1)
        # frame = cv2.resize(frame, (480, 210), interpolation=cv2.INTER_AREA)
        # cv2.imshow("frame", frame)
        TOTAL_FRAME += 1
        print(f"time:{int(time.time()-START_TIME)}\ntotal frame:{TOTAL_FRAME}")
        if(int(time.time()-START_TIME)>= 60):
            break
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
