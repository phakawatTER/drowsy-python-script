from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import requests
import math
from threading import Thread
from pygame import mixer
from links import API_PUSH_NOTIFICATION, API_LOGIN
from getpass import getpass
import json
import geocoder
import socket
import struct
import pickle
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p",
                "--shape-predictor",
                required=True,
                help="path to facial landmark predictor")
args = vars(ap.parse_args())
# START SERVER
print("WAITING FOR CLIENT")
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(("192.168.1.46", 8009))  # open port 8009 for connection
s.listen(1)  # ALLOW ONE USER AT A TIME
clientsocket, address = s.accept()
print("CLIENT CONNECTED")
print("WATING FOR AUTHENTICATION")
uid = clientsocket.recv(4096) # WAITING FOR CLIENT TO AUTHENTICATE
uid = uid.decode("utf-8")
print(uid)
print("READY TO OPERATE")


mixer.init()
mixer.set_num_channels(1)
latitude = 0
longitude = 0
#  PUSH NOTIFICATION TO FIREBASE
def pushNotification():
    global uid
    data = {
        "event":"Drowsy",
        "user_id": uid,
        "latlng": [latitude,longitude]
    }
    print(API_PUSH_NOTIFICATION)
    req = requests.post(url=API_PUSH_NOTIFICATION, data=data, timeout=1)
    # print out http request response
    print(req.text)


def playAlert():
    busy = mixer.music.get_busy()
    if not busy:
        # numberOfChannel = mixer.get_num_channels()
        mixer.music.load('./alarm_sounds/alarm.mp3')
        mixer.music.play()
        req_process = Thread(target=pushNotification)
        req_process.start()


def stopAlert():
    # wait 1 second to stop
    time.sleep(1)
    mixer.music.stop()


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
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 3

# initialize the frame counters and the total number of blinks
COUNTER = 0
TOTAL = 0
# TIME WHEN EYES ARE CLOSED AND NEXR SECOND
EYES_CLOSED_TIME = 0
NEXT_SECOND = 0
# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
# start the video stream thread
print("[INFO] starting video stream thread...")
# face_cascade = cv2.CascadeClassifier(
#     "cascades/haarcascade_frontalface_default.xml")

START_TIME = time.time()
TOTAL_FRAME = 10
AVG_FRAME = 10
WIDTH = 320
HEIGHT = 240
data = b""
payload_size = struct.calcsize(">Ldd")
while True:
    try:
        # READ DATABYTE FROM RASPBERRY PI
        # Reconstruct image from recieved byte
        while len(data) < payload_size:
            data += clientsocket.recv(4096)
            # print("data")
        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        unpacked_data = struct.unpack(">Ldd", packed_msg_size)
        msg_size = unpacked_data[0]
        latitude = unpacked_data[1]
        longitude = unpacked_data[2]
        while len(data) < msg_size:
            data += clientsocket.recv(4096)
        frame_data = data[:msg_size]
        data = data[msg_size:]
        # inp = np.asarray(bytearray(frame_data), dtype=np.uint8)
        frame = pickle.loads(frame_data, fix_imports=True, encoding="bytes")
        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
        # READ FRAME FROM CAMERA
        # ret,frame = cap.read()
        frame = cv2.resize(frame, (WIDTH, HEIGHT),
                           interpolation=cv2.INTER_AREA)  # RESIZE IMAGE
        # CONVERT INTO GREYSCALE IMAGE
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # detect faces in the grayscale frame
        rects = detector(gray, 0)

        # loop over the face detections
        for rect in rects:
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # extract the left and right eye coordinates, then use the
            # coordinates to compute the eye aspect ratio for both eyes
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            # average the eye aspect ratio together for both eyes
            ear = (leftEAR + rightEAR) / 2.0

            # compute the convex hull for the left and right eye, then
            # visualize each of the eyes
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

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

                busy = mixer.music.get_busy()
                # stop alert if mixer is playing
                # start new thread to stop mixer
                if busy:
                    stopMixer = Thread(target=stopAlert)
                    # stopMixer.start()
                # reset the eye frame counter
                COUNTER = 0
                EYES_CLOSED_TIME = 0
                NEXT_SECOND = 0
        # ALARM EVERY 1 SECOND SINCE EYES ARE CLOSED
            if (math.floor(EYES_CLOSED_TIME) % 1 == 0 and EYES_CLOSED_TIME >= 1):
                if (NEXT_SECOND < EYES_CLOSED_TIME):
                    NEXT_SECOND = math.floor(EYES_CLOSED_TIME) + 1
                    req_process = Thread(target=pushNotification)
                    req_process.start()
                    # playAlert()
            # Draw text
            cv2.putText(frame, "Eyes Aspect Ratio: {:.2f}".format(ear), (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)
            cv2.putText(frame, "EYES CLOSED: {:.2f} S".format(EYES_CLOSED_TIME),
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)
        cv2.putText(frame, "EAR THRESHOLD: {:.2f}".format(EYE_AR_THRESH),
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)
        cv2.putText(frame, "Latitude:{:.3f} Longitude:{:.3f}".format(
            latitude, longitude), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)

        cv2.imshow("frame", frame)
        # END FOR LOOP
        key = cv2.waitKey(1) & 0xFF
        if (key == ord("w")):
            EYE_AR_THRESH += 0.01
        if (key == ord("s")):
            EYE_AR_THRESH -= 0.01
        # if the `q` key was pressed, break from the loop
        if key == 27:
            break
        
        CURRENT_TIME = time.time()
        TIME_DIFF = int(CURRENT_TIME - START_TIME)
        TOTAL_FRAME = TOTAL_FRAME + 1
        AVG_FRAME = TOTAL_FRAME / TIME_DIFF
        # print(TOTAL_FRAME / TIME_DIFF)

    except Exception as err:
        print(err)
        pass

cv2.destroyAllWindows()  # destroy all windows
s.close()  # close connection
