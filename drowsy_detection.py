from scipy.spatial import distance as dist
from imutils.video import VideoStream
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
import connect as conn
import readserial
import sys
import socket
import pickle
import struct
import mappicosocket as ms
import serversocket as ss
import hashlib
import base64

test = hashlib.sha512(b"asdasf").hexdigest()
SECRET = ".YSWORD-DROWSY"
manager = Manager()
trip_data = manager.dict()
TRACKER_ID = "TLO12017000971"  # TER'S TRACKER


#mappico = ms.MappicoSocket(TRACKER_ID,trip_data)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--stream", required=False, help="ENABLE STREAMING")
ap.add_argument("-t", "--test", required=False, help="ENABLE TEST MODE")
ap.add_argument("-r", "--request", required=False, help="ENABLE HTTP REQUEST")
args = vars(ap.parse_args())
REQUEST_BOOL = args["request"]
TEST_BOOL = args["test"]
STREAM_BOOL = args["stream"]

if int(REQUEST_BOOL):
    try:
        print("http request enabled")
        connect = conn.Connect()
        rs = readserial.ReadSerial()
    except:
        pass


if TEST_BOOL:
    TRACKER_ID = "60000003"
    print(f"track id {TRACKER_ID}")
STREAM = 0
if STREAM_BOOL:
    STREAM = int(STREAM_BOOL)

if STREAM:
    END_POINT = input("END_POINT ADDRESS: ")
    END_POINT_PORT = int(input("END_POINT PORT: "))
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((END_POINT, END_POINT_PORT))
    print("SUCCESSFULLY CONNECTED TO ENDPOINT DEVICE")


SHAPE_PREDICTOR = "shape_predictor_68_face_landmarks.dat"
IS_AUTH = False
CO = 0
LPG = 0
SMOKE = 0
LATLNG = (0, 0)
DIRECTION = 0
SPEED = 0
PROGRAM_FINISHED = False
while not IS_AUTH:
    try:
        email = "test"
        password = "1234"
        IS_AUTH = connect.authenticate(email, hashlib.sha512(
            bytes(f"{password}{SECRET}", encoding="utf-8")).hexdigest())
        # IS_AUTH = connect.authenticate("phakawat.ter@gmail.com","123456789")
        if IS_AUTH:
            break
    except Exception as err:
        print(err)
        print("TRYING TO AUTHENTICATE....")
ACCTIME = connect.acctime  # ACCTIME
UID = connect.uid  # USER ID
server_socket = ss.ServerSokcet(uid=UID)
print(f"UID:{UID},ACCTIME:{ACCTIME}")

# START NEW PROCESS TO CONNECT WITH MAPPICO SOCKET...
mappico_socket_proc = Process(target=ms.MappicoSocket, args=(
    TRACKER_ID, trip_data, connect, UID, ACCTIME))
mappico_socket_proc.start()  # START SOCKET


def updateGasData():
    global LPG, CO, SMOKE
    REQ_TIME = time.time()
    while not PROGRAM_FINISHED:
        try:
            GAS_DATA = rs.readGas()
            CURRENT_TIME = time.time()
            if float(GAS_DATA[0]) >= 0:
                LPG = float(GAS_DATA[0])
            if float(GAS_DATA[1]) >= 0:
                CO = float(GAS_DATA[1])
            if float(GAS_DATA[2]) >= 0:
                SMOKE = float(GAS_DATA[2])
            if CO >= 70:
                TIME_DIFF = int(CURRENT_TIME - REQ_TIME)
                if int(TIME_DIFF) >= 5:
                    REQ_TIME = time.time()  # UPDATE REQTIME
                    proc = Process(target=connect.pushnotification, args=(
                        "Over CO", LATLNG, DIRECTION, SPEED))
                    proc.start()
                    proc.join()
        except Exception as err:
            pass


def updateCoordinate():
    global ACCTIME, CO, LPG, SMOKE, LATLNG
    while not PROGRAM_FINISHED:
        try:
            # print("OBD_UPDATED",trip_data)
            lat = trip_data["lat"]
            lon = trip_data["lon"]
            speed = trip_data["speed"]
            direction = trip_data["direction"]
            LATLNG = (lat, lon)
            DIRECTION = direction
            SPEED = speed
            wait_time = 2
            start_time = time.time()
            proc = Process(target=connect.updateTripData,
                           args=(CO, LATLNG, speed, direction))
            proc.start()
            proc.join()
            stop_time = time.time()
            if wait_time - (stop_time-start_time) > 0:
                time.sleep(wait_time - (stop_time-start_time) )
            print(stop_time - start_time)
            # connect.updateTripData(CO,LATLNG,speed,direction)
            # time.sleep(1.5)
        except Exception as err:
            # print(err)
            pass


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
EYE_AR_THRESH = 0.275
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
predictor = dlib.shape_predictor(SHAPE_PREDICTOR)

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
# start the video stream thread
print("[INFO] starting video stream thread...")


# UPDATE GAS AND GPS IF REQUEST BOOL IS SET
if REQUEST_BOOL or REQUEST_BOOL == None:
    # # CREATE AND START THREADS FOR UPDATING INFO IN BACKGROUND
    # print("START UPDATE_GAS THREAD...")
    # GAS_THREAD = Thread(target=updateGasData)
    # GAS_THREAD.start()
    print("START UPDATE_GPS THREAD...")
    COOR_THREAD = Thread(target=updateCoordinate)
    COOR_THREAD.start()


START_TIME = time.time()
TOTAL_FRAME = 10
AVG_FRAME = 10
WIDTH = 240
HEIGHT = 160
# USE HUAWEI IP CAM
#cap = cv2.VideoCapture("rtsp://admin:HuaWei123@192.168.2.3/LiveMedia/ch1/Media2")
# USE WEBCAM
cap = cv2.VideoCapture(0)
while True:
    try:
        ret, frame = cap.read()
        original_frame = frame
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

                # reset the eye frame counter
                COUNTER = 0
                EYES_CLOSED_TIME = 0
                NEXT_SECOND = 0

        # ALARM EVERY 2 SECOND SINCE EYES ARE CLOSED
            if (math.floor(EYES_CLOSED_TIME) % 2 == 0 and EYES_CLOSED_TIME >= 2):
                if (NEXT_SECOND < EYES_CLOSED_TIME):
                    NEXT_SECOND = math.floor(EYES_CLOSED_TIME) + 1
                    alarm_proc = Process(target=connect.pushnotification, args=(
                        "Drowsy", LATLNG, DIRECTION, SPEED))
                    alarm_proc.start()  # START ALARMIMG PROCESS
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
        result, image = cv2.imencode(".jpg", frame)

        # STREAM IMAGE THROUGH LAN
        if STREAM:
            try:
                data = pickle.dumps(image, 0)
                size = len(data)
                print(f"IMAGE SIZE OF {size} BYTES")
                s.sendall(struct.pack(">L", size)+data)
            except Exception as err:
                break
        try:
            img_as_text = base64.b64encode(image)
            server_socket.sendImage(img_as_text)
        except:
            pass

        # cv2.imshow("FRAME",frame)
        key = cv2.waitKey(1) & 0xff
        if key == 27:
            break
        CURRENT_TIME = time.time()
        TIME_DIFF = int(CURRENT_TIME - START_TIME)
        TOTAL_FRAME = TOTAL_FRAME + 1
        AVG_FRAME = TOTAL_FRAME / TIME_DIFF

    except Exception as err:
        print(err)
        pass

mappico_socket_proc.terminate()  # TERMINATING SOCKET PROCESS
cap.release()
cv2.destroyAllWindows()  # destroy all windows
PROGRAM_FINISHED = True
sys.exit(0)
# s.close()  # close connection
