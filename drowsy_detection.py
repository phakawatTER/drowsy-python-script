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
import connect as conn
import readserial
import sys
import pickle
import struct
import mappicosocket as ms
import serversocket as ss
import hashlib
import base64


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--test", required=False, help="ENABLE TEST MODE")
ap.add_argument("-r", "--serial", required=False, help="ENABLE READING SERIAL")
ap.add_argument("-u", "--username", required=True,
                help="Please enter your username ")
ap.add_argument("-p", "--password", required=True,
                help="Please enter your password")

args = vars(ap.parse_args())
TEST_BOOL = args["test"]
READING_SERIAL = False
if args["serial"]:
    READING_SERIAL = int(args["serial"])


connect = conn.Connect()
if READING_SERIAL:
    rs = readserial.ReadSerial()


STREAM = 0
notification_push = False
SECRET = ".YSWORD-DROWSY"
manager = Manager()
trip_data = manager.dict()
TRACKER_ID = "TLO12017000971"  # TER'S TRACKER
if TEST_BOOL:
    TRACKER_ID = "60000003"
print(f"TRACKER ID: {TRACKER_ID}")
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
        # email = input("Enter email: ")
        # password = getpass("Enter password: ")
        email = args["username"]
        password = args["password"]
        IS_AUTH = connect.authenticate(email, hashlib.sha512(
            bytes(f"{password}{SECRET}", encoding="utf-8")).hexdigest())
        if IS_AUTH:
            break
        else:
            print("Your email or password is incorrect")

    except Exception as err:
        print(err)
        print("failed to authenticate to server....")
ACCTIME = connect.acctime  # ACCTIME
UID = connect.uid  # USER ID
PUSH_TOKEN = connect.expoPushToken
server_socket = ss.ServerSokcet(uid=UID)
mappico_socket = Process(target=ms.MappicoSocket, args=(
    TRACKER_ID, trip_data, connect, UID, ACCTIME, PUSH_TOKEN))
mappico_socket.start()
print(f"UID:{UID},ACCTIME:{ACCTIME}")


def pushnotification(event, coords, direction, speed):
    print("push notification to server....")
    notification_push = True
    try:
        connect.pushnotification(event, coords, direction, speed)
    except:
        pass
    notification_push = False


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
        wait_time = 2
        try:
            lat = trip_data["lat"]
            lon = trip_data["lon"]
            speed = trip_data["speed"]
            direction = trip_data["direction"]
            LATLNG = (lat, lon)
            start_time = time.time()
            connect.updateTripData(CO, LATLNG, speed, direction)
            stop_time = time.time()
            # to make sure that the the value is updated every 2 seconds
            if wait_time - (stop_time-start_time) > 0:
                time.sleep(wait_time - (stop_time-start_time))
            # print(
            #     f"coords:({lat},{lon})\nspeed:{speed}\ndirection:{direction}")
            # print("___________________________________________________________")
        except Exception as err:
            print(err)
            time.sleep(wait_time)
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
ear = 0.00
EYE_AR_THRESH = 0.22
EYE_AR_CONSEC_FRAMES = 3
# @server_socket.on(f"trip_update_{UID}")
# def on_trip_update(data):
#     print("this is eye aspect ratio ",str(EYE_AR_THRESH))
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
# # UPDATE GAS AND GPS IF REQUEST BOOL IS SET
# # CREATE AND START THREADS FOR UPDATING INFO IN BACKGROUND
if READING_SERIAL:
    print("START UPDATE_GAS THREAD...")
    GAS_THREAD = Thread(target=updateGasData)
    GAS_THREAD.start()
print("START UPDATE_GPS THREAD...")
COORDS_THREAD = Thread(target=updateCoordinate)
COORDS_THREAD.start()

START_TIME = time.time()
PREV_TIME = 0
TOTAL_FRAME = 10
AVG_FRAME = 10
FPS = 0
WIDTH = 240
HEIGHT = 140
# USE HUAWEI IP CAM
# cap = cv2.VideoCapture(
    # "rtsp://admin:HuaWei123@192.168.1.38/LiveMedia/ch1/Media2")
# USE WEBCAM
cap = cv2.VideoCapture(0)
print("[INFO] starting video stream thread...")

while True:
    try:
        ret, frame = cap.read()
        # CONVERT INTO GREYSCALE IMAGE
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # detect faces in the grayscale frame
        start = time.time()
        faces = detector(gray, 0)
        stop = time.time()
        print("spent {} for face detection".format(round(stop-start, 2)))
        start = time.time()
        for index, face in enumerate(faces, start=0):
            if index > 0:
                break
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            # Drawing simple rectangle around found faces
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = predictor(gray, face)
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
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 2)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 2)
            cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 2)
            cv2.drawContours(frame, [innerMouthHull], -1, (0, 255, 0), 2)

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
            if (math.floor(EYES_CLOSED_TIME) % 2 == 0 and EYES_CLOSED_TIME >= 1):
                if (NEXT_SECOND < EYES_CLOSED_TIME) and not notification_push:
                    NEXT_SECOND = math.floor(EYES_CLOSED_TIME) + 1
                    alarm_thread = Thread(target=pushnotification, args=(
                        "Drowsy", LATLNG, DIRECTION, SPEED))
                    alarm_thread.start()  # START ALARMIMG PROCESS
        # # Draw text
        # cv2.putText(frame, "Eyes Aspect Ratio: {:.2f}".format(ear), (300, 30),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 255, 0), 1)
        # cv2.putText(frame, "EYES CLOSED: {:.2f} S".format(EYES_CLOSED_TIME),
        #             (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 255, 0), 1)
        # cv2.putText(frame, "EAR THRESHOLD: {:.2f}".format(EYE_AR_THRESH),
        #             (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 255, 0), 1)
        # cv2.putText(frame, "Latitude:{:.3f} Longitude:{:.3f}".format(
        #     LATLNG[0], LATLNG[1]), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 255, 0), 1)
        # cv2.putText(frame, "CO:{} LPG:{} SMOKE:{}".format(
        #     CO, LPG, SMOKE), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 255, 0), 1)
        stop = time.time()
        print("spent {} for face".format(round(stop-start, 2)))
        
        # Update current time and calculate fps and avg. fps.
        CURRENT_TIME = time.time()
        TIME_DIFF = int(CURRENT_TIME - START_TIME)
        TOTAL_FRAME = TOTAL_FRAME + 1
        AVG_FRAME = TOTAL_FRAME / TIME_DIFF
        if PREV_TIME == TIME_DIFF:
            FPS += 1
        else:
            print(
                f"Average FPS:{round(AVG_FRAME,2)} , TOTAL TIME :{TIME_DIFF} second(s) , FPS:{FPS}")
            FPS = 0
        PREV_TIME = TIME_DIFF

        # Resize frame because the image should not be too large for transmitting through socket..
        frame = cv2.resize(frame, (WIDTH, HEIGHT),
                           interpolation=cv2.INTER_AREA)
        result, image = cv2.imencode(".jpg", frame)
        try:
            img_as_text = base64.b64encode(image)
            server_socket.sendImage(
                jpg_text=img_as_text,
                coor=LATLNG, ear=ear,
                gas={
                    "co": CO,
                    "lpg": LPG,
                    "smoke": SMOKE
                },
                fps=AVG_FRAME,
                eye_close_time=EYES_CLOSED_TIME
            )
        except Exception as err:
            print(err)
            pass

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
