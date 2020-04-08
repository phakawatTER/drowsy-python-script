from scipy.spatial import distance as dist
import sys
import os
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
import pickle
import struct
import mappicosocket as ms
import serversocket as ss
import hashlib
import base64

current_dir = os.path.dirname(__file__)
def str2bool(v):
    v = v.lower()
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--test",default=False,type=str2bool, required=False, help="ENABLE TEST MODE")
ap.add_argument("-rs", "--read-serial", required=False,default=False,type=str2bool, help="ENABLE READING SERIAL")
ap.add_argument("-u", "--username", required=True,
                help="Please enter your username ")
ap.add_argument("-p", "--password", required=True,
                help="Please enter your password")
ap.add_argument("-r","--rotate",required=False,default=True,help="rotate sent image",type=str2bool)
ap.add_argument("-c", "--cam", required=False, default=0,
                type=int, help="Enter camera to be used")

args = vars(ap.parse_args())
ROTATE_IMAGE = args["rotate"]
TEST_BOOL = args["test"]
READING_SERIAL = args["read_serial"]


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
IS_AUTH = False
ear = 0
CO = 0
LPG = 0
SMOKE = 0
LATLNG = (0, 0)
DIRECTION = 0
SPEED = 0
PROGRAM_FINISHED = False
print("Authenticate to server ...")
import time
while not IS_AUTH:
    try:
        # email = input("Enter email: ")
        # password = getpass("Enter password: ")
        email = args["username"]
        password = args["password"]
        IS_AUTH = connect.authenticate(email, hashlib.sha512(
            bytes(f"{password}{SECRET}", encoding="utf-8")).hexdigest(),TRACKER_ID)
        if IS_AUTH:
            break
        else:
            print("Your email or password is incorrect")
        time.sleep(1)
    except Exception as err:
        print(err)
        print("failed to authenticate to server....")
        time.sleep(1)
ACCTIME = connect.acctime  # ACCTIME
UID = connect.uid  # USER ID
PUSH_TOKEN = connect.expoPushToken
server_socket = ss.ServerSokcet(uid=UID)
print(f"UID:{UID},ACCTIME:{ACCTIME}")

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
                    #proc.join()
        except Exception as err:
            pass


COUNTER = 0
TOTAL = 0
# TIME WHEN EYES ARE CLOSED AND NEXR SECOND
EYES_CLOSED_TIME = 0
NEXT_SECOND = 0
EYES_CLOSED_TIMER = 0

# Read data from serial port if READ_SERIAL flag is set
if READING_SERIAL:
    print("START UPDATE_GAS THREAD...")
    GAS_THREAD = Thread(target=updateGasData)
    GAS_THREAD.start()

START_TIME = time.time()
PREV_TIME = 0
TOTAL_FRAME = 0
AVG_FRAME = 10
FPS = 0
# WIDTH = 240
# HEIGHT = 140
# USE HUAWEI IP CAM
if args["cam"] >= 0 :
    cap = cv2.VideoCapture(args["cam"])
else:
    cap = cv2.VideoCapture(os.path.join(current_dir,"test_vdo.mp4"))

cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1280)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
print("[INFO] starting video stream thread...")
while True:
    try:

        ret, frame = cap.read()
        if ROTATE_IMAGE :
            frame = cv2.rotate(frame,cv2.ROTATE_180) # rotate frame by 180 degree
        HEIGHT, WIDTH, _ = frame.shape
        # set quality of image
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
        result, image = cv2.imencode(".jpg", frame, encode_param)
        img_as_text = base64.b64encode(image)
	# Send data through socket
        server_socket.sendImage(
            jpg_text=img_as_text,
            gas={
                "co": CO,
                "lpg": LPG,
                "smoke": SMOKE
            },
        )
        TOTAL_FRAME +=1 
        sys.stdout.write("\rFrame {} sent...".format(TOTAL_FRAME))
        sys.stdout.flush()         
    except Exception as err:
        print(err)
        pass

cap.release()
cv2.destroyAllWindows()  # destroy all windows
PROGRAM_FINISHED = True
os._eixt(0)
