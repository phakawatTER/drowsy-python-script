import time
import play_beep as pb
import datetime
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
import lcddriver
import base64

try:
    TRIP_VDO_DIR = "/home/phakawat/Desktop/trip_vdo"
    display = lcddriver.lcd() # lcd display
    beep  = pb.beep(interval=4.0,beep_duration=0.15) # beep object
    display.lcd_clear() # initially clean lcd

    current_dir = os.path.dirname(__file__)

    def str2bool(v): # parse string to boolean 
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
        TRACKER_ID = "60000003" # TRACKER FOR TESTING
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
    print("Authenticating to server ...")
    display.lcd_display_string("AUTHENTICATING",1)
    attempt_count = 0 
    while not IS_AUTH: # keep authenticate until authentication is completed...
        attempt_count += 1 
        try:
            # email = input("Enter email: ")
            # password = getpass("Enter password: ")
            email = args["username"]
            password = args["password"]
            IS_AUTH = connect.authenticate(email, hashlib.sha512(
                bytes(f"{password}{SECRET}", encoding="utf-8")).hexdigest(),TRACKER_ID)
            if IS_AUTH:
                display.lcd_clear()
                display.lcd_display_string("AUTHENTICATION",1)
                display.lcd_display_string("SUCCESSED!!!",2)
                time.sleep(1.5)
                break
            else:
                print("Your email or password is incorrect")
            time.sleep(1)
        except Exception as err:
            display.lcd_display_string("ATTEMPT: ({})".format(attempt_count),2) 
            print("failed to authenticate to server....")
            time.sleep(1)
    ACCTIME = connect.acctime  # ACCTIME
    UID = connect.uid  # USER ID
    PUSH_TOKEN = connect.expoPushToken # expo push token for notification request
    server_socket = ss.ServerSokcet(uid=UID) # middle sever socket instance
    print(f"UID:{UID},ACCTIME:{ACCTIME}")


    display.lcd_clear()
    display.lcd_display_string("WELCOME BACK",1)
    display.lcd_display_string("PHAKAWAT!",2)
    time.sleep(1.5)
    # update gas data task => this will be run as new a process
    def updateGasData():
        global LPG, CO, SMOKE
        REQ_TIME = time.time()
        def warn_driver():
            global START_ALERT_TIME
            START_ALERT_TIME = datetime.datetime.now() # update latest start alert time
            line1 = "WARNING !!!"
            event = "OVER CO"
            line2 = event
            display.lcd_clear()
            display.lcd_display_string(line1,1) # write screen line 1
            display.lcd_display_string(line2,2) # write screen line 2
            beep.play_beep_loop()

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
                if CO >= 80:
                    TIME_DIFF = int(CURRENT_TIME - REQ_TIME)
                    if int(TIME_DIFF) >= 5:
                        REQ_TIME = time.time()  # UPDATE REQTIME
                        warn = Thread(target=warn_driver)
                        warn.start()
                        proc = Process(target=connect.pushnotification, args=(
                            "Over CO", LATLNG, DIRECTION, SPEED))
                        proc.start()
            except Exception as err:
                pass
            time.sleep(2) # check gas every 2 seconds

    # Read data from serial port if READ_SERIAL flag is set
    if READING_SERIAL:
        GAS_THREAD = Thread(target=updateGasData)
        GAS_THREAD.start()

    TOTAL_FRAME = 0 # Number of frame sent...

    if args["cam"] >= 0 :
        cap = cv2.VideoCapture(args["cam"])
    else:
        cap = cv2.VideoCapture(os.path.join(current_dir,"test_vdo.mp4"))


    VDO_SHAPE = (1280,720)
    # Set shape of read video
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VDO_SHAPE[0])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, VDO_SHAPE[1])
    vdo_file = "{}.mp4".format(ACCTIME)
    vdo_file_dest = os.path.join(TRIP_VDO_DIR,vdo_file)
    record_stream_process = Process(target=os.system,args=("record_stream {}".format(vdo_file_dest),))
    record_stream_process.start() # start process to record stream
    print("[INFO] starting video stream thread...")
    display.lcd_clear()
    display.lcd_display_string("START STREAMING",1)
    time.sleep(2)
    display.lcd_clear()
    display.lcd_display_string("WAITING FOR ",1)
    display.lcd_display_string("SERVER RESPONSE",2)


    SHOW_ALERT_DURATION = 3 # duration to show alert on LCD screen
    START_ALERT_TIME = None
    trip_process_data = None # this will be updated by data received from server

    print("process_data_{}".format(UID))
    @server_socket.on("process_data_{}".format(UID))
    def on_data_recv(data):
        try:
            CURRENT_TIME = datetime.datetime.now()
            timediff = CURRENT_TIME - START_ALERT_TIME
            should_write = timediff.total_seconds() > SHOW_ALERT_DURATION # timediff in seconds compared with alert duration 
        except Exception as err:
            should_write = True
        if should_write:
            mar = data["mar"] # Mouth Aspect Ratio
            ear = data["ear"] # Ear Aspect Ration
            gas = data["gas"] # co ,lpg,smoke in ppm unit
            co = gas["co"]
            line1 = "EAR:{0:.2f}MAR:{1:.2f}".format(ear,mar)
            line2 = "CO:{0:.2f} ppm".format(co)
            display.lcd_clear()
            display.lcd_display_string(line1,1) # write screen line 1
            display.lcd_display_string(line2,2) # write screen line 2

    @server_socket.on("warn_driver_{}".format(UID))
    def on_warn_driver(data):
        global START_ALERT_TIME
        START_ALERT_TIME = datetime.datetime.now()
        line1 = "WARNING !!!"
        event = data["event"] # occured event!
        def warn_driver(event):
            if event == "Dangerous Eye Close":
                event = "Eye Close"
            line2 = event
            display.lcd_clear()
            display.lcd_display_string(line1,1) # write screen line 1
            display.lcd_display_string(line2,2) # write screen line 2
            beep.play_beep_loop()
        warn = Thread(target=warn_driver,args=(event,))
        warn.start()


    LOOP_DELAY = 0.06
    LAST_SAVE_VDO_TIME = None

    ## Start Streaming Data to Server
    while True:
        try:

            ret, frame = cap.read() # read image input
            if ROTATE_IMAGE :
                frame = cv2.rotate(frame,cv2.ROTATE_180) # rotate frame by 180 degree
            if LAST_SAVE_VDO_TIME == None:
                LAST_SAVE_VDO_TIME = datetime.datetime.now()
            timediff = datetime.datetime.now() - LAST_SAVE_VDO_TIME
            seconddiff = timediff.seconds # second difference
#            # set quality of image
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
            result, image = cv2.imencode(".jpg", frame, encode_param)
            img_as_text = base64.b64encode(image)
#            # Send data through socket
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
            time.sleep(LOOP_DELAY) ## ADDING DELAY TO AVOID SENDING TOO MUCH DATA
        except Exception as err:
            print(err)
    cap.release()
    PROGRAM_FINISHED = True
    os._eixt(0)

except KeyboardInterrupt:
    try:
        cap.release()
    except:
        pass
    PROGRAM_FINISHED = True
    import Jetson.GPIO as GPIO
    GPIO.cleanup() # clear I/O pins
    display.lcd_clear() # clear screen

"""
TODO: Play alarm sound on sleepiness or fatigue driving is detected 
TODO: Recieve realtime  processing data from middle server 
"""
