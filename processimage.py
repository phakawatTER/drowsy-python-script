import socketio
import cv2
from imutils import face_utils
import datetime 
import base64
import numpy as np
from scipy.spatial import distance as dist
from tensorflow.keras.models import load_model
from tensorflow.keras.backend import set_session
import tensorflow as tf
import math
import time
import connect as conn
import argparse
import threading
from middlesocket import MiddleServerSocket
from ddestimator import ddestimator
import sys
from multiprocessing import Process
from threading import Thread
import mappicosocket as ms
import schedule
import base64
import math
import os
import pickle
import json
from cal_face import cal_face
from camera_api import API_MIDDLE_SERVER_DOWNLOAD_VDO
import requests

# Keras uses Tensorflow Backend
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
set_session(sess) # set session for keras backend

current_directory = os.path.dirname(__file__)
root_directory = "/home/phakawat/"
local_endpoint = "http://localhost:4000"

INPUT_SHAPE = 128
TEST_VDO = os.path.join(root_directory,"test_vdo_night.mp4")
print(TEST_VDO)
VIDEO_SHAPE = (720,480) 

class ProcessImage(socketio.Client):
    DNN_MODEL_FILE = os.path.join(
        current_directory, "opencv_face_detector_uint8.pb")
    DNN_MODEL_CONFIG = os.path.join(
        current_directory, "opencv_face_detector.pbtxt")

    EYECLOSE_THRESHOLD = 0.22
    REQUEST_INTERVAL = 5 # seconds
    GAZE_REQUEST_TIME = None
    EYECLOSE_REQUEST_TIME = None
    FATIGUE_REQUEST_TIME = None

    def __init__(self, ip=local_endpoint,tracker_id="", uid="", acctime="", token="",pushtoken="",test=False,test_vdo_path=TEST_VDO,use_request=True,use_vdo=True):
        socketio.Client.__init__(self)
        self.use_vdo = use_vdo 
        self.use_request = use_request
        self.TEST_MODE = test 
        self.face_check = False
        self.face_known = False
        self.prev_val = 0
        self.current_val = 0
        self.uid = uid
        self.acctime = acctime
        self.token = token
        self.pushtoken = pushtoken
        self.img_data = ""
        self.START_TIME = time.time() # use for determining duration of the trip
        self.AVG_EAR = 0.35 # initial value for eye aspect ratio
        self.MAR = 0.1 # initial value for mouth aspect ratio
        if self.TEST_MODE or self.use_vdo: # if test mode is enabled
            print("LOADING ---> ",test_vdo_path)
            self.cap = cv2.VideoCapture(test_vdo_path)
        self.api_connect = conn.Connect( 
            token=token, uid=uid, acctime=acctime, expoPushToken=pushtoken)
        self.cal_face = cal_face() # cal_face object to store log of facial expression
        self.ddestimator  = ddestimator() # ddestimator object
        # data for to be streamed ...
        self.trip_data = {
            "speed":0,
            "direction":0,
            "uid":uid,
            "ear":0,
            "coor":(0,0),
            "gas":{"co":0,"lpg":0,"smoke":0}
        }
        self.data_is_recv = False # bool to check if first data packet is received...

       # create directory if it's not exist
        if not os.path.exists(os.path.join(root_directory, "trip_vdo", self.uid)):
            os.makedirs(os.path.join(root_directory, "trip_vdo", self.uid))
        self.vdo_writer = cv2.VideoWriter(os.path.join(root_directory, "trip_vdo", self.uid, "{}.mp4".format(
            acctime)), cv2.VideoWriter_fourcc(*'MP4V'),10,VIDEO_SHAPE)
        # SCHEDULER JOBS...
        # schedule to check if data is recieved
        # if data is not recieved anymore so proceed to terminate the process
        schedule.every(15).seconds.do(self.checkIfAlive)
        if not self.TEST_MODE: # update trip data to database if not test mode
            schedule.every(2).seconds.do(self.update_obd_data)
        schedule.every(2).seconds.do(self.send_processed_data)

        @self.event
        def connect():
            print("Connected to server...")

        @self.on("image_{}".format(uid)) # receive image streamed from JETSON NANO BOARD
        def get_image(data):
            self.data_is_recv = True
            self.current_val += 1
            b64_img = data["jpg_text"]
            del data["jpg_text"]
            self.trip_data.update(data)
            img_buffer = base64.b64decode(b64_img)
            jpg_as_np = np.frombuffer(img_buffer, dtype=np.uint8)
            self.img_data = cv2.imdecode(jpg_as_np, flags=1)

        @self.event
        def disconnect():
            print("Disconnected from server...")

        self.connect(ip) # connect to local server socket
        self.middle_server_socket = MiddleServerSocket() # connect to  middle server socket
        self.mpc_socket = ms.MappicoSocket(
            tracker_id, self.trip_data, connect=self.api_connect, uid=uid, acctime=acctime, pushToken=pushtoken) # connect to mappico socket

    def update_obd_data(self):
        try:
            print(self.trip_data["gas"])
            coor = self.trip_data["coor"]
            direction = self.trip_data["direction"]
            co = self.trip_data["gas"]["co"]
            speed = self.trip_data["speed"]
            data = {"acctime":self.acctime,"uid":self.uid,"latlng":coor,"speed":speed,"co":co,"direction":direction}
            self.emit("obd_update_data",data) # emit data to local server
        except Exception as err:
            print(err)
            pass

    def warn_driver(self,event):
        print("emit warn driver")
        data = {"uid":self.uid,"event":event}
        self.emit("warn_driver",data)

    def send_processed_data(self):
        print("emit processed data")
        try:
            if self.prev_val != self.current_val or self.TEST_MODE:
                data = self.trip_data.copy()
                data["uid"] = self.uid # add uid into data
                del data["jpg_text"] # delete jpg
                print([key for key in data]) # print dictionary key
                self.emit("process_data",data)
        except Exception as err:
            print(err)
    def load_landmark_model(self,path):
        self.landmark_model = load_model(path)

    def predict_face_landmark(self,face,show_exec_time=False):
        start = time.time()
        me = np.array(face)/255
        h,w,c = me.shape
        me = me.reshape((1,h,w,c))
        y_test = self.landmark_model.predict(me)
        label_points = (np.squeeze(y_test))
        stop = time.time()
        exec_time = round(stop-start,4)
        if show_exec_time:
            print("Execution time spent {} (s)".format(exec_time))
        return label_points

    def checkIfAlive(self):
        if self.data_is_recv:
            if self.prev_val != self.current_val:
                self.prev_val = self.current_val
            else:
                self.program_finish()

    def program_finish(self):
        print("Program finished...")
        self.disconnect()
        cv2.destroyAllWindows()
        self.vdo_writer.release()
        print("Successfully save trip vdo file as mp4...")
        request_middleserver_download = requests.post(url=API_MIDDLE_SERVER_DOWNLOAD_VDO,data={"uid":self.uid,"file":"{}.mp4".format(self.acctime)})
        print(request_middleserver_download.text)
        os._exit(0)

    def draw_face(self,frame,face_shape,origin,key_points):
        x_points = key_points[::2]
        y_points = key_points[1::2]
        f_width,f_height  = face_shape
        x1,y1 = origin
        scale_x = f_width/INPUT_SHAPE
        scale_y = f_height/INPUT_SHAPE
        x_points = np.array(x_points)*scale_x+x1
        x_points = x_points.astype(int)
        y_points = np.array(y_points)*scale_y+y1
        y_points = y_points.astype(int)
        coords = list(zip(x_points, y_points))
        for i, coord in enumerate(coords, start=0):
            x, y = coord
            COLOR = (255, 133, 71)
            if i in range(33, 51):  # face region
                COLOR = (0, 255, 0)
            elif i in range(51, 55):  # nose1
                COLOR = (25, 25, 25)
            elif i in range(55, 60):  # nose2
                COLOR = (0, 255, 255)
            elif i in range(60, 76):  # eye bounds
                COLOR = (255, 0, 255)
            elif i in range(76, 88):  # outer mouth
                COLOR = (255, 255, 148)
            elif i in range(88, 96):  # inner mouth
                COLOR = (0, 0, 255)
            elif i in range(96, 98):  # pupils
                COLOR = (0, 255, 0)
            cv2.circle(frame, (x, y), 0, COLOR, 2)

        return frame

    def draw_contour(self,frame,coords,color=(0,255,0)):
        # reshape given coordinates
        region = np.array(coords).reshape((-1, 1, 2)).astype(np.int32)
        frame = cv2.drawContours(frame, [region], -1,color, 2)
        return frame


    def draw_bounding_box(self,frame,coords):
        """Draw bounding box over driver face"""
        euler, rotation, translation = self.ddestimator.est_head_dir(coords)
        _, _, gaze_D = self.ddestimator.est_gaze_dir(coords)
        bc_2d_coords = self.ddestimator.proj_head_bounding_cube_coords(rotation, translation)
        gl_2d_coords = self.ddestimator.proj_gaze_line_coords(
             rotation, translation, gaze_D)
        frame = self.ddestimator.draw_gaze_line(frame, gl_2d_coords, (0, 255, 0), gaze_D)
        frame = self.ddestimator.draw_bounding_cube(frame, bc_2d_coords, (255, 0, 0), euler)
        return frame

    def process_img(self,frame,process_frame):
        blob = cv2.dnn.blobFromImage(process_frame, 1.0, (300, 300), [
            104, 117, 123], False, False)
        self.net.setInput(blob)
        faces = self.net.forward()
        HEIGHT, WIDTH, _ = frame.shape
        gray_dnn = cv2.cvtColor(process_frame, cv2.COLOR_BGR2GRAY)
        face_found = False
        for i in range(faces.shape[2]):
            if i > 0:
                break
            confidence = faces[0, 0, i, 2]
            # print("Face confidence {}".format(confidence))
            if confidence > 0.7:
                face_found = True
                x1 = int(faces[0, 0, i, 3] * WIDTH)
                y1 = int(faces[0, 0, i, 4] * HEIGHT)
                x2 = int(faces[0, 0, i, 5] * WIDTH)
                y2 = int(faces[0, 0, i, 6] * HEIGHT)
                original_face = frame[y1:y2 ,x1:x2] # get only face
                face = cv2.resize(original_face,(INPUT_SHAPE,INPUT_SHAPE),interpolation=cv2.INTER_AREA) # resize image 
                face_width = x2-x1 
                face_height = y2-y1 
                scale_x = face_width/INPUT_SHAPE
                scale_y = face_height/INPUT_SHAPE
                key_points = self.predict_face_landmark(face,show_exec_time=False) # predict facial landmark position
                x_points = (np.array(key_points[::2])*scale_x + x1).astype(np.int32)
                y_points = (np.array(key_points[1::2])*scale_y+ y1).astype(np.int32)
                coords = list(zip(x_points,y_points))
                avg_ear,_,_ = self.cal_face.cal_ear_98(coords)
                mar = self.cal_face.cal_mar_98(coords)

                #  check facial status 
                yawn = self.cal_face.check_yawn(duration=3) # check if user is yawning ...
                is_yawning,_,_= yawn
                eye_close ,eye_close_ot = self.cal_face.check_eye(duration=1)
#                sys.stdout.write(f"\rFACIAL LOG YAWN_STATUS:{yawn} EYE_CLOSE:{eye_close} EAR:{avg_ear} MAR:{mar}")
                sys.stdout.flush()

                # get facial log to determi
                ear_log = self.cal_face.get_log(self.cal_face.ear_log,period=2) # check back 5 seconds for ear log
                mar_log = self.cal_face.get_log(self.cal_face.mar_log,period=3) # check back 5seconds for mar log

                # check if ear avg ear over time
                ear_ot = np.mean(ear_log.values,axis=0)[1]
#                print("eye aspect ratio (over time):{}".format(ear_ot))
                self.AVG_EAR = avg_ear
                self.MAR = mar
                # draw left eye contour
                frame = self.draw_contour(frame,coords[60:68])
                # draw right eye contour
                frame = self.draw_contour(frame,coords[68:76])
                # draw outer mouth contour
                frame = self.draw_contour(frame,coords[76:88])
                # draw inner mouth contour
                frame = self.draw_contour(frame,coords[88:96])
                # draw bounding cube
                frame = self.draw_bounding_box(frame,coords)
                # draw facail landmark points
                frame = self.draw_face(frame,(int(x2-x1),int(y2-y1)),(x1,y1),key_points)
                
                # EYES WARNING
                if eye_close_ot  <  ProcessImage.EYECLOSE_THRESHOLD :
                    event = "Dangerous Eye Close"
                    latlng = self.trip_data["coor"]
                    speed = self.trip_data["speed"]
                    direction = self.trip_data["direction"]
                    if ProcessImage.EYECLOSE_REQUEST_TIME == None:
                        ProcessImage.EYECLOSE_REQUEST_TIME = datetime.datetime.now()
                    now = datetime.datetime.now()
                    timediff = now - ProcessImage.EYECLOSE_REQUEST_TIME
                    second_diff = timediff.seconds
                    if second_diff >= ProcessImage.REQUEST_INTERVAL:
                        self.warn_driver(event)
                        ProcessImage.EYECLOSE_REQUEST_TIME = now # update latest request time
                        print("Requesting for notification event:{} , time difference:{}".format(event,second_diff))
                        #make request for mobile notification
                        if self.use_request:  # request if use_request flag is set
                            request_thread = Thread(target=self.api_connect.pushnotification,args=(event,latlng,direction,speed),
                                kwargs={"uid":self.uid,"acctime":self.acctime,"pushToken":self.pushtoken})
                            request_thread.start() # start request process

                # FATIGUE WARNING
                if is_yawning:
                    event = "Fatigue"
                    latlng = self.trip_data["coor"]
                    speed = self.trip_data["speed"]
                    direction = self.trip_data["direction"]
                    if ProcessImage.FATIGUE_REQUEST_TIME == None:
                        ProcessImage.FATIGUE_REQUEST_TIME = datetime.datetime.now()
                    now = datetime.datetime.now()
                    timediff = now - ProcessImage.FATIGUE_REQUEST_TIME
                    second_diff = timediff.seconds
                    if second_diff >= ProcessImage.REQUEST_INTERVAL:
                        self.warn_driver(event)
                        ProcessImage.FATIGUE_REQUEST_TIME = now # update latest request time
                        print("Requesting for notification event:{} , time difference:{}".format(event,second_diff))
                        #make request for mobile notification
                        if self.use_request: # request if use_request flag is set
                            request_thread = Thread(target=self.api_connect.pushnotification,args=(event,latlng,direction,speed),
                                kwargs={"uid":self.uid,"acctime":self.acctime,"pushToken":self.pushtoken})
                            request_thread.start() # start request process
        

            # END IF FACE FOUND
        return (face_found, frame)


    def adjust_gamma(self,image, gamma=1.0):
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
                        for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)

    def apply_clahe(self,frame):
        # -----Converting image to LAB Color model-----------------------------------
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        # -----Splitting the LAB image to different channels-------------------------
        l, a, b = cv2.split(lab)
        # -----Applying CLAHE to L-channel-------------------------------------------
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        # -----Merge the CLAHE enhanced L-channel with the a and b channel-----------
        limg = cv2.merge((cl, a, b))
        # -----Converting image from LAB Color model to RGB model--------------------
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        return final

    def apply_sharpen(self,frame):
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        frame = cv2.filter2D(frame, -1, kernel)
        return frame

    # run image processing method...
    def run(self):
        self.net = cv2.dnn.readNetFromTensorflow(
            ProcessImage.DNN_MODEL_FILE, ProcessImage.DNN_MODEL_CONFIG)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        training_set_directory = os.path.join(
            current_directory, "training_set", self.uid)
        try:
            number_of_driver = len([
                name for name in os.listdir(training_set_directory) if not os.path.isfile(os.path.join(training_set_directory, name))])
        except:
            number_of_driver = 0
        while True:
            try:
                schedule.run_pending()  # keep running pending scheduler.
                if not self.TEST_MODE and not self.use_vdo:
                    if self.prev_val ==  self.current_val:
                        print("Freeze....Waiting for new frame")
                        continue
                    frame = self.img_data.copy() # copy frame 
                    #if self.prev_frame == None:
                     #   self.prev_frame = self.img_data.copy()
                else:
                    ret,frame = self.cap.read()
                    if not ret: # if video finsihed  then break the loop
                        break

                process_frame = self.apply_clahe(frame)
                process_frame = self.adjust_gamma(process_frame)
                # process_frame = self.apply_sharpen(process_frame)
                HEIGHT, WIDTH, _ = frame.shape
                # END CHECKING KNOWN FACE BLOCK
                _, process_frame = self.process_img(frame,process_frame)
                written_frame = cv2.resize(frame,VIDEO_SHAPE,interpolation=cv2.INTER_AREA) # resize frame for vdo file
                self.vdo_writer.write(written_frame) # capture frame as video
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY),75]
                _, image = cv2.imencode(".jpg", frame, encode_param)
                img_as_text = base64.b64encode(image)
                self.trip_data["ear"] = self.AVG_EAR # eye aspect ratio
                self.trip_data["mar"] = self.MAR # mouth aspect ratio
                self.trip_data["jpg_text"] = img_as_text.decode("utf-8")
                self.middle_server_socket.emit("livestream",self.trip_data)

            except Exception as err:
                print("Exception:----> ",err,"image_{}".format(self.uid))

        self.program_finish()

    # END FUNCTION "run"

if __name__ == "__main__":
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
    ap = argparse.ArgumentParser()
    ap.add_argument("-u", "--userid", required=True, help="Enter user id")
    ap.add_argument("-a", "--acctime", required=True, help="Enter trip acctime")
    ap.add_argument("-t", "--token", required=True, help="Enter session token")
    ap.add_argument("-x", "--pushtoken", required=True,
            help="Enter push notification token")
    ap.add_argument("-tid", "--tracker-id", required=False, default="60000003",
            help="Enter tracker id")
    ap.add_argument("--landmark-model", required=True, type=str,
        help="Path to landmark predictor model")
    ap.add_argument("--test",default=False,required=False, type=str2bool,
        help="Enable test mode")
    ap.add_argument("-vdo","--vdo-path",default=TEST_VDO,required=False,help="Path to test VDO file")
    ap.add_argument("--use-request",required=False,default=True,help="Request notification if event occur" , type=str2bool)
    ap.add_argument("--use-vdo",required=False,default=True,help="Request notification if event occur" , type=str2bool)
    args = vars(ap.parse_args())
    args["userid"] = args["userid"].replace(" ", "", 1)
    process_image = ProcessImage(tracker_id=args["tracker_id"], uid=args["userid"], acctime=args["acctime"], token=args["token"], pushtoken=args["pushtoken"] ,test=args["test"],test_vdo_path=args["vdo_path"],use_request=args["use_request"],use_vdo=args["use_vdo"]) # CREATE OBJECT
    process_image.load_landmark_model(args["landmark_model"]) # LOAD LANDMARK MODEL
    process_image.run() # RUN PROCESSING IMAGE PROCESS

'''
TODO LIST
- CNN facial landmark prediction model
- YOLO v3 for face detection and localization
- Train a model to take sequence of frame as input 
to predict drowsiness pattern from vdo 
'''
