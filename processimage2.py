import socketio
import cv2
from imutils import face_utils
import base64
import numpy as np
from scipy.spatial import distance as dist
from tensorflow.keras.models import load_model
from tensorflow.keras.backend import set_session
import tensorflow as tf
import math
import connect
import time
import connect as conn
import argparse
import threading
from middlesocket import MiddleServerSocket
import sys
import multiprocessing as mp
import mappicosocket as ms
import connect as conn
import schedule
import base64
from threading import Thread
import math
import os
import pickle
import json

# Keras uses Tensorflow Backend
sess = tf.Session()
set_session(sess)

current_directory = os.path.dirname(__file__)
root_directory = "/home/phakawat/"

EYE_AR_THRESH = 0.27
EYE_AR_CONSEC_FRAMES = 3
INPUT_SHAPE = 256

class ProcessImage(socketio.Client):
    DNN_MODEL_FILE = os.path.join(
        current_directory, "opencv_face_detector_uint8.pb")
    DNN_MODEL_CONFIG = os.path.join(
        current_directory, "opencv_face_detector.pbtxt")
    TEST_VIDEO_PATH = os.path.join(
        current_directory, "test_vdo.mp4")
    def __init__(self, ip="http://localhost:4000",tracker_id="", uid="", acctime="", token="", pushtoken="" ,test=False):
        socketio.Client.__init__(self)
        self.TEST_MODE = test 
        self.face_check = False
        self.face_known = False
        self.FRAME_TIMELINE = []
        self.prev_val = 0
        self.current_val = 0
        self.uid = uid
        self.acctime = acctime
        self.token = token
        self.pushtoken = pushtoken
        self.img_data = ""
        self.ear = 0
        self.TOTAL_FRAME = 0
        self.START_TIME = time.time()
        self.PREV_TIME = 0
        self.TOTAL_EAR = []
        self.AVG_FPS = 0
        self.AVG_DATA_RATE = 0
        self.COUNT_RECV_DATA = 0
        self.FPS = 0
        self.EYES_CLOSED_TIME = 0
        self.AVG_EAR = 0.35
        self.NEXT_SECOND = 0
        self.COUNTER = 0
        if self.TEST_MODE : # if test mode is enabled
            self.cap = cv2.VideoCapture(ProcessImage.TEST_VIDEO_PATH)
        self.api_connect = conn.Connect(
            token=token, uid=uid, acctime=acctime, expoPushToken=pushtoken)
        self.trip_data = {
            "uid":uid,
            "ear":0,
            "coor":(0,0),
            "gas":{"co":0,"lpg":0,"smoke":0}
        }
        self.data_is_recv = False
        # create directory if it's not exist
        if not os.path.exists(os.path.join(root_directory, "trip_vdo", self.uid)):
            os.makedirs(os.path.join(root_directory, "trip_vdo", self.uid))
        self.vdo_writer = cv2.VideoWriter(os.path.join(root_directory, "trip_vdo", self.uid, "{}.avi".format(
            acctime)), cv2.VideoWriter_fourcc(*'DIVX'), 10, (1280, 720))
        # try to use recognizer if driver face has been learned before
        # if it fails just pass
        try:
            self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
            print(os.path.join(current_directory,
                               "training_set", uid, "trainer.yml"))
            self.face_recognizer.read(os.path.join(
                current_directory, "training_set", uid, "trainer.yml"))
        except Exception as err:
            print(err)
            pass
        # schedule to check if data is recieved
        # if data is not recieved anymore so proceed to terminate the process
        schedule.every(15).seconds.do(self.checkIfAlive)
        if not self.TEST_MODE: # update trip data to database if not test mode
            schedule.every(2).seconds.do(self.update_obd_data)

        @self.event
        def connect():
            print("Connected to server...")

        @self.on("image_{}".format(uid)) # receive image streamed from JETSON NANO BOARD
        def get_image(data):
            self.COUNT_RECV_DATA += 1
            self.AVG_DATA_RATE = self.COUNT_RECV_DATA / \
                (time.time()-self.START_TIME)
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
            tracker_id, self.trip_data, connect=self.connect, uid=uid, acctime=acctime, pushToken=pushtoken) # connect to mappico socket

    def  update_obd_data(self):
        try:
            print(self.trip_data["gas"])
            coor = self.trip_data["coor"]
            direction = self.trip_data["direction"]
            co = self.trip_data["gas"]["co"]
            speed = self.trip_data["speed"]
            data = {"acctime":self.acctime,"uid":self.uid,"latlng":coor,"speed":speed,"co":co,"direction":direction}
            print(data)
            self.emit("obd_update_data",data) # emit data to local server
        except Exception as err:
            print(err)
            pass

    def load_landmark_model(self,path):
        self.landmark_model = load_model(path)

    def predict_face_landmark(self,face,show_exec_time=False):
        start = time.time()
        me = np.array(face)/255
        x_test = np.expand_dims(me, axis=0)
        x_test = np.expand_dims(x_test, axis=3)
        y_test = self.landmark_model.predict(x_test)
        label_points = (np.squeeze(y_test))
        stop = time.time()
        exec_time = round(stop-start,4)
        if show_exec_time:
            print("Execution time spent {} (s)".format(exec_time))
        return label_points

    def checkIfAlive(self):
        # print("I'm alive", str(self.current_val), str(self.prev_val))
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
        os._exit(0)
        # sys.exit(0)


    def draw_face(self,frame,face_shape,origin,key_points, draw_index=False, draw_point=True, draw_contour=False):
        x_points = key_points[::2]
        y_points = key_points[1::2]
        f_width,f_height  = face_shape
        x1,y1 = origin
        scale_x = f_width/INPUT_SHAPE
        scale_y = f_height/INPUT_SHAPE
        PADDING = 32
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
            if draw_index:
                cv2.putText(frame, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.2, (255, 255, 255), 1, cv2.LINE_AA)
        if draw_contour:
            left_eye = np.array(coords[60:68]).reshape((-1, 1, 2)).astype(np.int32)
            right_eye = np.array(coords[68:76]).reshape(
                (-1, 1, 2)).astype(np.int32)
            outer_mouth = np.array(coords[76:88]).reshape(
                (-1, 1, 2)).astype(np.int32)
            face_region = np.array(coords[0:33]).reshape(
                (-1, 1, 2)).astype(np.int32)
            cv2.drawContours(frame, [left_eye], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [right_eye], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [outer_mouth], -1, (0, 255, 0), 1)

        return frame

    def dnn_process_img(self,frame,process_frame):
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
                grey_face = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY) 
                key_points = self.predict_face_landmark(grey_face,show_exec_time=True)
                frame = self.draw_face(frame,(int(x2-x1),int(y2-y1)),(x1,y1),key_points)
        return (face_found, frame)


    def predict_face(self, frame_gray):
        print("[INFO] PREDICTING DRIVER FACE ...")
        _id, conf = self.face_recognizer.predict(frame_gray)
        print("face id:{} recognition confidence is {}% ".format(_id, conf % 100))
        if conf >= 40 and conf <= 100:
            return True
        return False

    # a function to be called when program finished and if detect new face
    def train_new_face(self):
        pass

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
        resize_scale = 0.7
        max_training_set = 60
        while True:
            try:
                schedule.run_pending()  # keep running pending scheduler.
                if not self.TEST_MODE:
                    frame = self.img_data
                else:
                    ret,frame = self.cap.read()
                    frame = cv2.rotate(frame,cv2.ROTATE_180)
                    if not ret: # if video finsihed  then break the loop
                        break
                process_frame = self.apply_clahe(frame)
                process_frame = self.adjust_gamma(process_frame)
                # process_frame = self.apply_sharpen(process_frame)
                HEIGHT, WIDTH, _ = frame.shape
                # END CHECKING KNOWN FACE BLOCK
                _, process_frame = self.dnn_process_img(frame,process_frame)
                CURRENT_TIME = time.time()
                if int(CURRENT_TIME - self.START_TIME) > self.PREV_TIME:
                    try:
                        self.AVG_EAR = round(
                            sum(self.TOTAL_EAR)/len(self.TOTAL_EAR), 3)
                        self.TOTAL_EAR = []
                    except:
                        pass
                else:
                    self.TOTAL_EAR.append(self.ear)
                # update previous time
                self.PREV_TIME = int(CURRENT_TIME-self.START_TIME)
                self.vdo_writer.write(frame) # capture frame as video
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 35]
                _, image = cv2.imencode(".jpg", frame, encode_param)
                img_as_text = base64.b64encode(image)
                self.trip_data["ear"] = self.AVG_EAR
                self.trip_data["jpg_text"] = img_as_text.decode("utf-8")
                self.middle_server_socket.emit("livestream",self.trip_data)
                key = cv2.waitKey(1) & 0xff
                if key == 27:
                    break
                self.TOTAL_FRAME += 1
                self.AVG_FPS = round(self.TOTAL_FRAME /
                                     (time.time()-self.START_TIME), 2)
 #               print([key for key in self.trip_data])
            except Exception as err:
                print(err)
                time.sleep(0.5)
                pass

        self.program_finish()

    # END FUNCTION "run"

    def eye_aspect_ratio(self, eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear


if __name__ == "__main__":
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
    ap.add_argument("--test",default=False,required=False, type=bool,
        help="Enable test mode")
    args = vars(ap.parse_args())
    print(args)
    args["userid"] = args["userid"].replace(" ", "", 1)
    process_image = ProcessImage(tracker_id=args["tracker_id"], uid=args["userid"], acctime=args["acctime"], token=args["token"], pushtoken=args["pushtoken"] ,test=args["test"] ) # CREATE OBJECT
    process_image.load_landmark_model(args["landmark_model"]) # LOAD LANDMARK MODEL
    process_image.run() # RUN PROCESSING IMAGE PROCESS

'''
TODO LIST
- CNN facial landmark prediction model
- YOLO v3 for face detection and localization
- Train a model to take sequence of frame as input 
to predict drowsiness pattern from vdo 
'''
