import socketio
import cv2
from imutils import face_utils
import base64
import numpy as np
import dlib
from scipy.spatial import distance as dist
import math
import connect
import time
import connect as conn
import argparse
import threading
import sys
import multiprocessing as mp
import mappicosocket as ms
import connect as conn
import schedule
import ddestimator as dde
import base64
from threading import Thread
import train_face_recognizer
import math
import os
import pickle
from camera_api import ENDPOINT
import json
current_directory = os.path.dirname(__file__)
print(current_directory)
# 60000003

ap = argparse.ArgumentParser()
ap.add_argument("-u", "--userid", required=True, help="Enter user id")
ap.add_argument("-a", "--acctime", required=True, help="Enter trip acctime")
ap.add_argument("-t", "--token", required=True, help="Enter session token")
ap.add_argument("-x", "--pushtoken", required=True,
                help="Enter push notification token")
ap.add_argument("-tid", "--tracker-id", required=False, default="60000003",
                help="Enter tracker id")
ap.add_argument("--hog", required=False, default=False, type=bool,
                help="Enter tracker id")
ap.add_argument("--dnn", required=False, default=True, type=bool,
                help="Enter tracker id")
args = vars(ap.parse_args())
args["userid"] = args["userid"].replace(" ", "", 1)
print(args)
# print(args["acctime"])
# print(args["userid"])
# print(args["tracker_id"])
# # grab the indexes of the facial landmarks for the left and
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
(innmStart, innmEnd) = face_utils.FACIAL_LANDMARKS_IDXS["inner_mouth"]
EYE_AR_THRESH = 0.27
EYE_AR_CONSEC_FRAMES = 3


class ProcessImage(socketio.Client):
    # SHAPE_PREDICTOR = os.path.join(current_directory, "68_landmarks_v1.00.dat")
    SHAPE_PREDICTOR = os.path.join(
        current_directory, "shape_predictor_68_face_landmarks.dat")
    DNN_MODEL_FILE = os.path.join(
        current_directory, "opencv_face_detector_uint8.pb")
    DNN_MODEL_CONFIG = os.path.join(
        current_directory, "opencv_face_detector.pbtxt")
    CNN_FACE_MODEL = os.path.join(
        current_directory, "mmod_human_face_detector.dat")

    def __init__(self, ip="http://localhost:4000", tracker_id=args["tracker_id"], uid=args["userid"], acctime=args["acctime"], token=args["token"], pushtoken=args["pushtoken"]):
        socketio.Client.__init__(self)
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
        self.NEXT_SECOND = 0
        self.COUNTER = 0
        self.ddestimator = dde.ddestimator()
        self.api_connect = conn.Connect(
            token=token, uid=uid, acctime=acctime, expoPushToken=pushtoken)
        self.trip_data = dict()
        self.data_is_recv = False
        # create directory if it's not exist
        if not os.path.exists(os.path.join(current_directory, "trip_vdo", self.uid)):
            os.makedirs(os.path.join(current_directory, "trip_vdo", self.uid))
        self.vdo_writer = cv2.VideoWriter(os.path.join(current_directory, "trip_vdo", self.uid, "{}.avi".format(
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
        # # connect to mappico socket
        self.mpc_socket = ms.MappicoSocket(
            tracker_id, self.trip_data, connect=self.connect, uid=uid, acctime=acctime, pushToken=pushtoken)

        # schedule to check if data is recieved
        # if data is not recieved anymore so proceed to terminate the process
        schedule.every(8).seconds.do(self.checkIfAlive)
        # schedule.every(1).minutes.do(self.checkYawning)

        @self.event
        def connect():
            print("Connected to server...")

        @self.on("image_{}".format(uid))
        def get_image(data):
            # print("[INFO] IMAGE RECIEVED ...")
            self.COUNT_RECV_DATA += 1
            self.AVG_DATA_RATE = self.COUNT_RECV_DATA / \
                (time.time()-self.START_TIME)
            self.data_is_recv = True
            self.current_val += 1
            b64_img = data["jpg_text"]
            del data["jpg_text"]
            self.trip_data = data
            # print(data)
            img_buffer = base64.b64decode(b64_img)
            jpg_as_np = np.frombuffer(img_buffer, dtype=np.uint8)
            self.img_data = cv2.imdecode(jpg_as_np, flags=1)
            # self.FRAME_TIMELINE.append(self.img_data)
            # print("RECV", len(self.FRAME_TIMELINE))

        @self.event
        def disconnect():
            print("Disconnected from server...")

        self.connect(ip)

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
        # if not self.face_known:
        #   train_face_recognizer.train_face_model(self.uid)
        self.vdo_writer.release()
        os._exit(0)
        # sys.exit(0)

    def draw_face(self, shape, frame, rect=None, draw_landmarks_point=False):
        points = self.ddestimator.dlib_shape_to_points(shape)
        euler, rotation, translation = self.ddestimator.est_head_dir(
            points)
        _, _, gaze_D = self.ddestimator.est_gaze_dir(points)
        bc_2d_coords = self.ddestimator.proj_head_bounding_cube_coords(
            rotation, translation)
        gl_2d_coords = self.ddestimator.proj_gaze_line_coords(
            rotation, translation, gaze_D)
        self.ddestimator.draw_gaze_line(
            frame, gl_2d_coords, (0, 255, 0), gaze_D)
        frame = self.ddestimator.draw_bounding_cube(
            frame, bc_2d_coords, (255, 0, 0), euler)
        # Draw box on face
        shape = face_utils.shape_to_np(shape)
        # extract the left and right eye coordinates, then use the
        # coordinates to compute the eye aspect ratio for both eyes
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        mouth = shape[mStart:mEnd]
        inner_mouth = shape[innmStart:innmEnd]
        leftEAR = self.eye_aspect_ratio(leftEye)
        rightEAR = self.eye_aspect_ratio(rightEye)
        # average the eye aspect ratio together for both eyes
        self.ear = (leftEAR + rightEAR) / 2.0
        # compute the convex hull for the left and right eye, then
        # visualize each of the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        mouthHull = cv2.convexHull(mouth)
        innerMouthHull = cv2.convexHull(inner_mouth)
        cv2.drawContours(
            frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(
            frame, [rightEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(
            frame, [mouthHull], -1, (0, 255, 0), 1)
        cv2.drawContours(
            frame, [innerMouthHull], -1, (0, 255, 0), 1)
        if rect:
            left, top = rect[0]
            right, bottom = rect[1]
            cv2.rectangle(frame, (left, top),
                          (right, bottom), (0, 255, 0), 2)
        if draw_landmarks_point:
            for (x, y) in points:
                cv2.circle(frame, (x, y), 1, (0, 255, 0), 1)

    def dnn_process_img(self, frame):
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [
            104, 117, 123], False, False)
        self.net.setInput(blob)
        faces = self.net.forward()
        HEIGHT, WIDTH, _ = frame.shape
        gray_dnn = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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
                width = abs(x2-x1)
                height = abs(y2-y1)
                offset_x = 0
                offest_y = 0
                if height > width:
                    offset_x = (height - width)/2
                    offset_x = int(offset_x)
                else:
                    offest_y = (width - height)/2
                    offest_y = int(offest_y)
                # create dlib rectangle class object
                # this object will be used as input for shape prediction
                face = dlib.rectangle(
                    left=x1-offset_x, top=y1-offest_y, right=x2+offset_x, bottom=y2+offest_y)
                self.face = face
                shape = self.predictor(gray_dnn, face)
                # self.draw_face(
                #     shape, frame, rect=((x1-offset_x, y1-offest_y), (x2+offset_x, y2+offest_y)), draw_landmarks_point=True)
                self.draw_face(
                    shape, frame, draw_landmarks_point=True)

        # cv2.imshow("DNN {}".format(self.uid), frame)
        return (face_found, frame)

    def hog_process_img(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray, 0)
        face_found = False
        for index, face in enumerate(faces, start=0):
            if index > 0:
                break
            face_found = True
            self.face = face
            left, top, right, bottom = face.left(), face.top(), face.right(), face.bottom()
            shape = self.predictor(gray, face)
            self.draw_face(shape, frame, rect=(
                (left, top), (right, bottom)), draw_landmarks_point=True)
            # check to see if the eye aspect ratio is below the blink
            # threshold, and if so, increment the blink frame counter
            if self.ear < EYE_AR_THRESH:
                self.COUNTER += 1
                self.EYES_CLOSED_TIME = self.COUNTER / self.AVG_FPS
            # otherwise, the eye aspect ratio is not below the blink
            # threshold
            else:
                # reset the eye frame counter
                self.COUNTER = 0
                self.EYES_CLOSED_TIME = 0
                self.NEXT_SECOND = 0
            # ALARM EVERY 2 SECOND SINCE EYES ARE CLOSED
            if (math.floor(self.EYES_CLOSED_TIME) % 2 == 1 and self.EYES_CLOSED_TIME >= 1):
                if (self.NEXT_SECOND < self.EYES_CLOSED_TIME):
                    self.NEXT_SECOND = math.floor(
                        self.EYES_CLOSED_TIME) + 1
                    req_proc = mp.Process(target=self.api_connect.pushnotification, args=(
                        "Drowsy", self.trip_data["coor"], self.trip_data["direction"], self.trip_data["speed"]))
                    # req_proc.start()
        # cv2.imshow("HOG {}".format(self.uid), frame)
        return (face_found, frame)

    def cnn_process_img(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faceRects = self.cnnFaceDetector(gray, 0)
        for faceRect in faceRects:
            x1 = faceRect.rect.left()
            y1 = faceRect.rect.top()
            x2 = faceRect.rect.right()
            y2 = faceRect.rect.bottom()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
        return frame

    # def face_identification(self):

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

    # run image processing method...
    def run(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(ProcessImage.SHAPE_PREDICTOR)
        self.cnnFaceDetector = dlib.cnn_face_detection_model_v1(
            ProcessImage.CNN_FACE_MODEL)
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
                frame = self.img_data
                HEIGHT, WIDTH, _ = frame.shape
                # check driver face
                # if not known face we will the notification to user mobile application
                # along with new driver face image
                # if not self.face_check:
                #     print("[INFO] CHECKING DRIVER FACE...")
                # #   TODO: will do face recognition for the driver with specific uid
                #     while True:
                #         print("[INFO] FINDING FACE...")
                #         gray = cv2.cvtColor(
                #             self.img_data.copy(), cv2.COLOR_BGR2GRAY)
                #         try:
                #             faces = self.detector(gray, 0)
                #             for index, face in enumerate(faces, start=0):
                #                 if index > 0:
                #                     break
                #                 left, top, right, bottom = face.left(), face.top(), face.right(), face.bottom()
                #                 print(((left, top), (right, bottom)))
                #                 face_cropped = frame[top:bottom, left:right]
                #                 face_cropped_gray = cv2.cvtColor(
                #                     face_cropped, cv2.COLOR_BGR2GRAY)
                #                 self.face_known = self.predict_face(
                #                     face_cropped_gray)
                #                 self.face_check = True
                #             if self.face_check:
                #                 break
                #         except Exception as err:
                #             self.face_check = True
                #             break
                #         time.sleep(0.5)
                # TODO: After knowing the result of face detected in image
                # If face is not known send notification to user mobile
                # ****  THIS SECION WILL BE IMPLEMETED  ***** #

                # END CHECKING KNOWN FACE BLOCK

                # if "dnn" flag is set then process frame with hog and show
                if args["dnn"]:
                    face_found, frame_dnn = self.dnn_process_img(frame.copy())
#                    cv2.imshow("DNN", frame_dnn)
                # if "dnn" flag is set to False then process frame with hog and show
                if not args["dnn"]:
                    face_found, frame_hog = self.hog_process_img(frame.copy())
#                    cv2.imshow("HOG", frame_hog)
                # start = time.time()
                # frame_cnn = self.cnn_process_img(frame.copy())
                # cv2.imshow("CNN", frame_cnn)
                # stop = time.time()
                # print("TIME SPENT CNN: {}".format(stop-start))

                # if "face_known" flag is not set means that the face was not recognized
                # collect driver face as training set as a next step
                # if not self.face_known:
                #     total_training_img = 0
                #     try:
                #         total_training_img = len(
                #             os.listdir(os.path.join(
                #                 training_set_directory, "driver_{}".format(number_of_driver+1))))
                #     except Exception as err:
                #         os.makedirs(os.path.join(
                #             training_set_directory, "driver_{}".format(number_of_driver+1)))
                #         print(err)
                #     print(total_training_img)

                #     # if the face is not known by the system then the program will collect face image
                #     # and those images will be used as training data
                #     if face_found and total_training_img < max_training_set and not self.face_known:
                #         try:
                #             left, top, right, bottom = self.face.left(
                #             ), self.face.top(), self.face.right(), self.face.bottom()
                #             side = max(abs(left-right), abs(top-bottom))
                #             offset = int(side/2)
                #             frame_copy = frame.copy()
                #             face = frame_copy[top:bottom, left:right]
                #             cv2.imwrite(os.path.join(training_set_directory, "driver_{}".format(number_of_driver+1),
                #                                      "IMG_{}.jpg".format(str((time.time())).replace(".", ""))), face)
                #         except Exception as err:
                #             print(err)
                #             pass
                CURRENT_TIME = time.time()
                if int(CURRENT_TIME - self.START_TIME) > self.PREV_TIME:
                    try:
                        self.AVG_EAR = round(
                            sum(self.TOTAL_EAR)/len(self.TOTAL_EAR), 3)
                        self.TOTAL_EAR = []
                      #  print(int(CURRENT_TIME-self.START_TIME), self.AVG_EAR)
                    except:
                        pass

                else:
                    self.TOTAL_EAR.append(self.ear)
                # update previous time
                self.PREV_TIME = int(CURRENT_TIME-self.START_TIME)
                frame = frame_dnn
                # frame = cv2.resize(frame,(2560,1440),interpolation=cv2.INTER_AREA)
                self.vdo_writer.write(frame)
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 35]
                _, image = cv2.imencode(".jpg", frame, encode_param)
                img_as_text = base64.b64encode(image)
                self.trip_data["jpg_text"] = img_as_text.decode("utf-8")
                self.trip_data["ear"] = self.AVG_EAR
                # self.tersocket.emit("send_image", self.trip_data)
                # self.emit("live_stream", self.trip_data)
                # stdout data to nodejs server
                cv2.imshow("frame", frame)
                print(str(json.dumps(self.trip_data))+"__END__")
                key = cv2.waitKey(1) & 0xff
                if key == 27:
                    break
                self.TOTAL_FRAME += 1
                self.AVG_FPS = round(self.TOTAL_FRAME /
                                     (time.time()-self.START_TIME), 2)
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
    process_image = ProcessImage()
    process_image.run()


'''
TODO LIST
- CNN facial landmark prediction model
- YOLO v3 for face detection and localization
- Train a model to take sequence of frame as input 
to predict drowsiness pattern from vdo 

'''