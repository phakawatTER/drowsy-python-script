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
import schedule
import ddestimator as dde
import tersocket
import base64

ap = argparse.ArgumentParser()
ap.add_argument("-u", "--userid", required=True, help="Enter user id")
ap.add_argument("-a", "--acctime", required=True, help="Enter trip acctime")
ap.add_argument("-t", "--token", required=True, help="Enter session token")
ap.add_argument("-x", "--pushtoken", required=True,
                help="Enter push notification token")
args = vars(ap.parse_args())
args["userid"] = args["userid"].replace(" ", "", 1)
print(args["acctime"])
print(args["userid"])
# grab the indexes of the facial landmarks for the left and
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
(innmStart, innmEnd) = face_utils.FACIAL_LANDMARKS_IDXS["inner_mouth"]
EYE_AR_THRESH = 0.27
EYE_AR_CONSEC_FRAMES = 3


class ProcessImage(socketio.Client):

    SHAPE_PREDICTOR = "C:/Users/peter/Desktop/drowsy-firebase-server/functions/drowsy-python-server/shape_predictor_68_face_landmarks.dat"

    def __init__(self, ip="http://localhost:4050", uid=args["userid"], acctime=args["acctime"], token=args["token"], pushtoken=args["pushtoken"]):
        socketio.Client.__init__(self)
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
        self.AVG_FPS = 0
        self.AVG_DATA_RATE = 0
        self.COUNT_RECV_DATA = 0
        self.FPS = 0
        self.EYES_CLOSED_TIME = 0
        self.NEXT_SECOND = 0
        self.COUNTER = 0
        self.ddestimator = dde.ddestimator()
        self.tersocket = tersocket.TerSocket()
        self.api_connect = conn.Connect(
            token=token, uid=uid, acctime=acctime, expoPushToken=pushtoken)
        self.trip_data = dict()
        self.data_is_recv = False
        # schedule to check if data is recieved
        # if data is not recieved anymore so proceed to terminate the process
        schedule.every(5).seconds.do(self.checkIfAlive)
        # schedule.every(1).minutes.do(self.checkYawning)

        @self.event
        def connect():
            print("Connected to server...")

        @self.on("image_{}".format(uid))
        def get_image(data):

            self.COUNT_RECV_DATA += 1
            self.AVG_DATA_RATE = self.COUNT_RECV_DATA / \
                (time.time()-self.START_TIME)
            self.data_is_recv = True
            self.current_val += 1
            b64_img = data["jpg_text"]
            del data["jpg_text"]
            self.trip_data = data
            img_buffer = base64.b64decode(b64_img)
            jpg_as_np = np.frombuffer(img_buffer, dtype=np.uint8)
            self.img_data = cv2.imdecode(jpg_as_np, flags=1)
            self.FRAME_TIMELINE.append(self.img_data)
            print("RECV", len(self.FRAME_TIMELINE))

        @self.event
        def disconnect():
            print("Disconnected from server...")

        self.connect(ip)

    def checkIfAlive(self):
        print("I'm alive", str(self.current_val), str(self.prev_val))
        if self.data_is_recv:
            if self.prev_val != self.current_val:
                self.prev_val = self.current_val
            else:
                print("Program finished...")
                self.disconnect()
                cv2.destroyAllWindows()
                sys.exit(0)

    def run(self):
        WIDTH = 720
        HEIGHT = 480
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(ProcessImage.SHAPE_PREDICTOR)
        modelFile = "C:/Users/peter/Desktop/drowsy-firebase-server/functions/drowsy-python-server/opencv_face_detector_uint8.pb"
        configFile = "C:/Users/peter/Desktop/drowsy-firebase-server/functions/drowsy-python-server/opencv_face_detector.pbtxt"
        wait_time = 0.06
        # time.sleep(1)
        while True:
            try:
                schedule.run_pending()
                start = time.time()
                # frame = self.FRAME_TIMELINE.pop(0)
                frame = self.img_data
                self.FRAME_TIMELINE.pop(0)
                # frame = self.img_data
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = detector(gray, 0)
                for index, face in enumerate(faces, start=0):
                    if index > 0:
                        break
                    shape = predictor(gray, face)
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
                    cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                    cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
                    cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
                    cv2.drawContours(
                        frame, [innerMouthHull], -1, (0, 255, 0), 1)
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
                            print(self.trip_data)
                result, image = cv2.imencode(".jpg", frame)
                img_as_text = base64.b64encode(image)
                self.trip_data["jpg_text"] = img_as_text
                self.trip_data["ear"] = self.ear
                self.tersocket.emit("send_image", self.trip_data)
                cv2.imshow("HOG {}".format(self.uid), frame)
                frame = cv2.resize(frame, (500, 360),
                                   interpolation=cv2.INTER_AREA)
                key = cv2.waitKey(1) & 0xff
                if key == 27:
                    break
                self.TOTAL_FRAME += 1
                self.AVG_FPS = round(self.TOTAL_FRAME /
                                     (time.time()-self.START_TIME), 2)
                stop = time.time()
                spent_time = stop-start
                time_per_frame = 1 / self.AVG_DATA_RATE
                # if spent_time < time_per_frame:
                #     if len(self.FRAME_TIMELINE) >= self.AVG_DATA_RATE:
                #         print("CASE A")
                #         time.sleep((time_per_frame - spent_time)*0.6)
                #     else:
                #         print("CASE B")
                #         time.sleep((time_per_frame - spent_time)*1.25)
            except Exception as err:
                pass

        self.disconnect()
        cv2.destroyAllWindows()
        sys.exit(0)

    def eye_aspect_ratio(self, eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear


if __name__ == "__main__":
    process_image = ProcessImage()
    process_image.run()
