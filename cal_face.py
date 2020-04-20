import pandas as pd
import numpy as np
import time
from scipy.spatial.distance import euclidean

class cal_face():
    eyes_close_threshold = 0.25
    mouth_open_threshold = 0.28
    yawning_time_threshold = 2.5
    dangerous_eye_close_time = 2.5
    def __init__(self):
        self.start_time = time.time() ## start time
        self.ear_log = pd.DataFrame({"timestamp":[],"ear":[]}) # log for for eyes aspect ratio over time
        self.mar_log = pd.DataFrame({"timestamp":[],"mar":[]}) # log for for mouth aspect ratio over time
        self.gaze_dir_log = pd.DataFrame({"timestamp":[],"gaze_dir":[]}) # log for gaze direction over time

    def cal_ear_98(self,points):
        """
        Calcualte EAR for left and right eyes
        and then get the average out of them...
        """
        # left eyes
        A = euclidean(points[61],points[67])
        B = euclidean(points[63],points[65])
        C = euclidean(points[60],points[64])
        EAR_LEFT = (A+B)/(2*C)
        # right eyes
        A = euclidean(points[69],points[75])
        B = euclidean(points[71],points[73])
        C = euclidean(points[68],points[72])
        EAR_RIGHT = (A+B)/(2*C)
        AVG_EAR = (EAR_LEFT + EAR_RIGHT)/2 
        df = pd.DataFrame({"timestamp":[int(time.time()-self.start_time)],"ear":[AVG_EAR]})
        self.ear_log = self.ear_log.append(df,ignore_index=True)
        return (AVG_EAR,EAR_LEFT,EAR_RIGHT)

    def cal_mar_98(self,points):
        A =  euclidean(points[89],points[95])
        B =  euclidean(points[91],points[93])
        C =  euclidean(points[88],points[82])
        MAR = (A+B)/(2*C)
        df = pd.DataFrame({"timestamp":[int(time.time()-self.start_time)],"mar":[MAR]})
        self.mar_log = self.mar_log.append(df,ignore_index=True)
        return MAR
    
    # store direction
    def store_gaze_dir(self,direction):
        df = pd.DataFrame({"timestamp":[int(time.time()-self.start_time)],"gaze_dir":[direction]})
        self.gaze_dir_log = self.gaze_dir_log.append(df,ignore_index=True)

    def get_log(self,log,period=3): # get log file of specific log base on given period
        stop_time = log.values[log.values.shape[0]-1][0] - period # latest timestamp  - period
        result = log[log.timestamp >=  stop_time]
        return result

    def check_distraction(self,duration=2.5):
        stop_time = self.gaze_dir_log.values[self.gaze_dir_log.values.shape[0]-1][0] - duration
        gaze_dir_result =  self.gaze_dir_log[self.gaze_dir_log.timestamp >= stop_time]
        mean_gd = np.mean(gaze_dir,axis=0)[1] # Calculate average gaze direction during the duration
        return mean_gd

    def check_yawn(self,duration=2.5):
        # check prev log for mar and ear due to given duration
        # ear
        stop_time = self.ear_log.values[self.ear_log.values.shape[0]-1][0] - duration
        ear_result = self.ear_log[self.ear_log.timestamp >= stop_time]
        mean_ear =  np.mean(ear_result,axis=0)[1]
        eye_close = mean_ear <= 0.30
        # mar
        stop_time = self.mar_log.values[self.mar_log.values.shape[0]-1][0] - duration
        mar_result = self.mar_log[self.mar_log.timestamp >= stop_time]
        mean_mar = np.mean(mar_result,axis=0)[1]
        mouth_is_open = mean_mar >= cal_face.mouth_open_threshold
        yawn = False
        if eye_close and mouth_is_open:
#        if mouth_is_open: 
           yawn = True
        return (yawn,mean_ear,mean_mar)

    def check_eye(self,duration=1.0):
        stop_time = self.ear_log.values[self.ear_log.values.shape[0]-1][0] - duration
        ear_result = self.ear_log[self.ear_log.timestamp >= stop_time]
        mean_ear =  np.mean(ear_result,axis=0)[1]
        eye_close = mean_ear < cal_face.eyes_close_threshold
        return (eye_close,mean_ear)



if __name__ == "__main__":
    fe = cal_face()
    new_data = pd.DataFrame({"timestamp":[time.time()],"ear":[1.123]})
