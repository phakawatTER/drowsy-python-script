#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 18:09:08 2019

@author: phakawat
"""
import json
import requests
from camera_api import \
    API_UPDATE_TRIPDATA,\
    API_CREATE_TRIP,\
    API_PUSH_NOTIFICATION,\
    API_UPDATE_LOCATION,\
    API_LOGIN,\
    API_UPDATE_GAS


class Connect:
    def __init__(self):
        self.token = None
        self.uid = None
        self.acctime = None
        self.expoPushToken = None
        pass

    def authenticate(self, username, password):
        payload = {"username": username,
                   "password": password, "from": "camera"}
        req = requests.post(url=API_LOGIN, data=payload ,timeout=1.5)
        response = req.json()
        # print(response)
        if (response["code"] == 200):
            self.userInfo = response["userInfo"]
            self.token = self.userInfo["token"]
            self.expoPushToken = self.userInfo["expoPushToken"]
            self.uid = self.userInfo["uid"]
            self.generateACC()
            return True
        else:
            print("Failed to connect to server")
            return False
       
    def generateACC(self):
        payload = {"uid": self.uid,
                   "pushToken": self.expoPushToken, "token": self.token}
        req = requests.post(url=API_CREATE_TRIP, data=payload)
        print(req.text)
        response = json.loads(req.text)
        self.acctime = response["acctime"]
        return print(f"YOUR ACCTIME IS {self.acctime}")

    def pushnotification(self, event, latlng, direction, speed, uid=None, acctime=None, pushToken=None):
        if uid == None:
            uid = self.uid
        if acctime == None:
            acctime = self.acctime
        if pushToken == None:
            pushToken = self.expoPushToken
        payload = {
            "event": event,
            "user_id": uid,
            "latlng": latlng,
            "acctime": acctime,
            "direction": direction,
            "speed": speed,
            "token": pushToken
        }
        try:
            req = requests.post(url=API_PUSH_NOTIFICATION, data=payload)
            print(req.text)
        except Exception as err:
            print(err)

    def updateTripData(self, co, latlng, speed, direction, url=API_UPDATE_TRIPDATA):
        payload = {"acctime": self.acctime, "latlng": latlng, "co": co,
                   "uid": self.uid, "speed": speed, "direction": direction}
        try:
            req = requests.post(url=url,
                                data=payload, timeout=1)
        except Exception as err:
            print(err)
            # pass
