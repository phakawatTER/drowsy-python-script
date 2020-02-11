import time
import socketio
import json
import cv2
import base64
import re


class ServerSokcet(socketio.Client):
    def __init__(self, uid="897192"):
        socketio.Client.__init__(self)

        self.uid = uid
        self.sent_data = ''
        @self.event
        def connect():
            print("ter's server connected...")
            pass

        @self.on("image_{}".format(self.uid))
        def recv_image(data):
            self.sent_data = data

        @self.event
        def disconnect():
            print("disconnected from ter's server...")

        self.connect("http://68.183.178.189:4050")

    def sendImage(self, jpg_txt):
        self.emit("send_image", {
            "uid": self.uid,
            "jpg_txt": jpg_txt
        })
