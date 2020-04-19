import time
import socketio
import json
import cv2
import base64
import re
from camera_api import SOCKET_ENDPOINT


class ServerSokcet(socketio.Client):
    def __init__(self, uid=None):
        socketio.Client.__init__(self)
        self.uid = uid
        self.sent_data = ''
        @self.event
        def connect():
            print("ter's server connected...")
            pass
	
#        @self.on("process_data_{}".format(uid))
#        def on_data_recv(data):
#            print("PROCESS DATA RECIEVED...")
#            print(data)

        @self.on("image_{}".format(self.uid))
        def recv_image(data):
            self.sent_data = data

        @self.event
        def disconnect():
            print("disconnected from ter's server...")

        self.connect(SOCKET_ENDPOINT)

    def sendImage(self, **kwargs):
        sent_data = {"uid": self.uid}
        for key, value in kwargs.items():
            sent_data[key] = value
        self.emit("send_image", sent_data)


if __name__ == "__main__":
    uid =  "-LrdlygY0H5IzMvDo-bh"
    socket = ServerSokcet(uid=uid)
#    @socket.on("process_data_{}".format(uid))
#    def on_data_recv(data):
#        print(data)
    #socket.sendImage(jpg_text="test",coor="test",eiei="989")
