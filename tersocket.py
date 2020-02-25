import socketio 
from camera_api import SOCKET_ENDPOINT

class TerSocket(socketio.Client):

    def __init__(self):
        socketio.Client.__init__(self)

        @self.event
        def connect():
            print("Connected")

        @self.event
        def disconnect():
            print("Disconnected")

        self.connect(SOCKET_ENDPOINT)