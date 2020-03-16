import socketio 
from camera_api import MIDDLE_SERVER_SOCKET_ENDPOINT
print(MIDDLE_SERVER_SOCKET_ENDPOINT)
class MiddleServerSocket(socketio.Client):
    def __init__(self,endpoint=MIDDLE_SERVER_SOCKET_ENDPOINT):
        socketio.Client.__init__(self)

        @self.event
        def connect():
            print("Connected")

        @self.event
        def disconnect():
            print("Disconnected")

        self.connect(endpoint)

if __name__ == "__main__":
    MiddleServerSocket()
