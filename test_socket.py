import socketio

sio = socketio.Client()

@sio.event
def connect():
    print("connected")
    
@sio.event
def disconnect():
    print("disconnected")
sio.connect("https://iwapp.mappico.co.th")
sio.wait()