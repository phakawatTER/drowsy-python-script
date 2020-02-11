import socketio
import json

class MappicoSocket:
    def __init__(self,TRACKER_ID,trip_data,connect=None,uid=None,acctime=None):
        self.connect = connect
        self.uid = uid
        self.acctime = acctime
        self.TRACKER_ID = TRACKER_ID
        self.sio = socketio.Client()
        self.USER_DATA = None
#        print(TRACKER_ID)

        @self.sio.event
        def connect():
            print("CONNECTED")
            pass

        @self.sio.on("obd_updated")
        def trip_updated(data):
            if data["id"] == self.TRACKER_ID:
                trip_data.update(data)

        @self.sio.on("obd_updated_event")
        def event_updated(data):
            print("EVENT DATA",data)
            eventname = data["eventname"]
            lat = data["lat"]
            lon = data["lon"]
            latlng = [lat,lon]
            direction = data["direction"]
            speed = data["speed"]
            if connect != None :
               self.connect.pushnotification(eventname,latlng,direction,speed,uid=self.uid,acctime=self.acctime)
        @self.sio.event
        def disconnect():
            print("DISCONNECTED FROM MAPPICO SOCKET...")

        self.sio.connect("https://iwapp.mappico.co.th")
        self.sio.emit("room","MAPPICO")
        self.sio.wait()
    def tripdata(self):
        return self.USER_DATA


if __name__ == "__main__":
    sock = MappicoSocket(60000003,dict())
