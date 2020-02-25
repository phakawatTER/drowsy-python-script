import socketio
import json


class MappicoSocket:
    def __init__(self, TRACKER_ID, trip_data, connect=None, uid=None, acctime=None, pushToken=None):
        self.connect = connect
        self.uid = uid
        self.acctime = acctime
        self.TRACKER_ID = TRACKER_ID
        self.sio = socketio.Client()
        
        self.sio.emit("room", "MAPPICO")
        print(f"TRACKER ID :{TRACKER_ID}")
        print(f"trip_data :{trip_data}")
        print(f"uid :{uid}")
        print(f"acctime :{acctime}")
        print(f"pushToken :{pushToken}")
        @self.sio.event
        def connect():
            print("mappico's socket connected...")
            

        @self.sio.on("obd_updated")
        def trip_updated(data):
            if data["id"] == self.TRACKER_ID:
                print(data)
                trip_data.update(data)

        @self.sio.on("obd_updated_event")
        def event_updated(data):
            eventname = data["eventname"]
            lat = data["lat"]
            lon = data["lon"]
            latlng = [lat, lon]
            direction = data["direction"]
            speed = data["speed"]
            if connect != None:
                self.connect.pushnotification(
                    eventname, latlng, direction, speed, uid=self.uid, acctime=self.acctime)

        @self.sio.event
        def disconnect():
            print("mappico's socket disconnected...")

        self.sio.connect("https://iwapp.mappico.co.th")
        # self.sio.wait()

if __name__ == "__main__":
    sock = MappicoSocket("60000003", dict())
