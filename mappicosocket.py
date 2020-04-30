import socketio
import json


class MappicoSocket:
    def __init__(self, TRACKER_ID, trip_data, connect=None, uid=None, acctime=None, pushToken=None):
        self.connect = connect
        self.uid = uid
        self.acctime = acctime
        self.TRACKER_ID = TRACKER_ID
        self.sio = socketio.Client()
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
                lat = data["lat"]
                lon = data["lon"]
                coor = (lat,lon)
                data["coor"] = coor
                del data["lat"]
                del data["lon"]
                trip_data.update(data)
#                print(data)


        @self.sio.on("obd_updated_event")
        def event_updated(data):
            eventname = data["eventname"]
            lat = data["lat"]
            lon = data["lon"]
            latlng = [lat, lon]
            direction = data["direction"]
            speed = data["speed"]
            if connect != None:
                return
                self.connect.pushnotification(
                    eventname, latlng, direction, speed, uid=self.uid, acctime=self.acctime)

        @self.sio.event
        def disconnect():
            print("mappico's socket disconnected...")

        self.connect_socket() # try to reconnect to server socket
        self.sio.emit("room", "MAPPICO")
        # self.sio.wait()
    def connect_socket(self):
        self.sio.connect("https://iwapp.mappico.co.th")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-id","--tracker-id",default="60000003",required=False,help="Tracker ID")
    args = vars(ap.parse_args())
    tracker_id = args["tracker_id"]
    sock = MappicoSocket(tracker_id, dict())
