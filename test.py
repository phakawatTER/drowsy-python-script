import cv2
import socket
import pickle
import struct
import requests
print("WAITING FOR CLIENT")
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(("192.168.1.44", 8009))
# cap = cv2.VideoCapture('https://admin:042830597Pt@192.168.2.3:443')
cap = cv2.VideoCapture(
    'rtsp://admin:HuaWei123@192.168.2.3/LiveMedia/ch1/Media1')
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (320, 240),
                       interpolation=cv2.INTER_AREA)
    frame = cv2.imencode(".jpg", frame)
    data = pickle.dumps(frame, 0)
    size = len(data)
    s.sendall(struct.pack(">Ldd", size, 14.03333, 103.11231423)+data)

    # print("READING FRAME")
    # cv2.imshow("frame", frame)

    key = cv2.waitKey(1) & 0xff
    if(key == 27):
        break
cap.release()
cv2.destroyAllWindows()
