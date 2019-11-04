import cv2
import requests
# cap = cv2.VideoCapture('https://admin:042830597Pt@192.168.2.3:443')
cap = cv2.VideoCapture(
    'rtsp://admin:HuaWei123@192.168.2.3/LiveMedia/ch1/Media1')
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (320, 240),
                       interpolation=cv2.INTER_AREA)
    # print("READING FRAME")
    cv2.imshow("frame", frame)

    key = cv2.waitKey(1) & 0xff
    if(key == 27):
        break
cap.release()
cv2.destroyAllWindows()
