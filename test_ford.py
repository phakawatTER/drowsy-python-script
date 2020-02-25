import cv2

cap = cv2.VideoCapture(1)

while True:
    _, frame = cap.read()
    # frame = cv2.resize(frame, (720, 480), interpolation=cv2.INTER_AREA)
    height, width, _ = frame.shape
    print((width, height))
    key = cv2.waitKey(1) & 0xff
    if key == 27:
        break
