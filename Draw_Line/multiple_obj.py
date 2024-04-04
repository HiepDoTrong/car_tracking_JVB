from imutils.video import VideoStream
from sys import argv
import imutils
import time
import cv2

# initialize OpenCV's special multi-object tracker
trackers = cv2.MultiTracker_create()

if int(argv[1]) == 0:
    vs = VideoStream(src=0).start()
    time.sleep(1.0)
else:
    vs = cv2.vs = cv2.VideoCapture(argv[2])
    #time.sleep(2)
while True:

    frame = vs.read()
    if int(argv[1]) == 0:
        frame = cv2.flip(frame, 1)
    else:
        frame = frame[1]

    if frame is None:
        break

    frame = imutils.resize(frame, width=600)
    (success, boxes) = trackers.update(frame)
    for box in boxes:
        (x, y, w, h) = [int(v) for v in box]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(100) & 0xFF
    if key == ord('q'):
        break

    if key == ord("s"):
        box = cv2.selectROI("Frame", frame, fromCenter=False,
                            showCrosshair=True)

        tracker = cv2.TrackerKCF_create()
        trackers.add(tracker, frame, box)
    elif key == ord('c'):
        boxes = []
        trackers = cv2.MultiTracker_create()

if int(argv[1]) != 0:
    vs.release()
else:
    vs.stop()

cv2.destroyAllWindows()
