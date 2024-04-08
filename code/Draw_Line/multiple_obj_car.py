from imutils.video import VideoStream
from sys import argv
import imutils
import time
import cv2

# initialize OpenCV's special multi-object tracker
car_trackers = cv2.MultiTracker_create()
human_tracker_1 = cv2.MultiTracker_create()
human_tracker_2 = cv2.MultiTracker_create()


output_name = "processed_" + argv[1]

vs = cv2.vs = cv2.VideoCapture(argv[1])

W = int(vs.get(3)) 
H = int(vs.get(4))
print(W, H)
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
writer = cv2.VideoWriter("a.avi", fourcc, 10,
                         (W, H))
while True:
    frame = vs.read()[1]
    if frame is None:
        break

    #frame = imutils.resize(frame, width=600)
    (success, boxes) = car_trackers.update(frame)
    for box in boxes:
        (x, y, w, h) = [int(v) for v in box]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.putText(frame, "Car_1", (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, 2)

    (success, boxes) = human_tracker_1.update(frame)
    for box in boxes:
        (x, y, w, h) = [int(v) for v in box]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
        cv2.putText(frame, "Human_1_Car_1", (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, 2)

    (success, boxes) = human_tracker_2.update(frame)
    for box in boxes:
        (x, y, w, h) = [int(v) for v in box]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
        cv2.putText(frame, "Human_2_Car_1", (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, 2)

    cv2.imshow("Frame", frame)
    writer.write(frame)


    key = cv2.waitKey(100) & 0xFF
    if key == ord('q'):
        break

    if key == ord("s"):
        box = cv2.selectROI("Frame", frame, fromCenter=False,
                            showCrosshair=True)

        tracker = cv2.TrackerKCF_create()
        car_trackers = cv2.MultiTracker_create()
        car_trackers.add(tracker, frame, box)
    if key == ord('h'):
        box = cv2.selectROI("Frame", frame, fromCenter=False,
                            showCrosshair=True)

        tracker = cv2.TrackerKCF_create()
        human_tracker_1 = cv2.MultiTracker_create()
        human_tracker_1.add(tracker, frame, box)

    if key == ord('k'):
        box = cv2.selectROI("Frame", frame, fromCenter=False,
                            showCrosshair=True)

        tracker = cv2.TrackerKCF_create()
        human_tracker_2 = cv2.MultiTracker_create()
        human_tracker_2.add(tracker, frame, box)
    elif key == ord('c'):
        boxes = []
        human_trackers = cv2.MultiTracker_create()
        car_trackers = cv2.MultiTracker_create()

vs.release()
writer.release()
cv2.destroyAllWindows()
