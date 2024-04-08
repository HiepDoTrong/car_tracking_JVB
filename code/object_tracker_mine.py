from pyimagesearch.centroidtracker_mine import CentroidTracker
from imutils.video import VideoStream
import numpy as np
import cv2
import argparse
import imutils
import time
import torch

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="Minimum probability to filter weak detections")
args = vars(ap.parse_args())

# initialize our centroid tracker and frame dimensions
cars = CentroidTracker()
persons = CentroidTracker()
(H, W) = (None, None)

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
import matplotlib
matplotlib.use('Agg')
model.classes = [0, 2]  # person and car

# initialize the video stream and allow the camera sensor to warmup
print("[INFO] starting video stream...")
# STREAM
# vs = VideoStream(src=0).start()
# time.sleep(2.0)
# VIDEO
vs = cv2.VideoCapture('/home/dotronghiep/Documents/Datasets/Car_tracking/Data_demo_09012024/test_4.mp4')

while True:
    _, frame = vs.read()
    # frame = imutils.resize(frame, width=1980)

    if W is None or H is None:
        H, W = frame.shape[:2]

    results = model(frame)
    detections = results.pred[0].detach().cpu().numpy()
    cars_in_frame = []
    persons_in_frame = []
    


    # for i in range(0, detections.shape[2]):
    #     if detections[0, 0, i, 2] > args['confidence']:
    #         box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
    #         rects.append(box.astype("int"))

    #         (startX, startY, endX, endY) = box.astype("int")
    #         cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

    for i in range(0, detections.shape[0]):
    # print(detections[i, 5])
        if detections[i, 5] == 2:
            if detections[i, 4] > args['confidence']:
                box = detections[i, 0:4] 
                cars_in_frame.append(box.astype("int"))
                (startX, startY, endX, endY) = box.astype("int")
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
        if detections[i, 5] == 0:
            if detections[i, 4] > args['confidence']:
                box = detections[i, 0:4] 
                persons_in_frame.append(box.astype("int"))
                (startX, startY, endX, endY) = box.astype("int")
                cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)



    car_objects = cars.update(cars_in_frame)
    person_objects = persons.update(persons_in_frame)

    for (objectID, centroid) in car_objects.items():
        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

    for (objectID, centroid) in person_objects.items():
        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (255, 0, 0), -1)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

cv2.destroyAllWindows()
vs.stop()
