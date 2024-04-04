from pyimagesearch.centroidtracker import CentroidTracker, Person, Car
from imutils.video import VideoStream
import numpy as np
import cv2
import argparse
import imutils
import time
import torch        
from ultralytics import YOLO

def draw_line(event, x, y, flags, param):
    global line_start, line_end, drawing, line_drawn

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        line_start = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            line_end = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        line_end = (x, y)
        line_drawn = True

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="Minimum probability to filter weak detections")
args = vars(ap.parse_args())

cars = Car()
persons = Person()
(H, W) = (None, None)

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
import matplotlib
matplotlib.use('Agg')
model.classes = [0, 2]  # person and car
screen_width, screen_height = 1920, 1080

print("[INFO] starting video stream...")
# STREAM
# vs = VideoStream(src=0).start()
# time.sleep(2.0)
# VIDEO
vs = cv2.VideoCapture('/home/dotronghiep/Documents/Datasets/Car_tracking/JVB/D_4.mp4')
# vs = cv2.VideoCapture('/home/dotronghiep/Documents/Datasets/Car_tracking/JVB/1_1.mp4')


# Đọc frame đầu tiên từ video
ret, first_frame = vs.read()

# Tạo cửa sổ để vẽ đường kẻ trên frame đầu tiên
cv2.namedWindow('drawing')
cv2.setMouseCallback('drawing', draw_line)

# Khởi tạo biến
line_start = None
line_end = None
drawing = False
line_drawn = False
line_thickness = 2

# Hiển thị frame đầu tiên của video và chờ người dùng vẽ đường
while not line_drawn:
    frame_copy = first_frame.copy()

    if line_start and line_end:
        cv2.line(frame_copy, line_start, line_end, (0, 0, 255), line_thickness)

    cv2.imshow('drawing', frame_copy)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Đặt con trỏ đọc video lại ở frame đầu tiên
vs.set(cv2.CAP_PROP_POS_FRAMES, 0)
cv2.destroyWindow('drawing')
while True:
    ret, frame = vs.read()
    # frame = imutils.resize(frame, width=1980)

    results = model(frame)
    # for yolov5
    detections = results.pred[0].detach().cpu().numpy()
    cars_in_frame = []
    persons_in_frame = []
    
    for i in range(0, detections.shape[0]):
        if detections[i, 5] == 2:
            if detections[i, 4] > args['confidence']:
                box = detections[i, 0:4] 
                rect = box.astype("int")
                cars_in_frame.append(rect)
                # (startX, startY, endX, endY) = box.astype("int")
                # cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
        if detections[i, 5] == 0:
            if detections[i, 4] > args['confidence']:
                box = detections[i, 0:4] 
                rect = box.astype("int")
                persons_in_frame.append(rect)
                # (startX, startY, endX, endY) = box.astype("int")
                # cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)

    car_objects, overlines = cars.update(cars_in_frame, line_start, line_end)
    person_objects = persons.update(persons_in_frame, car_objects, overlines)

    for ((objectID, centroid_rect), (_, overline))  in zip(car_objects.items(), overlines.items()):
        if overline==1:
            centroid, rect = centroid_rect
            (startX, startY, endX, endY) = rect
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            text = "Car {}".format(objectID) 
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

    for (objectID, centroid_rect_carID) in person_objects.items():
        centroid, rect, carID = centroid_rect_carID
        (startX, startY, endX, endY) = rect
        cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)
        text = "Per {} Car {}".format(objectID, carID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (255, 0, 0), -1)


    cv2.line(frame, line_start, line_end, (0, 0, 255), line_thickness)

    aspect_ratio = frame.shape[1] / frame.shape[0]  # Tỷ lệ khung hình ban đầu
    new_height = screen_height
    new_width = int(new_height * aspect_ratio)

    if new_width > screen_width:
        new_width = screen_width
        new_height = int(new_width / aspect_ratio)

    resized_frame = cv2.resize(frame, (new_width, new_height))
    
    cv2.imshow("Frame", resized_frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

print(line_start, line_end)

cv2.destroyAllWindows()
vs.stop()
