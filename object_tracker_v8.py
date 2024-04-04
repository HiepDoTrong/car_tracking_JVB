from pyimagesearch.centroidtracker import CentroidTracker, Person, Car
from imutils.video import VideoStream
import numpy as np
import cv2
import argparse
import imutils
import time
import torch        
from ultralytics import YOLO
import matplotlib
import torch.nn.functional as F
from functions import *
import pandas as pd
import slots as sl

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
ap.add_argument("-c", "--confidence")
args = vars(ap.parse_args())
screen_width, screen_height = 1280, 640

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('demo/G2_conf02.mp4', fourcc, 30.0, (screen_width, screen_height))

cars = Car()
persons = Person()
(H, W) = (None, None)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLO('yolov9c.pt')
model.to(device)  # pretrained YOLOv8n model

matplotlib.use('Agg')

person_confidence = 0.2
car_confidence = 0.2

print("[INFO] starting video stream...")
# STREAM
# vs = VideoStream(src=0).start()
# time.sleep(2.0)
# VIDEO
vs = cv2.VideoCapture('/home/dotronghiep/Documents/JVBCompany/Simple-object-tracking-with-OpenCV/short_video/camG/camG2.mp4')
df = pd.read_csv('/home/dotronghiep/Documents/JVBCompany/Simple-object-tracking-with-OpenCV/centroid_coordinates/cam1.csv')
df = df.drop(columns=[df.columns[0], df.columns[-1]])
columns = np.array(df.columns)
values = df.values.squeeze()
slots = []
for column, value in zip(columns, values):
    if column[0] == 'x':
        x = int(value*screen_width)
    else:
        y = int(value*screen_height)
        name = column[-2:]
        slot = sl.Slot(name, x, y)
        slots.append(slot)
# vs = cv2.VideoCapture('/home/dotronghiep/Documents/Datasets/Car_tracking/Data_demo_09012024/test_4.mp4')

# # Đọc frame đầu tiên từ video
# ret, first_frame = vs.read()
# first_frame = cv2.resize(first_frame, (1280, 640))

# # Tạo cửa sổ để vẽ đường kẻ trên frame đầu tiên
# cv2.namedWindow('drawing')
# cv2.setMouseCallback('drawing', draw_line)

# # Khởi tạo biến
# line_start = None
# line_end = None
# drawing = False
# line_drawn = False
# line_thickness = 2

# polygon = np.array([[820, 95], [1280, 370], [1225, 640], [0, 610], [80, 340]], np.int32)
# polygon of cam D
# polygon = np.array([[820, 100], [1280, 370], [1225, 640], [0, 610], [0, 210], [220, 105]], np.int32)
# polygon of cam G
polygon = np.array([[50, 440],[345, 85],[800, 85], [1280, 440], [1280, 780], [0, 640]], np.int32)


# Hiển thị frame đầu tiên của video và chờ người dùng vẽ đường
# while not line_drawn:
#     # Hiển thị frame đầu tiên
#     frame_copy = first_frame.copy()

#     if line_start and line_end:
#         cv2.line(frame_copy, line_start, line_end, (0, 0, 255), line_thickness)
#     cv2.imshow('drawing', frame_copy)
#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('q'):
#         break

# Đặt con trỏ đọc video lại ở frame đầu tiên
# vs.set(cv2.CAP_PROP_POS_FRAMES, 0)
# cv2.destroyWindow('drawing')
rest = -1
while True:
    rest += 1
    ret, frame = vs.read()
    frame = cv2.resize(frame, (1280, 640))
    if rest % 5 == 0:
        # start inference time
        # inference_st = time.time() 
        # Convert numpy array to PyTorch tensor and change dimensions
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float().to(device)
        # Resize frame to (3, 1088, 1920)
        frame_tensor = F.interpolate(frame_tensor.unsqueeze(0), size=(640, 1280))
        frame_tensor = frame_tensor / 255.0
        # frame = frame.unsqueeze(0)

        results = model(frame_tensor, classes=[0,2,5,7], conf=0.1)  # car 2 5
        # for yolov5
        detections = results[0].boxes
        # end inference time
        # inference_et = time.time()
        # print(f"Inference time: {inference_et - inference_st}")
        cars_in_frame = []
        persons_in_frame = []
        
        # start draw time
        draw_st = time.time() 
        for box in detections:
            confidence = box.conf.cpu().item()
            if int(box.cls) == 0:
                if confidence > person_confidence:
                    rect = np.array(box.xyxy.cpu(), dtype=int).squeeze()
                    persons_in_frame.append(rect)
            else:
                if confidence > car_confidence:
                    rect = np.array(box.xyxy.cpu(), dtype=int).squeeze()
                    cars_in_frame.append(rect)
        cars_in_frame = non_max_suppression(cars_in_frame, overlapThresh=0.85)
        # print(len(cars_in_frame), len(persons_in_frame))
        car_objects, overlines, car_disappeared = cars.update(cars_in_frame)
        # person_objects, person_disappeared = persons.update(persons_in_frame, car_objects, overlines)

    for ((objectID, centroid_rect), (_, overline), (_, car_dis))  in zip(car_objects.items(), overlines.items(), car_disappeared.items()):
    # for (objectID, centroid_rect)  in car_objects.items():
        # to detect the car which cross the line
        # if overline==1:
        centroid, rect = centroid_rect
        # if centroid_in_polygon(centroid, polygon) and car_dis == 0:
        if car_dis == 0:
            (startX, startY, endX, endY) = rect
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 1)
            text = "{}".format(objectID) 
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 2, (0, 0, 255), -1)

    # for (objectID, centroid_rect_carID) in person_objects.items():
    #     centroid, rect, carID = centroid_rect_carID
    #     (startX, startY, endX, endY) = rect
    #     cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)
    #     text = "Per {} Car {}".format(objectID, carID)
    #     cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    #     cv2.circle(frame, (centroid[0], centroid[1]), 4, (255, 0, 0), -1)


    # cv2.line(frame, line_start, line_end, (0, 0, 255), line_thickness)
    cv2.polylines(frame, [polygon], isClosed=True, color=(255, 255, 255), thickness=2)
    aspect_ratio = frame.shape[1] / frame.shape[0]  # Tỷ lệ khung hình ban đầu
    new_height = screen_height
    new_width = int(new_height * aspect_ratio)

    if new_width > screen_width:
        new_width = screen_width
        new_height = int(new_width / aspect_ratio)

    resized_frame = cv2.resize(frame, (new_width, new_height))
    # end draw time
    # draw_et = time.time()
    # print(f"Draw time: {draw_et - draw_st}")
    out.write(resized_frame)
    cv2.imshow("Frame", resized_frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
out.release()
# print(line_start, line_end)

cv2.destroyAllWindows()
vs.stop()
