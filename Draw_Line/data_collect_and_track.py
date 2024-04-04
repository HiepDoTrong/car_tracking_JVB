# import the necessary packages

from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2


from detection_model import load_model
from detection_model import CLASSES
from roi2name import Roi2nameKNN


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str,
                help="path to optional input video file")
ap.add_argument("-o", "--output", type=str,
                help="path to optional output video file")
ap.add_argument("-c", "--confidence", type=float, default=0.4,
                help="minimum probability to filter weak detections")
ap.add_argument("-s", "--skip-frames", type=int, default=30,
                help="# of skip frames between detections")
ap.add_argument("-r", "--recognizer", type=str,
                help="path to recognition model")

args = vars(ap.parse_args())


net = load_model()
roi2name = Roi2nameKNN()
# if a video path was not supplied, grab a reference to the webcam
if not args.get("input", False):
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

# otherwise, grab a reference to the video file
else:
    # print("[INFO] opening video file...")
    vs = cv2.VideoCapture(args["input"])


trackers = []
name_ids = []
writer = None
totalFrames = 0

# for clecting train data
data = {}
i_ID = 0
only_track = False
data_collecting = False

# start the frames per second throughput estimator
fps = FPS().start()

# loop over frames from the video stream
W = None
H = None

fps = FPS().start()

while True:
    # grab the next frame and handle if we are reading from either
    # VideoCapture or VideoStream
    frame = vs.read()
    frame = frame[1] if args.get("input", False) else frame

    # if we are viewing a video and we did not grab a frame then we
    # have reached the end of the video
    if args["input"] is not None and frame is None:
        break

    # resize the frame to have a maximum width of 500 pixels (the
    # less data we have, the faster we can process it), then convert
    # the frame from BGR to RGB for dlib
    frame = imutils.resize(frame, width=640)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # if the frame dimensions are empty, set them
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    # if we are supposed to be writing a video to disk, initialize
    # the writer
    if args["output"] is not None and writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 30,
                                 (W, H), True)

    # initialize the current status along with our list of bounding
    # box rectangles returned by either (1) our object detector or
    # (2) the correlation trackers
    status = "Waiting"
    rects = []

    # check to see if we should run a more computationally expensive
    # object detection method to aid our tracker and collect data

    if ((totalFrames % args["skip_frames"] == 0) and not only_track) or (data_collecting and not only_track):
        # set the status and initialize our new set of object trackers
        status = "Detecting"
        trackers = []
        name_ids = []

        # convert the frame to a blob and pass the blob through the
        # network and obtain the detections
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
        net.setInput(blob)
        detections = net.forward()

        # loop over the detections
        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated
            # with the prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by requiring a minimum
            # confidence
            if confidence > args["confidence"]:
                # extract the index of the class label from the
                # detections list
                idx = int(detections[0, 0, i, 1])

                # if the class label is not a person, ignore it
                if CLASSES[idx] != "person":
                    continue

                # compute the (x, y)-coordinates of the bounding box
                # for the object
                box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = box.astype("int")

                # get name_id
                try:
                    roi = frame[startY-10: endY+10, startX-10: endX+10]
                    roi = cv2.resize(roi, (60, 160))
                except:
                    roi = frame[startY: endY, startX: endX ]
                    roi = cv2.resize(roi, (60, 160))

                # no data
                if len(data.keys()) == 0 and not data_collecting:
                    name_ids.append('unknown')
                else:
                    if not data_collecting:
                        roi_name = roi2name.predict([roi])[0]
                        roi_name = '{}:{:3.2f}'.format(roi_name[0], roi_name[1])
                        name_ids.append(roi_name)
                    else:
                        name_ids.append('collecting')
                        i_ID += 1
                        data[i_ID] = [roi]
                        only_track = True
                        print('tag')

                # construct a dlib rectangle object from the bounding
                # box coordinates and then start the dlib correlation
                # tracker
                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(startX, startY, endX, endY)
                tracker.start_track(rgb, rect)

                # add the tracker to our list of trackers so we can
                # utilize it during skip frames
                trackers.append(tracker)
                rects.append((startX, startY, endX, endY))

    else:
        # loop over the trackers
        for i_tracker, tracker in enumerate(trackers):
            # set the status of our system to be 'tracking' rather
            # than 'waiting' or 'detecting'
            status = "Tracking"

            # update the tracker and grab the updated position
            tracker.update(rgb)
            pos = tracker.get_position()

            # unpack the position object
            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())
            try:
                roi = frame[startY: endY, startX: endX]
                roi = cv2.resize(roi, (60, 160))

                if data_collecting:
                    data[i_ID - len(trackers) + 1] += [roi]
            except:
                pass

            # add the bounding box coordinates to the rectangles list
            rects.append((startX, startY, endX, endY))

    for ir, (startX, startY, endX, endY) in enumerate(rects):
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
        text = "{n}".format(n=name_ids[ir])
        cv2.putText(frame, text, (startX, (startY + endY) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # show the output frame

    cv2.imshow("Frame", frame)
    if writer is not None:
        writer.write(frame)

    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

    if key == ord("c"):
        data_collecting = True
        only_track = False

    if key == ord("d"):
        only_track = False
        data_collecting = False
        shape = data[1][0].shape
        print(shape)
        # create blank images as unknown
        X = [np.zeros(shape=shape) for i in range(100)]
        y = [0] * 100
        for k, val in data.items():
            X += val
            y += [k] * len(val)

        y = np.array(y)
        X = np.array(X)
        print(X.shape)
        print(y.shape)
        roi2name.trainXy(images=X, labels=y)

    totalFrames += 1
    fps.update()

fps.stop()


print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
# check to see if we need to release the video writer pointer
if writer is not None:
    writer.release()

# if we are not using a video file, stop the camera video stream
if not args.get("input", False):
    vs.stop()

# otherwise, release the video file pointer
else:
    vs.release()

# close any open windows
cv2.destroyAllWindows()
