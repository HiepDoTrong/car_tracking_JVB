import cv2
import numpy as np

# Load the video
video_path = "./short_video/camG/camG1.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width, height = 1280, 640
print(f"FPS: {fps}, Width: {width}, Height: {height}")

# Load the video
# input video
video_path = "/home/dotronghiep/Documents/JVBCompany/car_tracking_JVB/video_short/human_camD/human_camD.mp4"
cap = cv2.VideoCapture(video_path)
fourcc = cv2.VideoWriter_fourcc(*"XVID")

# output video
out_blur = cv2.VideoWriter("/home/dotronghiep/Documents/JVBCompany/car_tracking_JVB/code/human_camD_blur.mp4", fourcc, fps, (width, height))
# Add blur to each frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Apply Gaussian blur to the frame
    frame = cv2.resize(frame, (width, height))

    blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)

    # Write the blurred frame to the output video
    out_blur.write(blurred_frame)

    # Display the blurred frame
    cv2.imshow("Blurred Video", blurred_frame)
    # cv2.imshow("Blurred Video", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and writer
cap.release()
out_blur.release()

# Close all windows
cv2.destroyAllWindows()