import cv2

# Hàm xử lý sự kiện khi click chuột để vẽ đường kẻ
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

# Khởi tạo video capture từ file hoặc camera
video_capture = cv2.VideoCapture('/home/dotronghiep/Documents/Datasets/Car_tracking/Data_demo_09012024/test_4.mp4')

# Đọc frame đầu tiên từ video
ret, first_frame = video_capture.read()

# Tạo cửa sổ để vẽ đường kẻ trên frame đầu tiên
cv2.namedWindow('video')
cv2.setMouseCallback('video', draw_line)

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
        cv2.line(frame_copy, line_start, line_end, (0, 255, 0), line_thickness)

    cv2.imshow('video', frame_copy)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Đặt con trỏ đọc video lại ở frame đầu tiên
video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Đọc video và chạy nếu đường kẻ đã được vẽ
while True:
    ret, frame = video_capture.read()

    if not ret:
        break

    if line_drawn:
        if line_start and line_end:
            cv2.line(frame, line_start, line_end, (0, 255, 0), line_thickness)

    cv2.imshow('video', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Giải phóng bộ nhớ và đóng cửa sổ
video_capture.release()
cv2.destroyAllWindows()
