import cv2
import numpy as np

polygon_points = []

video_path = r'test3.mp4'
cap = cv2.VideoCapture(video_path)


def mouse_callback(event, x, y, flags, param):
    global polygon_points
    if event == cv2.EVENT_LBUTTONDOWN:
        polygon_points.append((x, y))
        print(f"Point Added: (X: {x}, Y: {y})")



while True:

    ret, frame = cap.read()
    frame = cv2.resize(frame, (1920, 1080))
    
    cv2.namedWindow('Frame')
    cv2.setMouseCallback('Frame', mouse_callback)

   
    if len(polygon_points) > 1:
        cv2.polylines(frame, [np.array(polygon_points)], isClosed=False, color=(0, 255, 0), thickness=2)

    cv2.imshow('Frame', frame)

    
    key = cv2.waitKey(0)
    if key == 27:
        break

cv2.destroyAllWindows()
cap.release()


for point in polygon_points:
    print(f"X: {point[0]}, Y: {point[1]}")
