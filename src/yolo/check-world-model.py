from ultralytics import YOLOWorld
import cv2

cap = cv2.VideoCapture(0)
model = YOLOWorld("yolov8m-world.pt")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # Perform inference
    results = model.predict(frame)
    #results[0].show()
    annotated_frame = results[0].plot()
    cv2.imshow('frame', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
