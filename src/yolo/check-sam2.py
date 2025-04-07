from ultralytics import SAM
import cv2
import time

model = SAM("sam2.1_s.pt")
model.info()

cap  =  cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    result = model(frame)
    result[0].show()
    break