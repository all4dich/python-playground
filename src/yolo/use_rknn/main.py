from ultralytics import YOLO
import cv2

model = YOLO("yolo11n.pt")
model.export(format="rknn", name="rk3588")
# Load the exported RKNN model
rknn_model = YOLO("./yolo11n_rknn_model", task="detect")
#results = rknn_model("https://ultralytics.com/images/bus.jpg")
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    results = rknn_model(frame)
    #results = model.predict(frame)
    for result in results:
        annotated_frame = result.plot()
        cv2.imshow('frame', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
