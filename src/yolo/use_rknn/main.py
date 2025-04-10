from ultralytics import YOLO
import cv2
import argparse
import os
from datetime import datetime
import logging

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--model_name", default="yolov8n")
arg_parser.add_argument("--task", "-t", default="detect")
args = arg_parser.parse_args()
model_name = args.model_name
task = args.task
model_file = f"{model_name}.pt"
model_path_rknn = f"{model_name}_rknn_model"

if not os.path.exists(model_path_rknn):
    model = YOLO(model_file)
    start = datetime.now()
    model.export(format="rknn", name="rk3588")
    end = datetime.now()
    hostname = os.uname()[1]
    logging.info(f"Hostname {hostname} : Exported {model_file} to RKNN in {end - start} seconds")

rknn_model = YOLO(f"./{model_name}_rknn_model", task=f"{task}")
# results = rknn_model("https://ultralytics.com/images/bus.jpg")
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    results = rknn_model.predict(frame, verbose=False   )
    # results = model.predict(frame)
    for result in results:
        annotated_frame = result.plot()
        cv2.imshow('frame', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
