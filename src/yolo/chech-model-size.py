from ultralytics import YOLO
import argparse
import logging


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--yolo_model", type=str, default="yolov5s.pt")
args = arg_parser.parse_args()
model_name = args.yolo_model
# Load the YOLO model
logging.info(f"Loading YOLO model: {model_name}")
# Get yolov5 model's dimensions for in/out

model = YOLO(model_name)

logging.info(f"Yolo model: {args.yolo_model}")
