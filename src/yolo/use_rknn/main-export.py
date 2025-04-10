from ultralytics import YOLO
import argparse
from datetime import datetime
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--model_name", default="yolov8n.pt")
args = arg_parser.parse_args()
model_name = args.model_name
model = YOLO(model_name)
start = datetime.now()
model.export(format="rknn", name="rk3588")
end = datetime.now()
hostname = os.uname()[1]
logging.info(f"Hostname {hostname} : Exported {model_name} to RKNN in {end - start} seconds")

#rknn_model = YOLO("./yolov8n_rknn_model", task="detect")
#results = rknn_model("https://ultralytics.com/images/bus.jpg")
