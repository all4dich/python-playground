from ultralytics import YOLO, solutions
import cv2
import argparse
import csv
import json
from collections import Counter

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--model", default="yolo11n.pt")
args = arg_parser.parse_args()
cap = cv2.VideoCapture(0)
model = YOLO(args.model)
#model.track(source="/Users/sunjoo/Downloads/Airshow.mp4")


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    results = model.predict(frame)
    for result in results:
        annotated_frame = result.plot()
        cv2.imshow('frame', annotated_frame)
        name_counts = Counter()
        annotated_reader =  csv.DictReader(result.to_csv().splitlines())
        annotated_reader_json = json.loads(result.to_json())
        for row in annotated_reader_json:
            name = row['name']
            name_counts[name] += 1
        for name, count in name_counts.items():
            print(f"Name: {name}, Count: {count}")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            import sys
            sys.exit()