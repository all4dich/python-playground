from ultralytics import YOLO
import cv2
import argparse
import csv
import json
from collections import Counter
import logging
import random
import os
import sys
from clearml import Task
from datetime import datetime

# Set default console logging level as error
logging.basicConfig(level=logging.ERROR)

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--model", default="yolo11n.pt")
arg_parser.add_argument("--source", default=0)
arg_parser.add_argument("--debug", action="store_true")
arg_parser.add_argument("--output-dir", default=os.environ['HOME'])
args = arg_parser.parse_args()
output_dir = args.output_dir
cap = cv2.VideoCapture(int(args.source))
model = YOLO(args.model)

model_name = args.model.split(".")[0]
task = Task.init(project_name="video inference", task_name=f"Capture from cam - {str(datetime.now())}")
task.set_parameter("model_variant",model_name)
task.add_tags(model_name)
def count_detected_objects(counted_objects, inference_logger=None, iteration=None):
    object_counts = Counter()
    for row in counted_objects:
        name = row['name']
        object_counts[name] += 1
    for name, count in object_counts.items():
        print(f"Name: {name}, Count: {count}")
        inference_logger.report_scalar("Count of Items", name, value=count, iteration=iteration)
    print("\n")


def draw_line_boxes(original_image, boxes_data, inference_model=None,
                    color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), thickness=2):
    try:
        # Load the image
        img = original_image.copy()
        # Iterate through each bounding box
        classes = boxes_data.cls
        i = 0
        for box in boxes_data.xywh:
            class_id = int(classes[i])
            i = i + 1
            class_name = inference_model.names[class_id]
            # Extract box coordinates
            x_center, y_center, width, height = box
            # Convert (xywh) to (x1, y1, x2, y2)
            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)

            # Draw the rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            # Write text on the top of box
            cv2.putText(img, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        return img
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
args = dict(device="mps")
# Set hyperparameter
task.connect(args)
logger = task.get_logger()
i = 0
while cap.isOpened():
    i = i + 1
    ret, frame = cap.read()
    if not ret:
        break
    results = model.predict(frame, verbose=False, device="mps")
    for result in results:
        annotated_frame = result.plot()
        annotated_box = result.boxes
        my_image = draw_line_boxes(frame, annotated_box, model)
        if i % 50 == 0:
            logger.report_image("Output", "annoated", iteration=i, image=annotated_frame)
        cv2.imshow('frame', annotated_frame)
        cv2.imshow('original', frame)
        cv2.imshow('custom', my_image)
#        if args.debug:
#            cv2.imwrite(f"{output_dir}/annotated_frame.jpg", annotated_frame)
#            cv2.imwrite(f"{output_dir}/original.jpg", frame)
#            cv2.imwrite(f"{output_dir}/original-new.jpg", my_image)

        name_counts = Counter()
        annotated_reader = csv.DictReader(result.to_csv().splitlines())
        annotated_reader_json = json.loads(result.to_json())
        count_detected_objects(annotated_reader_json, logger, i)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            sys.exit()
