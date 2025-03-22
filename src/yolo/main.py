import cv2
import time
import csv
from ultralytics import YOLO
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
import argparse
import logging


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--video_path", type=str, default="0")
arg_parser.add_argument("--video_fps", type=int, default=15)
arg_parser.add_argument("--video_output_path", type=str, default="/tmp/output.mp4")
arg_parser.add_argument("--influxdb_url", type=str, default="http://tiburon.keti.re.kr:38086")
arg_parser.add_argument("--influxdb_token", type=str, default="CPp4oJCt4rnYwHDo")
arg_parser.add_argument("--influxdb_org", type=str, default="KETI")
arg_parser.add_argument("--influxdb_bucket", type=str, default="DeviceMetrics")
arg_parser.add_argument("--influxdb_measurement", type=str, default="inference_metrics")
arg_parser.add_argument("--influxdb_update", action="store_true")
arg_parser.add_argument("--device", type=str, default="AGX_Orin")
arg_parser.add_argument("--yolo_model", type=str, default="yolo11n.pt")
arg_parser.add_argument("--verbose", type=bool, default=False)
arg_parser.add_argument("--show", action="store_true")

args = arg_parser.parse_args()
model_name = args.yolo_model
# Load the YOLO model
model = YOLO(model_name)

# Open the video file
# Get argument from command line
video_path = args.video_path
video_output_path = args.video_output_path
# Convert string to integer
try:
    # Use camera attached to the device
    video_path = int(video_path)
    # Check if the camera is available
except ValueError:
    logging.info("Video path is not an integer, assuming it is a file path")

try:
    cap = cv2.VideoCapture(video_path)
except Exception as e:
    logging.error(e)
    logging.error(f"Please check the video path: {video_path}")

# Print logs to the console for video in/out
logging.info(f"Video input path: {video_path}")
logging.info(f"Video output path: {args.video_output_path}")
logging.info(f"Yolo model: {args.yolo_model}")

# InfluxDB configuration
if args.influxdb_update:
    bucket = args.influxdb_bucket
    org = args.influxdb_org
    token = args.influxdb_tokenz
    url = args.influxdb_url
    client = InfluxDBClient(url=url, token=token, org=org)
    write_api = client.write_api(write_options=SYNCHRONOUS)

# Loop through the video frames
print("# Start # ")
i = 1
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = args.video_fps
frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
out = cv2.VideoWriter(video_output_path, fourcc, fps, frame_size)
while cap.isOpened():
#    print(i)
    i = i + 1
    # Read a frame from the video
    success, frame = cap.read()
    current_time_millis = int(round(time.clock_gettime(time.CLOCK_REALTIME) * 1000))
    if success:
        # Run YOLO inference on the frame
        # Rotate frame to the correct orientation
        #frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        #frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        results = model(frame,verbose=args.verbose)
        # Extract inference speed metrics
        speed = results[0].speed
        orig_img = results[0].orig_img
        time_inference = speed["inference"]
        time_postprocess = speed["postprocess"]
        time_preprocess = speed["preprocess"]

        # Write the speed metrics to a CSV file
        # writer.writerow([current_time_millis, time_inference, time_postprocess, time_preprocess])
        # Print the speed metrics
        # print(f"current: {current_time_millis}, inference: {time_inference} ms, postprocess: {time_postprocess} ms, preprocess: {time_preprocess} ms")

        # Write the speed metrics to InfluxDB
        if args.influxdb_update:
            logging.info("Updating InfluxDB")
            point = Point(args.influxdb_measurement) \
                .field("current_time", int(current_time_millis)) \
                .field("inference_time", float(time_inference)) \
                .field("postprocess_time", float(time_postprocess)) \
                .field("preprocess_time", float(time_preprocess)) \
                .tag("device", args.device) \
                .tag("yolo_model", model_name) \
                .time(int(current_time_millis), WritePrecision.MS)
            write_api.write(bucket=bucket, org=org, record=point)
        else:
            logging.info("InfluxDB update is disabled")
        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        if args.show:
            cv2.imshow("YOLO Inference", annotated_frame)
            cv2.imshow("Original Frame", orig_img)
        sunjoo_box = results[0].boxes
        # Save the annotated frame to a video file. All the frames will be saved in the same video file
        out.write(annotated_frame)
        #cv2.VideoWriter("/tmp/output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (frame.shape[1], frame.shape[0])).write(annotated_frame)
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
out.release()
cv2.destroyAllWindows()
print("# End # ")
