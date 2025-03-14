import cv2
import time
import csv
from ultralytics import YOLO
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
import argparse


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--video_path", type=str, default="/tmp/ori.mp4")
arg_parser.add_argument("--influxdb_url", type=str, default="http://tiburon.keti.re.kr:38086")
arg_parser.add_argument("--influxdb_token", type=str, default="CPp4oJCt4rnYwHDo")
arg_parser.add_argument("--influxdb_org", type=str, default="KETI")
arg_parser.add_argument("--influxdb_bucket", type=str, default="DeviceMetrics")
arg_parser.add_argument("--influxdb_measurement", type=str, default="inference_metrics")
arg_parser.add_argument("--device", type=str, default="AGX_Orin")
arg_parser.add_argument("--yolo_model", type=str, default="yolo11n.pt")
arg_parser.add_argument("--verbose", type=bool, default=False)
args = arg_parser.parse_args()
model_name = args.yolo_model
# Load the YOLO model
model = YOLO(model_name)

# Open the video file
# Get argument from command line
video_path = args.video_path
cap = cv2.VideoCapture(video_path)

# InfluxDB configuration
bucket = args.influxdb_bucket
org = args.influxdb_org
token = args.influxdb_token
url = args.influxdb_url
client = InfluxDBClient(url=url, token=token, org=org)
write_api = client.write_api(write_options=SYNCHRONOUS)

# Loop through the video frames
print("# Start # ")
i = 1
while cap.isOpened():
    print(i)
    i = i + 1
    # Read a frame from the video
    success, frame = cap.read()
    current_time_millis = int(round(time.clock_gettime(time.CLOCK_REALTIME) * 1000))
    if success:
        # Run YOLO inference on the frame
        results = model(frame,verbose=args.verbose)
        # Extract inference speed metrics
        speed = results[0].speed
        time_inference = speed["inference"]
        time_postprocess = speed["postprocess"]
        time_preprocess = speed["preprocess"]

        # Write the speed metrics to a CSV file
        # writer.writerow([current_time_millis, time_inference, time_postprocess, time_preprocess])
        # Print the speed metrics
        # print(f"current: {current_time_millis}, inference: {time_inference} ms, postprocess: {time_postprocess} ms, preprocess: {time_preprocess} ms")

        point = Point(args.influxdb_measurement) \
            .field("current_time", int(current_time_millis)) \
            .field("inference_time", float(time_inference)) \
            .field("postprocess_time", float(time_postprocess)) \
            .field("preprocess_time", float(time_preprocess)) \
            .tag("device", args.device) \
            .tag("yolo_model", model_name) \
            .time(int(current_time_millis), WritePrecision.MS)

        write_api.write(bucket=bucket, org=org, record=point)
        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLO Inference", annotated_frame)
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
print("# End # ")
