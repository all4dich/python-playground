import cv2
import time
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("yolo11n.pt")

# Open the video file
# Get argument from command line
video_path = "/Users/sunjoo/ori.mp4"
cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    current_time_millis = int(round(time.clock_gettime(time.CLOCK_REALTIME) * 1000))
    if success:
        # Run YOLO inference on the frame
        results = model(frame,verbose=False)
        # Extract inference speed metrics
        speed = results[0].speed
        time_inference = speed["inference"]
        time_postprocess = speed["postprocess"]
        time_preprocess = speed["preprocess"]

        # Print the speed metrics
        print(f"current: {current_time_millis}, inference: {time_inference} ms, postprocess: {time_postprocess} ms, preprocess: {time_preprocess} ms")

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
