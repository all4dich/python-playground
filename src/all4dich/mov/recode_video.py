import cv2
import argparse
import logging
import time
argparser = argparse.ArgumentParser()
argparser.add_argument("--video_path", type=str, default="0")
args  = argparser.parse_args()

# Get a stream from Camera and write all frames to a video file using OpenCV. Resulting video's FPS has to be same with the input video.
cap = cv2.VideoCapture(int(args.video_path))
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
#out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 15, (frame_width, frame_height))
while cap.isOpened():
    start = time.time()
    time.sleep(0.01)
    success, frame = cap.read()
    if not success:
        break
    else:
        cv2.imshow('frame', frame)
        end = time.time()
        print("FPS: ", 1/(end-start))
        print(end)
        #out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break