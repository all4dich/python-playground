import torch

# Load the YOLOv5 model
model_path = "/Users/sunjoo/workspace/python-playground/src/yolo/yolov5su.pt"
model = torch.load(model_path, weights_only=False)

# Use loaed model to detect objects in the image
image = "/tmp/1.png"
model.eval()