from pathlib import Path
from ultralytics import YOLO
from ultralytics import settings
from clearml import Task
from datetime import datetime
#from ultralytics.utils import SETTINGS
import os

print(settings)

#yolo_path_str = f"{os.environ['HOME']}/.local/yolo"
#yolo_path = Path(f"{yolo_path_str}")
#yolo_path.mkdir(parents=True, exist_ok=True)
#
#runs_path_str = f"{yolo_path_str}/runs"
#runs_path = Path(f"{runs_path_str}")
#runs_path.mkdir(parents=True, exist_ok=True)

#SETTINGS['weights_dir'] = f"{yolo_path_str}/weights"
#SETTINGS['datasets_dir'] = f"{yolo_path_str}/datasets"
#SETTINGS['runs_dir'] = runs_path_str

#settings.update({'weights_dir': f"{yolo_path_str}/weights",
#                 'datasets_dir': f"{yolo_path_str}/datasets",
#                 'runs_dir': runs_path_str})
model_path = "yolo11n.pt"
model = YOLO(model_path)
task = Task.init(project_name="Shoreline Park", task_name=f"yolo11n-{str(datetime.now())}", tags=["yolo11n", "test_task", model_path])
task.add_tags("coco8.yaml")
task.set_parameter("model_variant", "yolo11n")
# Train the model
args = dict(data="coco8.yaml", epochs=10, device="cpu")
task.connect(args)
results = model.train(**args)
model.eval()

