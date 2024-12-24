from ultralytics import YOLO

model = YOLO('yolov8m-pose.pt')

model.train(
    data='E:/PythonProjects/dataset/dataset.yaml',
    epochs=50,
    imgsz=608,
    batch=16,
    device='cpu'
)
