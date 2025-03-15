from ultralytics import YOLO

model = YOLO("yolo11n-pose.pt")
results = model.train(data="data.yaml", epochs=100, imgsz=640)