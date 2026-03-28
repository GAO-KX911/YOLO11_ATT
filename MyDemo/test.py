from ultralytics import YOLO

model = YOLO("yolo11n.pt")
result = model("ultralytics/assets/")