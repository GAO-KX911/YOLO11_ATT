import torch
from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/11/Att_yaml/yolo11-P2-CARAFE-P2DRFG.yaml").model
model.eval()

x = torch.randn(1, 3, 640, 640)

with torch.no_grad():
    y = model(x)

print(type(y))
