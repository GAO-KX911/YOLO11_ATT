from ultralytics import YOLO

model = YOLO("runs/detect/train2/weights/best.pt")
layers, params, grads, flops = model.info()

print("layers:", layers)
print("params:", params)
print("params(M):", params / 1e6)
print("grads:", grads)
print("GFLOPs:", flops)