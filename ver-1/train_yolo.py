from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(data="e:/Internship/Milestone 1/dataset/yolo_data/data.yaml", epochs=2, imgsz=416, batch=4, patience=0)
print("Finished training.")
