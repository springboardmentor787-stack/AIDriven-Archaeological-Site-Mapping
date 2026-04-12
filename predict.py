from ultralytics import YOLO

model = YOLO("runs/detect/yolov8s_archaeology2/weights/best.pt")

results = model.predict(
    source=r"C:\Users\HARSHIL SOMISETTY\HS\Infosys Springboard Virtual Internship 6.0\Archaeological-Site-Mapping-AI-V2\dataset\train\images\nalanda_03_jpg.rf.47406a8714608a48fd4e4e5980570938.jpg",
    conf=0.4,
    save=True
)