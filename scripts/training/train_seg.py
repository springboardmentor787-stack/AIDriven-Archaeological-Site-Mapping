from ultralytics import YOLO
import os


def resolve_yolo_base_weights():
    candidates = [
        "models/yolov8s.pt",
        "yolov8s.pt",
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    raise FileNotFoundError("Could not find yolov8s base weights in models/ or repository root.")


def resolve_dataset_config():
    candidates = [
        "config/data.yaml",
        "data.yaml",
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    raise FileNotFoundError("Could not find data.yaml in config/ or repository root.")

def main():
    model = YOLO(resolve_yolo_base_weights())

    model.train(
        data=resolve_dataset_config(),
        epochs=80,
        imgsz=640,
        batch=8,
        device=0,
        workers=8,
        name="yolov8s_archaeology"
    )

if __name__ == "__main__":
    main()