from ultralytics import YOLO

def main():
    model = YOLO("yolov8s.pt")

    model.train(
        data="data.yaml",
        epochs=80,
        imgsz=640,
        batch=8,
        device=0,
        workers=8,
        name="yolov8s_archaeology"
    )

if __name__ == "__main__":
    main()