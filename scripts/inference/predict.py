import argparse
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLO inference on one image.")
    parser.add_argument(
        "--source",
        default="dataset/test/images",
        help="Image path or folder for inference.",
    )
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold.")
    return parser.parse_args()


def main():
    args = parse_args()
    model = YOLO("runs/detect/yolov8s_archaeology2/weights/best.pt")
    model.predict(source=args.source, conf=args.conf, save=True)


if __name__ == "__main__":
    main()