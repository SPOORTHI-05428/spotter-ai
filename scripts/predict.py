import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with a YOLOv8 model")
    parser.add_argument("--weights", type=str, required=True, help="Path to trained .pt weights")
    parser.add_argument("--source", type=str, required=True, help="Image/dir/video file/URL/0(webcam)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--device", type=str, default="", help="CUDA device (e.g., '0') or 'cpu'")
    parser.add_argument("--save", action="store_true", help="Save visualized predictions")
    parser.add_argument("--show", action="store_true", help="Display predictions in a window")
    parser.add_argument("--project", type=str, default="runs/detect", help="Project directory")
    parser.add_argument("--name", type=str, default="predict", help="Run name")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    Path(args.project).mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)
    results = model.predict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        project=args.project,
        name=args.name,
        save=args.save,
        show=args.show,
    )
    # Force materialize generator to ensure run completes before exit
    _ = list(results)


if __name__ == "__main__":
    main()


