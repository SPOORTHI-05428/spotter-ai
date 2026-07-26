
import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLOv8 model with Ultralytics")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset YAML")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Model name or path to .pt/.yaml")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for training")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default="", help="CUDA device (e.g., '0') or 'cpu'")
    parser.add_argument("--project", type=str, default="runs/detect", help="Project directory")
    parser.add_argument("--name", type=str, default="train", help="Run name")
    parser.add_argument("--resume", action="store_true", help="Resume training from last checkpoint")
    parser.add_argument("--workers", type=int, default=8, help="Number of dataloader workers")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    project_dir = Path(args.project)
    project_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        resume=args.resume,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()


