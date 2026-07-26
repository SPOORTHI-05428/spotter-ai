import argparse
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export YOLOv8 model to various formats")
    parser.add_argument("--weights", type=str, required=True, help="Path to trained .pt weights")
    parser.add_argument(
        "--format",
        type=str,
        default="onnx",
        choices=[
            "onnx",
            "openvino",
            "engine",
            "tflite",
            "tfjs",
            "torchscript",
            "coreml",
            "pb",
        ],
        help="Export format",
    )
    parser.add_argument("--dynamic", action="store_true", help="Enable dynamic axes where supported")
    parser.add_argument("--half", action="store_true", help="FP16 where supported")
    parser.add_argument("--int8", action="store_true", help="INT8 where supported (e.g., TensorRT)")
    parser.add_argument("--device", type=str, default="", help="CUDA device (e.g., '0') or 'cpu'")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.weights)
    model.export(
        format=args.format,
        dynamic=args.dynamic,
        half=args.half,
        int8=args.int8,
        device=args.device,
    )


if __name__ == "__main__":
    main()


