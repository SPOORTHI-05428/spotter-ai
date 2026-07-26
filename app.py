from flask import Flask, request, jsonify, render_template, send_from_directory, url_for
from ultralytics import YOLO
import os
import cv2
import uuid
from pathlib import Path
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "bmp", "mp4", "avi", "mov", "mkv", "webm"}
MODEL_PATH = "runs/detect/sanity_coco128/weights/best.pt"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load YOLO model
model = YOLO(MODEL_PATH)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULT_FOLDER"] = RESULT_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200MB max


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def is_video(filename: str) -> bool:
    ext = filename.rsplit(".", 1)[1].lower() if "." in filename else ""
    return ext in {"mp4", "avi", "mov", "mkv", "webm"}


EXAMPLES_FOLDER = "static/examples"

@app.route("/")
def index():
    examples = []
    if os.path.isdir(EXAMPLES_FOLDER):
        for f in sorted(os.listdir(EXAMPLES_FOLDER)):
            if allowed_file(f):
                name = f.rsplit(".", 1)[0].replace("_", " ").replace("-", " ").title()
                examples.append({"path": f"/{EXAMPLES_FOLDER}/{f}", "name": name})
    return render_template("index.html", examples=examples)


@app.route("/predict-example", methods=["POST"])
def predict_example():
    data = request.get_json()
    example_path = data.get("path", "")
    confidence = float(data.get("confidence", 0.25))

    # Strip leading slash and resolve
    rel_path = example_path.lstrip("/")
    abs_path = os.path.join(os.getcwd(), rel_path)

    if not os.path.isfile(abs_path):
        return jsonify({"error": "Example file not found"}), 404

    uid = uuid.uuid4().hex
    ext = abs_path.rsplit(".", 1)[1].lower()
    ext_map = {"mp4", "avi", "mov", "mkv", "webm"}

    if ext in ext_map:
        return process_video(abs_path, uid, confidence)
    else:
        return process_image(abs_path, uid, confidence)


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    # Save uploaded file with unique name
    original_name = secure_filename(file.filename)
    ext = original_name.rsplit(".", 1)[1].lower()
    uid = uuid.uuid4().hex
    upload_filename = f"{uid}_{original_name}"
    upload_path = os.path.join(app.config["UPLOAD_FOLDER"], upload_filename)
    file.save(upload_path)

    confidence = float(request.form.get("confidence", 0.25))

    if is_video(upload_filename):
        return process_video(upload_path, uid, confidence)
    else:
        return process_image(upload_path, uid, confidence)


def process_image(image_path: str, uid: str, confidence: float):
    """Run YOLO on a single image and return the annotated result."""
    results = model.predict(source=image_path, conf=confidence, save=False, verbose=False)
    result = results[0]

    # Annotate the image
    annotated_frame = result.plot()
    output_filename = f"result_{uid}.jpg"
    output_path = os.path.join(app.config["RESULT_FOLDER"], output_filename)
    cv2.imwrite(output_path, annotated_frame)

    # Extract detection data
    detections = []
    if result.boxes is not None:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            cls_name = result.names[cls_id]
            conf_val = float(box.conf[0])
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            detections.append({
                "class": cls_name,
                "confidence": round(conf_val, 3),
                "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            })

    return jsonify({
        "type": "image",
        "original": f"/{image_path.replace(os.sep, '/')}",
        "result": f"/static/results/{output_filename}",
        "detections": detections,
    })


def process_video(video_path: str, uid: str, confidence: float):
    """Run YOLO on a video and return annotated frames."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return jsonify({"error": "Could not open video file"}), 500

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_skip = max(1, int(fps // 5))  # Process ~5fps for performance

    output_video_path = os.path.join(app.config["RESULT_FOLDER"], f"result_{uid}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    frame_idx = 0
    processed_frames = 0
    out_writer = None
    all_detections = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0:
            results = model.predict(source=frame, conf=confidence, save=False, verbose=False)
            annotated = results[0].plot()

            if out_writer is None:
                h, w = annotated.shape[:2]
                out_writer = cv2.VideoWriter(output_video_path, fourcc, fps / frame_skip, (w, h))

            out_writer.write(annotated)
            processed_frames += 1

            # Collect detections from this frame
            if results[0].boxes is not None:
                for box in results[0].boxes:
                    all_detections.append({
                        "class": results[0].names[int(box.cls[0])],
                        "confidence": round(float(box.conf[0]), 3),
                    })

        frame_idx += 1

    cap.release()
    if out_writer:
        out_writer.release()

    # Summarize unique detections
    unique_classes = {}
    for d in all_detections:
        cls_name = d["class"]
        if cls_name not in unique_classes or d["confidence"] > unique_classes[cls_name]:
            unique_classes[cls_name] = d["confidence"]

    summary = [{"class": k, "max_confidence": round(v, 3)} for k, v in unique_classes.items()]
    summary.sort(key=lambda x: x["max_confidence"], reverse=True)

    return jsonify({
        "type": "video",
        "result": f"/static/results/result_{uid}.mp4",
        "total_frames": total_frames,
        "processed_frames": processed_frames,
        "detections": summary,
    })


@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory("static", filename)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

