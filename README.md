# 🎯 SpotterAI — AI-Powered Object Detection

> **See the world through AI eyes.** SpotterAI is a full-stack, real-time object detection web application powered by **Ultralytics YOLOv8** and **Flask**. Upload images/videos, capture from your camera, or try sample images to instantly detect and identify objects with stunning visual annotations.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🖼️ **Image Detection** | Upload JPG, PNG, BMP, or GIF — SpotterAI draws bounding boxes with labels & confidence scores |
| 🎬 **Video Processing** | Upload MP4, AVI, MOV, or WebM — annotated video output with frame-by-frame detection |
| 📸 **Live Camera Capture** | Take photos directly from your webcam for instant AI analysis |
| 🎥 **Video Recording** | Record video clips from your camera and run object detection on them |
| 🔄 **Camera Switch** | Toggle between front and back cameras on mobile devices |
| 🎚️ **Adjustable Confidence** | Fine-tune the confidence threshold (0.05–0.95) to control detection sensitivity |
| ⚡ **Quick Examples** | Pre-loaded sample images to test the AI with one click |
| 📊 **Detection Dashboard** | Detailed results table with object class, confidence %, and bounding box coordinates |
| ⬇️ **Download Results** | Save annotated images and videos directly to your device |
| 🌙 **Stunning Dark UI** | Modern, responsive interface with animated particles, gradient effects, and smooth transitions |
| 💻 **Responsive Design** | Works seamlessly on desktop, tablet, and mobile devices |

---

## 🧠 Tech Stack

| Technology | Purpose |
|------------|---------|
| **Python 3.10+** | Core backend language |
| **Flask 3.x** | Web framework for routing and serving the app |
| **Ultralytics YOLOv8** | State-of-the-art object detection model (80+ COCO classes) |
| **OpenCV** | Image/video processing and annotation rendering |
| **HTML5 / CSS3** | Responsive, animated frontend with custom properties |
| **Vanilla JavaScript** | Client-side interactivity, camera API, drag-and-drop, and async requests |
| **Werkzeug** | Secure file upload handling |
| **PyYAML** | Dataset configuration parsing |

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/SPOORTHI-05428/spotter-ai.git
cd spotter-ai
```

### 2. Set Up Virtual Environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

> On **Linux/macOS**: `source .venv/bin/activate`

### 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 4. (Optional) Train a Custom Model

If you want to train on your own dataset:

```powershell
python scripts\train.py --data data\dataset.yaml --model yolov8n.pt --epochs 100 --imgsz 640
```

### 5. Run the App

```powershell
python app.py
```

Open your browser and go to **http://localhost:5000** 🎉

---

## 🖥️ Application Structure

```
spotter-ai/
├── app.py                      # Flask web application (routes, YOLO inference)
├── requirements.txt            # Python dependencies
├── yolov8n.pt                  # Pre-trained YOLOv8 nano weights
├── README.md                   # You are here 📍
├── .gitignore                  # Git ignore rules
├── data/
│   ├── dataset.yaml            # Dataset configuration
│   ├── images/{train,val}/     # Training & validation images
│   └── labels/{train,val}/     # YOLO-format label files
├── datasets/
│   └── coco128/                # COCO128 sample dataset for training
├── scripts/
│   ├── train.py                # Training script
│   ├── predict.py              # Command-line inference script
│   └── export.py               # Model export (ONNX, TFLite, TensorRT, etc.)
├── static/
│   ├── examples/               # Sample images for quick testing
│   ├── uploads/                # User-uploaded media (auto-created)
│   └── results/                # Annotated output files (auto-created)
├── templates/
│   └── index.html              # Frontend UI (HTML + CSS + JS)
└── runs/
    └── detect/                 # Training runs, predictions, and model weights
```

---

## 📡 API Endpoints

### `GET /`
Serves the main web interface with sample image gallery.

### `POST /predict`
Upload an image or video for object detection.

**Request:** `multipart/form-data`
- `file` — The image/video file
- `confidence` — Confidence threshold (0.05–0.95, default: 0.25)

**Response (Image):**
```json
{
  "type": "image",
  "original": "/static/uploads/xxx.jpg",
  "result": "/static/results/result_xxx.jpg",
  "detections": [
    { "class": "person", "confidence": 0.92, "bbox": [100, 50, 300, 400] }
  ]
}
```

**Response (Video):**
```json
{
  "type": "video",
  "result": "/static/results/result_xxx.mp4",
  "total_frames": 300,
  "processed_frames": 25,
  "detections": [
    { "class": "car", "max_confidence": 0.88 }
  ]
}
```

### `POST /predict-example`
Run detection on a pre-loaded sample image.

**Request:** `application/json`
```json
{
  "path": "/static/examples/cat.jpg",
  "confidence": 0.25
}
```

---

## 📦 Dependencies

```
ultralytics>=8.2.0
opencv-python>=4.9.0
flask>=3.0.0
werkzeug>=3.0.0
numpy>=1.24.0
PyYAML>=6.0.1
torch>=2.2.0
torchvision>=0.17.0
matplotlib>=3.8.0
tqdm>=4.66.0
```

---

## 🧪 Training on Custom Data

1. Place your images in `data/images/train` and `data/images/val`
2. Place corresponding YOLO-format label `.txt` files in `data/labels/train` and `data/labels/val`
3. Update `data/dataset.yaml` with your class names and count
4. Run:

```powershell
python scripts\train.py --data data\dataset.yaml --model yolov8n.pt --epochs 100 --imgsz 640
```

---

## 📤 Exporting a Model

Export your trained model to various formats:

```powershell
python scripts\export.py --weights runs\detect\train\weights\best.pt --format onnx
```

Supported formats: `onnx`, `openvino`, `engine` (TensorRT), `tflite`, `tfjs`, `torchscript`, `coreml`, `pb`

---

## 🧩 Use Cases

- 🛡️ **Security & Surveillance** — Real-time monitoring with object detection
- 🚗 **Autonomous Vehicles** — Pedestrian, sign, and obstacle detection
- 🏭 **Industrial Inspection** — Defect detection and quality control
- 🌿 **Wildlife Monitoring** — Track and count animal species
- 🏥 **Medical Imaging** — Assist in anomaly detection
- 📱 **Smart Retail** — Customer behavior analysis and automated checkouts

---

## 🙌 Credits

- Built with [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- Web framework by [Flask](https://flask.palletsprojects.com/)
- COCO dataset used for training reference models
- Inspired by the YOLO community and computer vision research

---

## 📄 License

This project is for educational and research purposes. YOLOv8 weights are subject to the [AGPL-3.0 License](https://github.com/ultralytics/ultralytics/blob/main/LICENSE).

---

<p align="center">Made with ❤️ by <a href="https://github.com/SPOORTHI-05428">SPOORTHI-05428</a></p>

