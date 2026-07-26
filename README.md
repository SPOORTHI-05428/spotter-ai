# YOLOv8 Project Scaffold

Setup:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Dataset layout:
```
data/
  images/{train,val}
  labels/{train,val}
  dataset.yaml
```

`data/dataset.yaml` example:
```yaml
train: data/images/train
val: data/images/val
nc: 1
names: ["object"]
```

Train:
```powershell
python .\scripts\train.py --data data\dataset.yaml --model yolov8n.pt --epochs 100 --imgsz 640
```

Predict:
```powershell
python .\scripts\predict.py --weights runs\detect\train\weights\best.pt --source path\to\media --save
```

Export:
```powershell
python .\scripts\export.py --weights runs\detect\train\weights\best.pt --format onnx
```

Source: [`https://youtu.be/fu2tfOV9vbY?si=JVMhUzd2w6l-tM71`](https://youtu.be/fu2tfOV9vbY?si=JVMhUzd2w6l-tM71)
