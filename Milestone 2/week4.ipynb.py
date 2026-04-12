!pip install ultralytics

# ---

from ultralytics import YOLO

# ---

!unzip "/content/Archaeological DetectionFinal.v2i.yolov8.zip"

# ---

!ls

# ---

!ls train/images | head
!ls train/labels | head

# ---

model = YOLO("yolov8n.pt")

# ---

model.train(
    data="data.yaml",
    epochs=20,
    imgsz=640,
    batch=8
)

# ---

model.predict(
    source="/content/test/images",
    save=True
)

# ---

!ls runs/detect/predict

# ---

from PIL import Image
from IPython.display import display

display(Image.open("/content/runs/detect/predict/raaw6_jpg.rf.808b2b14edad52b1630337f81d9836e5.jpg"))
display(Image.open("/content/runs/detect/predict/output_709_1_b_png.rf.9de4686e6cdefad4c21504a65dd13e23.jpg"))
display(Image.open("/content/runs/detect/predict/output_123_g_png.rf.8c433488ecb60d755d3aed6883705a3c.jpg"))
display(Image.open("/content/runs/detect/predict/output_697_g_png.rf.afa7f6e0f8e6bb06b123c75d3e93a5c9.jpg"))