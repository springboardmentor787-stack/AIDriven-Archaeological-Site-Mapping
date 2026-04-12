import cv2
import torch
import numpy as np
from ultralytics import YOLO
import segmentation_models_pytorch as smp

# -------- Load YOLO model --------
yolo_model = YOLO("runs/detect/yolov8s_archaeology2/weights/best.pt")

# -------- Load DeepLab model --------
seg_model = smp.DeepLabV3Plus(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=6
)

seg_model.load_state_dict(torch.load("deeplab_model.pth", map_location="cpu"))
seg_model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
seg_model = seg_model.to(device)

# -------- Load image --------
image_path = "test_image.jpg"
image = cv2.imread(image_path)
orig = image.copy()

img = cv2.resize(image, (512,512))
img_tensor = torch.tensor(img.transpose(2,0,1)/255.0, dtype=torch.float32).unsqueeze(0).to(device)

# -------- Segmentation --------
with torch.no_grad():
    seg_pred = seg_model(img_tensor)
    seg_mask = torch.argmax(seg_pred, dim=1).squeeze().cpu().numpy()

seg_mask = cv2.resize(seg_mask.astype(np.uint8), (orig.shape[1], orig.shape[0]))

# color map for masks
colors = {
    3: (0,0,255),      # ruins
    5: (0,255,0)       # vegetation
}

overlay = orig.copy()

for cls, color in colors.items():
    overlay[seg_mask == cls] = color

combined = cv2.addWeighted(orig, 0.7, overlay, 0.3, 0)

# -------- YOLO Detection --------
results = yolo_model(image_path, conf=0.5)

for box in results[0].boxes:
    x1,y1,x2,y2 = map(int, box.xyxy[0])
    cls = int(box.cls[0])
    conf = float(box.conf[0])

    label = yolo_model.names[cls]

    cv2.rectangle(combined,(x1,y1),(x2,y2),(255,0,0),2)
    cv2.putText(combined,f"{label} {conf:.2f}",(x1,y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),2)

# -------- Show result --------
cv2.imshow("Archaeological AI Pipeline", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()