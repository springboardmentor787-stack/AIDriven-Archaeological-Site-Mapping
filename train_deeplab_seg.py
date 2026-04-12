import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from torchmetrics.classification import MulticlassJaccardIndex

DATASET_PATH = "seg_dataset"

# -----------------------------
# Dataset Loader
# -----------------------------
class SegDataset(Dataset):

    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(img_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img_name = self.images[idx]

        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name.replace(".jpg", ".png"))

        image = cv2.imread(img_path)
        image = cv2.resize(image, (512, 512))
        image = image.transpose(2, 0, 1) / 255.0

        mask = cv2.imread(mask_path, 0)
        mask = cv2.resize(mask, (512, 512))

        return (
            torch.tensor(image, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.long),
        )


# -----------------------------
# Dice Score (manual)
# -----------------------------
def dice_score(pred, target, num_classes=6):

    dice = 0.0

    for cls in range(num_classes):

        pred_cls = (pred == cls)
        target_cls = (target == cls)

        intersection = (pred_cls & target_cls).sum().float()
        union = pred_cls.sum().float() + target_cls.sum().float()

        if union == 0:
            continue

        dice += (2 * intersection) / union

    return dice / num_classes


# -----------------------------
# Dataset
# -----------------------------
train_dataset = SegDataset(
    f"{DATASET_PATH}/train/images",
    f"{DATASET_PATH}/train/masks"
)

valid_dataset = SegDataset(
    f"{DATASET_PATH}/valid/images",
    f"{DATASET_PATH}/valid/masks"
)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=4)


# -----------------------------
# Model
# -----------------------------
model = smp.DeepLabV3Plus(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=6
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)


# -----------------------------
# Training Setup
# -----------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()

iou_metric = MulticlassJaccardIndex(num_classes=6).to(device)


# -----------------------------
# Training Loop
# -----------------------------
EPOCHS = 10

for epoch in range(EPOCHS):

    model.train()
    running_loss = 0

    for imgs, masks in train_loader:

        imgs = imgs.to(device)
        masks = masks.to(device)

        preds = model(imgs)

        loss = criterion(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} Loss:", running_loss / len(train_loader))

#SAVE MODEL 

torch.save(model.state_dict(), "deeplab_model.pth")
print("Segmentation model saved as deeplab_model.pth")


# -----------------------------
# Validation
# -----------------------------
model.eval()

iou_score = 0
dice_total = 0

with torch.no_grad():

    for imgs, masks in valid_loader:

        imgs = imgs.to(device)
        masks = masks.to(device)

        preds = model(imgs)
        preds = torch.argmax(preds, dim=1)

        iou_score += iou_metric(preds, masks)
        dice_total += dice_score(preds, masks)

print("\n✅ Segmentation Evaluation Results")
print("IoU Score:", (iou_score / len(valid_loader)).item())
print("Dice Score:", (dice_total / len(valid_loader)).item())