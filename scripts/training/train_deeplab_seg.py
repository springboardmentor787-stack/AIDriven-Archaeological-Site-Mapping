import argparse
import os
import random
from typing import Dict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


DATASET_PATH = "seg_dataset"
ARTIFACT_DIR = os.path.join("runs", "segmentation")
NUM_CLASSES = 6


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SegDataset(Dataset):
    def __init__(self, img_dir: str, mask_dir: str, image_size: int = 512, augment: bool = False):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.augment = augment
        self.images = sorted(os.listdir(img_dir))

    def __len__(self):
        return len(self.images)

    def _mask_path_from_image(self, img_name: str) -> str:
        stem, _ = os.path.splitext(img_name)
        for ext in (".png", ".jpg", ".jpeg"):
            candidate = os.path.join(self.mask_dir, stem + ext)
            if os.path.exists(candidate):
                return candidate
        return os.path.join(self.mask_dir, stem + ".png")

    def _augment(self, image: np.ndarray, mask: np.ndarray):
        if random.random() < 0.5:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)

        if random.random() < 0.2:
            image = cv2.flip(image, 0)
            mask = cv2.flip(mask, 0)

        if random.random() < 0.4:
            alpha = 0.8 + random.random() * 0.4
            beta = random.randint(-20, 20)
            image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

        if random.random() < 0.25:
            noise = np.random.normal(0, 8, image.shape).astype(np.int16)
            image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        return image, mask

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = self._mask_path_from_image(img_name)

        image = cv2.imread(img_path)
        if image is None:
            raise RuntimeError(f"Failed to read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, 0)
        if mask is None:
            raise RuntimeError(f"Failed to read mask: {mask_path}")

        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        if self.augment:
            image, mask = self._augment(image, mask)

        image = image.transpose(2, 0, 1) / 255.0

        return (
            torch.tensor(image, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.long),
        )


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, num_classes: int, smooth: float = 1e-6) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    target_one_hot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()

    dims = (0, 2, 3)
    intersection = torch.sum(probs * target_one_hot, dims)
    cardinality = torch.sum(probs + target_one_hot, dims)

    dice = (2.0 * intersection + smooth) / (cardinality + smooth)

    # Ignore absent classes in batch-level mean.
    valid = (target_one_hot.sum(dims) > 0).float()
    dice = (dice * valid).sum() / torch.clamp(valid.sum(), min=1.0)
    return 1.0 - dice


def confusion_matrix(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> torch.Tensor:
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    valid = (target_flat >= 0) & (target_flat < num_classes)
    bins = num_classes * target_flat[valid] + pred_flat[valid]
    cm = torch.bincount(bins, minlength=num_classes * num_classes)
    return cm.reshape(num_classes, num_classes)


def metrics_from_confusion(cm: torch.Tensor) -> Dict[str, np.ndarray | float]:
    cm = cm.float()
    tp = torch.diag(cm)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp

    iou_den = tp + fp + fn
    dice_den = 2 * tp + fp + fn

    iou = torch.where(iou_den > 0, tp / iou_den, torch.nan)
    dice = torch.where(dice_den > 0, 2 * tp / dice_den, torch.nan)

    macro_iou = torch.nanmean(iou).item()
    macro_dice = torch.nanmean(dice).item()

    return {
        "macro_iou": macro_iou,
        "macro_dice": macro_dice,
        "per_class_iou": iou.cpu().numpy(),
        "per_class_dice": dice.cpu().numpy(),
    }


def evaluate_model(model, loader, ce_criterion, device):
    model.eval()
    running_loss = 0.0
    cm = torch.zeros((NUM_CLASSES, NUM_CLASSES), device=device)

    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            logits = model(imgs)
            ce_loss = ce_criterion(logits, masks)
            d_loss = soft_dice_loss(logits, masks, NUM_CLASSES)
            loss = 0.6 * ce_loss + 0.4 * d_loss
            running_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            cm += confusion_matrix(preds, masks, NUM_CLASSES)

    stats = metrics_from_confusion(cm)
    return (running_loss / len(loader), stats)


def parse_args():
    parser = argparse.ArgumentParser(description="Train DeepLabV3+ segmentation model with stronger defaults.")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--image-size", type=int, default=512)
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(42)
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    train_dataset = SegDataset(
        f"{DATASET_PATH}/train/images",
        f"{DATASET_PATH}/train/masks",
        image_size=args.image_size,
        augment=True,
    )

    valid_dataset = SegDataset(
        f"{DATASET_PATH}/valid/images",
        f"{DATASET_PATH}/valid/masks",
        image_size=args.image_size,
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    model = smp.DeepLabV3Plus(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=NUM_CLASSES,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Training device: {device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=4, min_lr=1e-6
    )
    ce_criterion = torch.nn.CrossEntropyLoss()

    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    history = []
    best_iou = -1.0
    best_epoch = 0
    best_stats = None
    best_model_path = os.path.join(ARTIFACT_DIR, "deeplab_model_best.pth")
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        running_train_loss = 0.0

        for imgs, masks in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                logits = model(imgs)
                ce_loss = ce_criterion(logits, masks)
                d_loss = soft_dice_loss(logits, masks, NUM_CLASSES)
                loss = 0.6 * ce_loss + 0.4 * d_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_train_loss += loss.item()

        train_loss = running_train_loss / len(train_loader)
        val_loss, val_stats = evaluate_model(model, valid_loader, ce_criterion, device)
        val_iou = val_stats["macro_iou"]
        val_dice = val_stats["macro_dice"]
        current_lr = optimizer.param_groups[0]["lr"]

        history.append(
            {
                "epoch": epoch + 1,
                "lr": float(current_lr),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_iou": float(val_iou),
                "val_dice": float(val_dice),
            }
        )

        print(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"lr={current_lr:.6f} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_iou={val_iou:.4f} | val_dice={val_dice:.4f}"
        )

        scheduler.step(val_iou)

        if val_iou > best_iou:
            best_iou = val_iou
            best_epoch = epoch + 1
            best_stats = val_stats
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> New best model saved ({best_model_path})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch + 1} (best epoch={best_epoch}, best IoU={best_iou:.4f})")
                break

    # Keep backward compatibility with existing inference scripts.
    if os.path.exists(best_model_path):
        state = torch.load(best_model_path, map_location="cpu")
        os.makedirs("models", exist_ok=True)
        torch.save(state, os.path.join("models", "deeplab_model.pth"))
        torch.save(state, "deeplab_model.pth")
        print("Best checkpoint copied to models/deeplab_model.pth")

    metrics_df = pd.DataFrame(history)
    metrics_csv_path = os.path.join(ARTIFACT_DIR, "deeplab_metrics.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"Saved metrics CSV: {metrics_csv_path}")

    # Loss curve
    loss_curve_path = os.path.join(ARTIFACT_DIR, "deeplab_loss_curve.png")
    plt.figure(figsize=(8, 5))
    plt.plot(metrics_df["epoch"], metrics_df["train_loss"], marker="o", label="Train Loss")
    plt.plot(metrics_df["epoch"], metrics_df["val_loss"], marker="o", label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("DeepLabV3+ Loss Curves")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_curve_path, dpi=150)
    plt.close()
    print(f"Saved loss curve: {loss_curve_path}")

    # IoU/Dice curve
    curve_path = os.path.join(ARTIFACT_DIR, "deeplab_iou_dice_curve.png")
    plt.figure(figsize=(8, 5))
    plt.plot(metrics_df["epoch"], metrics_df["val_iou"], marker="o", label="Val IoU")
    plt.plot(metrics_df["epoch"], metrics_df["val_dice"], marker="o", label="Val Dice")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("DeepLabV3+ Validation IoU/Dice")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(curve_path, dpi=150)
    plt.close()
    print(f"Saved metric curve: {curve_path}")

    if best_stats is not None:
        per_class_df = pd.DataFrame(
            {
                "class_id": list(range(NUM_CLASSES)),
                "iou": best_stats["per_class_iou"],
                "dice": best_stats["per_class_dice"],
            }
        )
        per_class_path = os.path.join(ARTIFACT_DIR, "deeplab_per_class_metrics.csv")
        per_class_df.to_csv(per_class_path, index=False)
        print(f"Saved per-class metrics: {per_class_path}")

    print("\n✅ Segmentation Training/Evaluation Complete")
    print(f"Best Epoch: {best_epoch}")
    print(f"Best IoU: {best_iou:.6f}")
    if best_stats is not None:
        print(f"Best Dice: {best_stats['macro_dice']:.6f}")


if __name__ == "__main__":
    main()