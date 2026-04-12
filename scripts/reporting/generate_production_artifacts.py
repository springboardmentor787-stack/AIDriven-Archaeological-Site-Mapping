import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from ultralytics import YOLO
import segmentation_models_pytorch as smp

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


ROOT = Path(__file__).resolve().parents[2]
DATE_TAG = datetime.now().strftime("%Y%m%d")
OUT_ROOT = ROOT / "artifacts" / f"model_metrics_{DATE_TAG}"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def file_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    return round(path.stat().st_size / (1024 * 1024), 4)


def read_yaml(path: Path) -> dict:
    if not path.exists() or yaml is None:
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def safe_copy(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)
    return True


def git_commit_hash() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True)
        return out.strip()
    except Exception:
        return "unknown"


def dataset_inventory() -> dict:
    det = {}
    for split in ["train", "valid", "test"]:
        img_dir = ROOT / "dataset" / split / "images"
        lbl_dir = ROOT / "dataset" / split / "labels"
        det[split] = {
            "images": len(list(img_dir.glob("*"))) if img_dir.exists() else 0,
            "labels": len(list(lbl_dir.glob("*.txt"))) if lbl_dir.exists() else 0,
        }

    seg = {}
    for split in ["train", "valid", "test"]:
        img_dir = ROOT / "seg_dataset" / split / "images"
        msk_dir = ROOT / "seg_dataset" / split / "masks"
        seg[split] = {
            "images": len(list(img_dir.glob("*"))) if img_dir.exists() else 0,
            "masks": len(list(msk_dir.glob("*"))) if msk_dir.exists() else 0,
        }

    terrain_csv = ROOT / "terrain_model" / "erosion_dataset.csv"
    terrain_rows = 0
    if terrain_csv.exists():
        try:
            terrain_rows = int(pd.read_csv(terrain_csv).shape[0])
        except Exception:
            terrain_rows = 0

    return {
        "detection": det,
        "segmentation": seg,
        "terrain_rows": terrain_rows,
    }


def benchmark_yolo(model_path: Path, sample_image: Path, runs: int = 5) -> dict:
    if not model_path.exists() or not sample_image.exists():
        return {"latency_ms_mean": None, "latency_ms_std": None, "throughput_fps": None}

    model = YOLO(str(model_path))
    model.predict(source=str(sample_image), conf=0.25, verbose=False, device="cpu")

    timings = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model.predict(source=str(sample_image), conf=0.25, verbose=False, device="cpu")
        timings.append((time.perf_counter() - t0) * 1000.0)

    mean_ms = float(np.mean(timings))
    std_ms = float(np.std(timings))
    fps = 1000.0 / mean_ms if mean_ms > 0 else None
    return {
        "latency_ms_mean": round(mean_ms, 3),
        "latency_ms_std": round(std_ms, 3),
        "throughput_fps": round(fps, 3) if fps is not None else None,
    }


def build_yolo_training_charts(df: pd.DataFrame, out_png: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()

    axes[0].plot(df["epoch"], df["train/box_loss"], label="train/box_loss")
    axes[0].plot(df["epoch"], df["val/box_loss"], label="val/box_loss")
    axes[0].set_title("Box Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(df["epoch"], df["metrics/precision(B)"], label="precision")
    axes[1].plot(df["epoch"], df["metrics/recall(B)"], label="recall")
    axes[1].set_title("Precision / Recall")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    axes[2].plot(df["epoch"], df["metrics/mAP50(B)"], label="mAP50")
    axes[2].plot(df["epoch"], df["metrics/mAP50-95(B)"], label="mAP50-95")
    axes[2].set_title("mAP Curves")
    axes[2].legend()
    axes[2].grid(alpha=0.25)

    axes[3].plot(df["epoch"], df["train/cls_loss"], label="train/cls_loss")
    axes[3].plot(df["epoch"], df["val/cls_loss"], label="val/cls_loss")
    axes[3].set_title("Cls Loss")
    axes[3].legend()
    axes[3].grid(alpha=0.25)

    plt.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_text(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(text)


def f1_from_pr(p: float, r: float) -> float:
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def yolo_model_report(model_key: str, run_dir: Path, base_weight: Path, sample_image: Path, common_meta: dict, dataset_meta: dict):
    out_dir = ensure_dir(OUT_ROOT / model_key)
    copied = []

    args = read_yaml(run_dir / "args.yaml")
    results_csv = run_dir / "results.csv"
    if not results_csv.exists():
        return

    df = pd.read_csv(results_csv)
    best_idx = int(df["metrics/mAP50-95(B)"].idxmax())
    final_row = df.iloc[-1].to_dict()
    best_row = df.iloc[best_idx].to_dict()

    for artifact_name in [
        "results.png",
        "PR_curve.png",
        "P_curve.png",
        "R_curve.png",
        "F1_curve.png",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
    ]:
        src = run_dir / artifact_name
        dst = out_dir / "raw_artifacts" / artifact_name
        if safe_copy(src, dst):
            copied.append(str(dst.relative_to(OUT_ROOT)).replace("\\", "/"))

    weight_path = run_dir / "weights" / "best.pt"
    latency = benchmark_yolo(weight_path, sample_image)

    custom_plot = out_dir / "training_curves_custom.png"
    build_yolo_training_charts(df, custom_plot)
    copied.append(str(custom_plot.relative_to(OUT_ROOT)).replace("\\", "/"))

    epochs = int(len(df))
    total_training_sec = float(df["time"].iloc[-1]) if "time" in df.columns else None
    sec_per_epoch = (total_training_sec / epochs) if total_training_sec else None

    summary = {
        "model_name": model_key,
        "task_type": "object_detection",
        "status": "trained",
        "data_used": {
            "detection_splits": dataset_meta["detection"],
            "dataset_config": args.get("data", "config/data.yaml"),
        },
        "epochs": epochs,
        "general_training_metrics": {
            "final": {
                "precision": float(final_row["metrics/precision(B)"]),
                "recall": float(final_row["metrics/recall(B)"]),
                "f1_estimated": float(f1_from_pr(final_row["metrics/precision(B)"], final_row["metrics/recall(B)"])),
                "map50": float(final_row["metrics/mAP50(B)"]),
                "map50_95": float(final_row["metrics/mAP50-95(B)"]),
                "train_box_loss": float(final_row["train/box_loss"]),
                "train_cls_loss": float(final_row["train/cls_loss"]),
                "train_dfl_loss": float(final_row["train/dfl_loss"]),
                "val_box_loss": float(final_row["val/box_loss"]),
                "val_cls_loss": float(final_row["val/cls_loss"]),
                "val_dfl_loss": float(final_row["val/dfl_loss"]),
            },
            "best_map50_95_epoch": int(best_row["epoch"]),
            "best_map50_95": float(best_row["metrics/mAP50-95(B)"]),
            "best_map50": float(best_row["metrics/mAP50(B)"]),
            "best_precision": float(df["metrics/precision(B)"].max()),
            "best_recall": float(df["metrics/recall(B)"].max()),
        },
        "advanced_metrics": {
            "roc_auc": None,
            "confusion_matrix_artifact": "raw_artifacts/confusion_matrix.png",
            "pr_curve_artifact": "raw_artifacts/PR_curve.png",
        },
        "operational_metrics": {
            "training_time_sec": round(total_training_sec, 3) if total_training_sec else None,
            "avg_time_per_epoch_sec": round(sec_per_epoch, 3) if sec_per_epoch else None,
            "model_size_mb": file_size_mb(weight_path),
            "latency_throughput_cpu": latency,
        },
        "reproducibility": {
            "hyperparameters": args,
            "model_files": {
                "trained_weight": str(weight_path.relative_to(ROOT)).replace("\\", "/") if weight_path.exists() else None,
                "base_weight": str(base_weight.relative_to(ROOT)).replace("\\", "/") if base_weight.exists() else None,
            },
            "code_versions": common_meta,
            "configurations": {
                "requirements_file": "requirements.txt",
                "dataset_config": str((ROOT / "config" / "data.yaml").relative_to(ROOT)).replace("\\", "/"),
            },
            "logs": {
                "results_csv": str(results_csv.relative_to(ROOT)).replace("\\", "/"),
                "args_yaml": str((run_dir / "args.yaml").relative_to(ROOT)).replace("\\", "/"),
            },
            "deployment_assets": {
                "streamlit": "app.py",
                "nextjs_api": "geo-ai-ui/app/api/predict/route.ts",
                "python_worker": "geo-ai-ui/server/predict_worker.py",
            },
        },
        "artifacts": copied,
    }

    write_json(out_dir / "metrics_summary.json", summary)

    report = f"""# {model_key} Production Metrics Report

## Snapshot
- Model type: YOLO detection
- Run directory: {run_dir.relative_to(ROOT).as_posix()}
- Epochs: {epochs}
- Dataset config: {args.get('data', 'config/data.yaml')}

## Key Metrics
- Final Precision: {summary['general_training_metrics']['final']['precision']:.6f}
- Final Recall: {summary['general_training_metrics']['final']['recall']:.6f}
- Final F1 (estimated): {summary['general_training_metrics']['final']['f1_estimated']:.6f}
- Final mAP50: {summary['general_training_metrics']['final']['map50']:.6f}
- Final mAP50-95: {summary['general_training_metrics']['final']['map50_95']:.6f}
- Best mAP50-95: {summary['general_training_metrics']['best_map50_95']:.6f} (epoch {summary['general_training_metrics']['best_map50_95_epoch']})

## Training/Loss Metrics
- Train Box Loss: {summary['general_training_metrics']['final']['train_box_loss']:.6f}
- Train Cls Loss: {summary['general_training_metrics']['final']['train_cls_loss']:.6f}
- Train DFL Loss: {summary['general_training_metrics']['final']['train_dfl_loss']:.6f}
- Val Box Loss: {summary['general_training_metrics']['final']['val_box_loss']:.6f}
- Val Cls Loss: {summary['general_training_metrics']['final']['val_cls_loss']:.6f}
- Val DFL Loss: {summary['general_training_metrics']['final']['val_dfl_loss']:.6f}

## Operational Metrics
- Training time (sec): {summary['operational_metrics']['training_time_sec']}
- Avg epoch time (sec): {summary['operational_metrics']['avg_time_per_epoch_sec']}
- Model size (MB): {summary['operational_metrics']['model_size_mb']}
- Mean latency CPU (ms): {summary['operational_metrics']['latency_throughput_cpu']['latency_ms_mean']}
- Throughput CPU (FPS): {summary['operational_metrics']['latency_throughput_cpu']['throughput_fps']}

## Produced Artifacts
- metrics_summary.json
- training_curves_custom.png
- raw_artifacts/results.png
- raw_artifacts/confusion_matrix.png
- raw_artifacts/confusion_matrix_normalized.png
- raw_artifacts/PR_curve.png
- raw_artifacts/P_curve.png
- raw_artifacts/R_curve.png
- raw_artifacts/F1_curve.png
"""
    write_text(out_dir / "metrics_report.md", report)


def yolo_base_report(model_key: str, model_path: Path, sample_image: Path, common_meta: dict, dataset_meta: dict):
    out_dir = ensure_dir(OUT_ROOT / model_key)
    latency = benchmark_yolo(model_path, sample_image)

    summary = {
        "model_name": model_key,
        "task_type": "object_detection",
        "status": "base_pretrained_only",
        "data_used": {
            "detection_splits": dataset_meta["detection"],
            "notes": "No repository-local training run metrics found for this base checkpoint.",
        },
        "epochs": None,
        "general_training_metrics": {
            "loss": None,
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "map50": None,
            "map50_95": None,
        },
        "advanced_metrics": {
            "roc_auc": None,
            "confusion_matrix": None,
            "iou": None,
            "dice": None,
        },
        "operational_metrics": {
            "model_size_mb": file_size_mb(model_path),
            "latency_throughput_cpu": latency,
        },
        "reproducibility": {
            "model_files": {
                "base_weight": str(model_path.relative_to(ROOT)).replace("\\", "/") if model_path.exists() else None,
            },
            "code_versions": common_meta,
            "configurations": {
                "requirements_file": "requirements.txt",
            },
            "deployment_assets": {
                "streamlit": "app.py",
                "nextjs_api": "geo-ai-ui/app/api/predict/route.ts",
            },
        },
        "artifacts": [],
    }

    write_json(out_dir / "metrics_summary.json", summary)
    report = f"""# {model_key} Base Checkpoint Report

## Status
- Base pretrained checkpoint only
- No local training-run CSV found in repository for full training metrics

## Operational
- Model size (MB): {summary['operational_metrics']['model_size_mb']}
- Mean latency CPU (ms): {summary['operational_metrics']['latency_throughput_cpu']['latency_ms_mean']}
- Throughput CPU (FPS): {summary['operational_metrics']['latency_throughput_cpu']['throughput_fps']}
"""
    write_text(out_dir / "metrics_report.md", report)


def resolve_seg_checkpoint() -> Path:
    candidates = [
        ROOT / "models" / "deeplab_model.pth",
        ROOT / "runs" / "segmentation" / "deeplab_model_best.pth",
    ]
    for path in candidates:
        if path.exists():
            return path
    return ROOT / "models" / "deeplab_model.pth"


def eval_segmentation_model(common_meta: dict, dataset_meta: dict):
    out_dir = ensure_dir(OUT_ROOT / "deeplabv3plus_segmentation")

    ckpt = resolve_seg_checkpoint()
    if not ckpt.exists():
        write_text(out_dir / "metrics_report.md", "Segmentation checkpoint not found.")
        write_json(out_dir / "metrics_summary.json", {"status": "missing_checkpoint"})
        return

    model = smp.DeepLabV3Plus(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=6,
    )
    state = torch.load(str(ckpt), map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    img_dir = ROOT / "seg_dataset" / "valid" / "images"
    msk_dir = ROOT / "seg_dataset" / "valid" / "masks"
    image_paths = sorted([p for p in img_dir.glob("*") if p.is_file()])

    cm = np.zeros((6, 6), dtype=np.int64)
    latencies = []

    for idx, img_path in enumerate(image_paths):
        stem = img_path.stem
        mask_candidates = [msk_dir / f"{stem}.png", msk_dir / f"{stem}.jpg", msk_dir / f"{stem}.jpeg"]
        mask_path = next((p for p in mask_candidates if p.exists()), None)
        if mask_path is None:
            continue

        image = cv2.imread(str(img_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            continue

        image_r = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR)
        mask_r = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)

        x = torch.tensor(image_r.transpose(2, 0, 1) / 255.0, dtype=torch.float32).unsqueeze(0)
        t0 = time.perf_counter()
        with torch.no_grad():
            logits = model(x)
            pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.int32)
        if idx >= 1:
            latencies.append((time.perf_counter() - t0) * 1000.0)

        truth = mask_r.astype(np.int32)
        valid = (truth >= 0) & (truth < 6)
        bins = 6 * truth[valid].ravel() + pred[valid].ravel()
        cm += np.bincount(bins, minlength=36).reshape(6, 6)

    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp

    iou_den = tp + fp + fn
    dice_den = (2 * tp + fp + fn)

    per_iou = np.divide(tp, iou_den, out=np.zeros_like(tp), where=iou_den > 0)
    per_dice = np.divide(2 * tp, dice_den, out=np.zeros_like(tp), where=dice_den > 0)

    macro_iou = float(np.mean(per_iou))
    macro_dice = float(np.mean(per_dice))
    pixel_acc = float(tp.sum() / cm.sum()) if cm.sum() > 0 else 0.0

    mean_ms = float(np.mean(latencies)) if latencies else None
    fps = (1000.0 / mean_ms) if mean_ms and mean_ms > 0 else None

    # confusion matrix heatmap
    fig1 = plt.figure(figsize=(7, 6))
    plt.imshow(cm, cmap="Blues")
    plt.title("Segmentation Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    plt.tight_layout()
    fig1.savefig(out_dir / "segmentation_confusion_matrix.png", dpi=170)
    plt.close(fig1)

    fig2 = plt.figure(figsize=(9, 4))
    x = np.arange(6)
    plt.bar(x - 0.2, per_iou, width=0.4, label="IoU")
    plt.bar(x + 0.2, per_dice, width=0.4, label="Dice")
    plt.xticks(x, [str(i) for i in range(6)])
    plt.xlabel("Class ID")
    plt.ylabel("Score")
    plt.title("Per-class IoU / Dice")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    fig2.savefig(out_dir / "segmentation_per_class_scores.png", dpi=170)
    plt.close(fig2)

    summary = {
        "model_name": "deeplabv3plus_segmentation",
        "task_type": "semantic_segmentation",
        "status": "evaluated_on_valid_split",
        "data_used": {
            "segmentation_splits": dataset_meta["segmentation"],
            "eval_split": "seg_dataset/valid",
            "images_evaluated": int(len(image_paths)),
        },
        "epochs": "unknown_from_checkpoint",
        "general_training_metrics": {
            "pixel_accuracy": pixel_acc,
            "macro_iou": macro_iou,
            "macro_dice": macro_dice,
            "per_class_iou": [float(v) for v in per_iou],
            "per_class_dice": [float(v) for v in per_dice],
            "loss": None,
        },
        "advanced_metrics": {
            "iou": macro_iou,
            "dice": macro_dice,
            "confusion_matrix_artifact": "segmentation_confusion_matrix.png",
        },
        "operational_metrics": {
            "model_size_mb": file_size_mb(ckpt),
            "latency_ms_mean_cpu": round(mean_ms, 3) if mean_ms is not None else None,
            "throughput_fps_cpu": round(fps, 3) if fps is not None else None,
            "resource_usage": None,
        },
        "reproducibility": {
            "model_files": {
                "checkpoint": str(ckpt.relative_to(ROOT)).replace("\\", "/"),
            },
            "training_script": "scripts/training/train_deeplab_seg.py",
            "code_versions": common_meta,
            "configurations": {
                "requirements_file": "requirements.txt",
            },
            "deployment_assets": {
                "streamlit": "app.py",
                "nextjs_api": "geo-ai-ui/app/api/predict/route.ts",
            },
        },
        "artifacts": [
            "segmentation_confusion_matrix.png",
            "segmentation_per_class_scores.png",
        ],
    }

    write_json(out_dir / "metrics_summary.json", summary)
    report = f"""# DeepLabV3+ Segmentation Production Report

## Evaluation Scope
- Checkpoint: {summary['reproducibility']['model_files']['checkpoint']}
- Evaluated split: seg_dataset/valid
- Images evaluated: {summary['data_used']['images_evaluated']}

## Key Metrics
- Pixel Accuracy: {pixel_acc:.6f}
- Macro IoU: {macro_iou:.6f}
- Macro Dice: {macro_dice:.6f}

## Operational
- Model size (MB): {summary['operational_metrics']['model_size_mb']}
- Mean latency CPU (ms): {summary['operational_metrics']['latency_ms_mean_cpu']}
- Throughput CPU (FPS): {summary['operational_metrics']['throughput_fps_cpu']}

## Produced Artifacts
- metrics_summary.json
- segmentation_confusion_matrix.png
- segmentation_per_class_scores.png
"""
    write_text(out_dir / "metrics_report.md", report)


def eval_xgboost_erosion(common_meta: dict, dataset_meta: dict):
    out_dir = ensure_dir(OUT_ROOT / "xgboost_erosion")

    csv_path = ROOT / "terrain_model" / "erosion_dataset.csv"
    model_path = ROOT / "terrain_model" / "erosion_model.pkl"
    if not csv_path.exists() or not model_path.exists():
        write_text(out_dir / "metrics_report.md", "Terrain dataset or model not found.")
        write_json(out_dir / "metrics_summary.json", {"status": "missing_inputs"})
        return

    df = pd.read_csv(csv_path)
    model = joblib.load(model_path)

    target = "erosion" if "erosion" in df.columns else ("erosion_risk" if "erosion_risk" in df.columns else None)
    if target is None:
        write_text(out_dir / "metrics_report.md", "Could not find target column in terrain dataset.")
        write_json(out_dir / "metrics_summary.json", {"status": "missing_target"})
        return

    if hasattr(model, "feature_names_in_"):
        feature_cols = [c for c in model.feature_names_in_ if c in df.columns]
    else:
        feature_cols = [c for c in df.columns if c != target]

    X = df[feature_cols]
    y = df[target].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y if y.nunique() > 1 else None,
    )

    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = y_pred.astype(float)

    acc = float(accuracy_score(y_test, y_pred))
    p = float(precision_score(y_test, y_pred, zero_division=0))
    r = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_test, y_prob)) if y_test.nunique() > 1 else None
    ap = float(average_precision_score(y_test, y_prob)) if y_test.nunique() > 1 else None

    mse = float(mean_squared_error(y_test, y_prob))
    mae = float(mean_absolute_error(y_test, y_prob))
    r2 = float(r2_score(y_test, y_prob))

    cm = confusion_matrix(y_test, y_pred)
    report_text = classification_report(y_test, y_pred, digits=4)

    fig1 = plt.figure(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(values_format="d")
    plt.title("XGBoost Confusion Matrix")
    plt.tight_layout()
    fig1.savefig(out_dir / "confusion_matrix_eval.png", dpi=170)
    plt.close(fig1)

    if roc_auc is not None:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fig2 = plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"ROC AUC={roc_auc:.4f}")
        plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.grid(alpha=0.25)
        plt.tight_layout()
        fig2.savefig(out_dir / "roc_curve_eval.png", dpi=170)
        plt.close(fig2)

    prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_prob)
    fig3 = plt.figure(figsize=(6, 5))
    plt.plot(rec_curve, prec_curve, label=f"AP={ap:.4f}" if ap is not None else "PR")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    fig3.savefig(out_dir / "pr_curve_eval.png", dpi=170)
    plt.close(fig3)

    # runtime benchmark
    one = X_test.iloc[[0]]
    times = []
    for _ in range(200):
        t0 = time.perf_counter()
        _ = model.predict_proba(one) if hasattr(model, "predict_proba") else model.predict(one)
        times.append((time.perf_counter() - t0) * 1000.0)

    latency_ms = float(np.mean(times))
    throughput = 1000.0 / latency_ms if latency_ms > 0 else None

    # copy existing artifacts if available
    copied = []
    for src_name in [
        ROOT / "terrain_model" / "artifacts" / "xgboost_classification_report.txt",
        ROOT / "terrain_model" / "artifacts" / "xgboost_confusion_matrix.png",
        ROOT / "terrain_model" / "feature_importance.png",
    ]:
        dst = out_dir / "raw_artifacts" / src_name.name
        if safe_copy(src_name, dst):
            copied.append(str(dst.relative_to(OUT_ROOT)).replace("\\", "/"))

    write_text(out_dir / "classification_report_eval.txt", report_text)

    summary = {
        "model_name": "xgboost_erosion",
        "task_type": "binary_classification",
        "status": "evaluated_on_holdout_split",
        "data_used": {
            "rows_total": int(df.shape[0]),
            "features_used": feature_cols,
            "target": target,
            "train_rows": int(X_train.shape[0]),
            "test_rows": int(X_test.shape[0]),
            "terrain_rows_reference": dataset_meta["terrain_rows"],
        },
        "epochs": None,
        "general_training_metrics": {
            "accuracy": acc,
            "precision": p,
            "recall": r,
            "f1": f1,
            "roc_auc": roc_auc,
            "average_precision": ap,
        },
        "regression_metrics": {
            "mse": mse,
            "mae": mae,
            "r2": r2,
            "rmse": float(math.sqrt(mse)),
        },
        "advanced_metrics": {
            "confusion_matrix": cm.tolist(),
        },
        "operational_metrics": {
            "model_size_mb": file_size_mb(model_path),
            "latency_ms_mean_cpu": round(latency_ms, 6),
            "throughput_fps_cpu": round(throughput, 3) if throughput else None,
            "resource_usage": None,
        },
        "reproducibility": {
            "training_script": "terrain_model/train_xgboost.py",
            "model_files": {
                "model": str(model_path.relative_to(ROOT)).replace("\\", "/"),
                "dataset": str(csv_path.relative_to(ROOT)).replace("\\", "/"),
            },
            "code_versions": common_meta,
            "configurations": {
                "requirements_file": "requirements.txt",
            },
            "logs": {
                "existing_report": "terrain_model/artifacts/xgboost_classification_report.txt",
            },
            "deployment_assets": {
                "streamlit": "app.py",
                "nextjs_api": "geo-ai-ui/app/api/predict/route.ts",
            },
        },
        "artifacts": [
            "classification_report_eval.txt",
            "confusion_matrix_eval.png",
            "roc_curve_eval.png",
            "pr_curve_eval.png",
        ] + copied,
    }

    write_json(out_dir / "metrics_summary.json", summary)

    report = f"""# XGBoost Erosion Production Report

## Evaluation Scope
- Dataset: {summary['reproducibility']['model_files']['dataset']}
- Target: {target}
- Features: {', '.join(feature_cols)}
- Train/Test rows: {summary['data_used']['train_rows']}/{summary['data_used']['test_rows']}

## Classification Metrics
- Accuracy: {acc:.6f}
- Precision: {p:.6f}
- Recall: {r:.6f}
- F1: {f1:.6f}
- ROC-AUC: {roc_auc}
- Average Precision: {ap}

## Regression-style Metrics on Probabilities
- MSE: {mse:.6f}
- MAE: {mae:.6f}
- RMSE: {math.sqrt(mse):.6f}
- R2: {r2:.6f}

## Operational
- Model size (MB): {summary['operational_metrics']['model_size_mb']}
- Mean latency CPU (ms): {summary['operational_metrics']['latency_ms_mean_cpu']}
- Throughput CPU (FPS): {summary['operational_metrics']['throughput_fps_cpu']}

## Produced Artifacts
- metrics_summary.json
- classification_report_eval.txt
- confusion_matrix_eval.png
- roc_curve_eval.png
- pr_curve_eval.png
"""
    write_text(out_dir / "metrics_report.md", report)


def write_global_index(common_meta: dict, dataset_meta: dict):
    model_dirs = [p for p in OUT_ROOT.iterdir() if p.is_dir()]
    index = {
        "generated_at": datetime.now().isoformat(),
        "output_root": str(OUT_ROOT.relative_to(ROOT)).replace("\\", "/"),
        "models": sorted([p.name for p in model_dirs]),
        "code_versions": common_meta,
        "dataset_inventory": dataset_meta,
    }
    write_json(OUT_ROOT / "index.json", index)

    lines = [
        "# Model Metrics Artifact Index",
        "",
        f"Generated at: {index['generated_at']}",
        f"Git commit: {common_meta['git_commit']}",
        "",
        "## Models Included",
    ]
    for name in index["models"]:
        lines.append(f"- {name}: see {name}/metrics_report.md and {name}/metrics_summary.json")

    lines += [
        "",
        "## Dataset Inventory",
        json.dumps(dataset_meta, indent=2),
        "",
        "## Notes",
        "- Detection uses mAP metrics (standard for object detection).",
        "- Segmentation includes IoU and Dice computed on validation split.",
        "- XGBoost includes both classification and regression-style probability metrics.",
        "- Base checkpoints without local training runs include operational metrics and reproducibility metadata.",
    ]
    write_text(OUT_ROOT / "README.md", "\n".join(lines))


def main():
    ensure_dir(OUT_ROOT)

    common_meta = {
        "git_commit": git_commit_hash(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "requirements_file": "requirements.txt",
    }
    dataset_meta = dataset_inventory()

    # representative sample image for latency testing
    sample_image_candidates = [
        ROOT / "dataset" / "test" / "images",
        ROOT / "dataset" / "valid" / "images",
        ROOT / "dataset" / "train" / "images",
    ]
    sample_image = None
    for folder in sample_image_candidates:
        if folder.exists():
            files = [p for p in folder.glob("*") if p.is_file()]
            if files:
                sample_image = files[0]
                break

    # Trained YOLO runs
    yolo_model_report(
        "yolov8s_archaeology2",
        ROOT / "runs" / "detect" / "yolov8s_archaeology2",
        ROOT / "models" / "yolov8s.pt",
        sample_image,
        common_meta,
        dataset_meta,
    )
    yolo_model_report(
        "yolov8n_train2",
        ROOT / "runs" / "detect" / "train2",
        ROOT / "models" / "yolov8n.pt",
        sample_image,
        common_meta,
        dataset_meta,
    )

    # Base checkpoints (no local training CSV in repo)
    for key, path in [
        ("yolo11n_base", ROOT / "models" / "yolo11n.pt"),
        ("yolov8s_base", ROOT / "models" / "yolov8s.pt"),
        ("yolov8n_base", ROOT / "models" / "yolov8n.pt"),
    ]:
        yolo_base_report(key, path, sample_image, common_meta, dataset_meta)

    # Segmentation and erosion
    eval_segmentation_model(common_meta, dataset_meta)
    eval_xgboost_erosion(common_meta, dataset_meta)

    write_global_index(common_meta, dataset_meta)
    print(f"Artifacts generated at: {OUT_ROOT}")


if __name__ == "__main__":
    main()
