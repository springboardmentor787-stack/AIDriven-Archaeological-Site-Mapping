from torchmetrics import JaccardIndex, Dice

iou = JaccardIndex(task="binary")
dice = Dice()

print("Metrics ready")