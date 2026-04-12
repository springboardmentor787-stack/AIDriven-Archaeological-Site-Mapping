import torch
import segmentation_models_pytorch as smp

# create model
model = smp.DeepLabV3Plus(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=2,
)

# move to GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

print("DeepLabV3+ model ready!")