!pip install rasterio

# ---

from google.colab import files
uploaded = files.upload()

# ---

import cv2
import numpy as np
import pandas as pd
import random

# ---

all_features = []

for filename in uploaded.keys():

    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize for speed
    img = cv2.resize(img, (256, 256))

    red = img[:,:,0].astype(float)
    green = img[:,:,1].astype(float)

    # NDVI
    ndvi = (green - red) / (green + red + 1e-5)

    # Slope
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(float)
    dx, dy = np.gradient(gray)
    slope = np.sqrt(dx**2 + dy**2)

    # Elevation
    elevation = gray

    h, w = gray.shape

    for i in range(h):
        for j in range(w):
            all_features.append([
                slope[i][j],
                ndvi[i][j],
                elevation[i][j]
            ])

# ---

sample_size = 200000  # 2 lakh rows (perfect size)

if len(all_features) > sample_size:
    all_features = random.sample(all_features, sample_size)

# ---

df = pd.DataFrame(all_features, columns=["slope", "ndvi", "elevation"])

# ---

def label_data(slope, ndvi):
    if slope > 20 and ndvi < 0.2:
        return 1  # erosion-prone
    else:
        return 0  # stable

df["label"] = [label_data(s, n) for s, n in zip(df["slope"], df["ndvi"])]

# ---

print(df.head())

print("\nClass Distribution:")
print(df["label"].value_counts())

# ---

import matplotlib.pyplot as plt
plt.imshow(ndvi, cmap='RdYlGn')
plt.title("NDVI Map")
plt.colorbar()
plt.show()

# ---

plt.imshow(slope, cmap='terrain')
plt.title("Slope Map")
plt.colorbar()
plt.show()

# ---

df.to_csv("erosion_dataset.csv", index=False)

# ---

from google.colab import files
files.download("erosion_dataset.csv")