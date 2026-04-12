import random
import pandas as pd
import json
from datetime import datetime
from pathlib import Path

data = []
class_0, class_1 = [], []

while len(class_0) < 500 or len(class_1) < 500:

    slope = random.uniform(0, 45)
    vegetation = random.uniform(0, 1)
    elevation = random.uniform(300, 500)
    rainfall = random.uniform(0, 200)
    soil = random.choice([1, 2, 3])

    # NEW FEATURES 🔥
    boulders = random.uniform(0, 0.5)
    ruins = random.uniform(0, 0.3)
    structures = random.uniform(0, 0.2)

    score = (
        0.5 * slope +
        -40 * vegetation +
        0.2 * rainfall +
        -5 * soil +
        0.05 * elevation +
        15 * boulders +
        10 * ruins +
        8 * structures +
        random.uniform(-10, 10)
    )

    erosion = 1 if score > 20 else 0

    row = [
        slope, vegetation, elevation, rainfall, soil,
        boulders, ruins, structures, erosion
    ]

    if erosion == 0 and len(class_0) < 500:
        class_0.append(row)
    elif erosion == 1 and len(class_1) < 500:
        class_1.append(row)

data = class_0 + class_1
random.shuffle(data)

df = pd.DataFrame(data, columns=[
    "slope", "vegetation", "elevation", "rainfall", "soil",
    "boulders", "ruins", "structures", "erosion"
])

df.to_csv("terrain_model/erosion_dataset.csv", index=False)

provenance_path = Path("terrain_model/erosion_dataset_provenance.json")
provenance = {
    "dataset_file": "terrain_model/erosion_dataset.csv",
    "generator_script": "terrain_model/generate_dataset_v3.py",
    "generation_method": "synthetic_random_balanced",
    "generated_at_utc": datetime.utcnow().isoformat() + "Z",
    "row_count": int(df.shape[0]),
    "column_count": int(df.shape[1]),
    "columns": list(df.columns),
    "target_column": "erosion",
    "class_distribution": {
        "0": int((df["erosion"] == 0).sum()),
        "1": int((df["erosion"] == 1).sum()),
    },
    "balance_strategy": "while-loop with caps: 500 class_0 + 500 class_1",
}

with provenance_path.open("w", encoding="utf-8") as f:
    json.dump(provenance, f, indent=2)

print("✅ Dataset v3 created")
print("📝 Provenance saved to terrain_model/erosion_dataset_provenance.json")
print(df.head())

