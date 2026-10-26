from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from torchvision.models import resnet18, ResNet18_Weights

# Load the pretrained model and remove its classification layer.
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
model.fc = torch.nn.Identity()
model.eval()

# Use the image preparation expected by the pretrained model.
preprocess = weights.transforms()

image_paths = sorted(Path("data/bottle/train/good").glob("*.png"))

if len(image_paths) < 10:
    raise ValueError("Not enough training images found in data/bottle/train/good")

# Reserve some normal images to help set the anomaly threshold.
reference_paths, validation_paths = train_test_split(
    image_paths, test_size=0.2, random_state=42
)


def extract_features(paths):
    features = []

    with torch.no_grad():
        for index, path in enumerate(paths, start=1):
            with Image.open(path) as image:
                tensor = preprocess(image.convert("RGB")).unsqueeze(0)

            features.append(model(tensor).squeeze(0).numpy())

            if index % 20 == 0 or index == len(paths):
                print(f"  Processed {index}/{len(paths)} images")

    return np.stack(features)


print("Extracting reference features...")
reference_features = extract_features(reference_paths)

print("Extracting validation features...")
validation_features = extract_features(validation_paths)

# Compare each validation image with its closest normal reference image.
neighbors = NearestNeighbors(n_neighbors=1, metric="euclidean")
neighbors.fit(reference_features)

distances, _ = neighbors.kneighbors(validation_features)
threshold = float(np.percentile(distances[:, 0], 95))

joblib.dump(
    {"neighbors": neighbors, "threshold": threshold},
    "detector.joblib",
)

print(f"\nThreshold: {threshold:.4f}")
print("Detector saved as detector.joblib")
