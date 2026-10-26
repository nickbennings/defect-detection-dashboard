from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
from torchvision.models import resnet18, ResNet18_Weights

# Load the saved detector and the same pretrained feature extractor.
detector = joblib.load("detector.joblib")
neighbors = detector["neighbors"]
threshold = detector["threshold"]

weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
model.fc = torch.nn.Identity()
model.eval()
preprocess = weights.transforms()

test_path = Path("data/bottle/test")
actual = []
predicted = []

print("Evaluating test images...")

with torch.no_grad():
    for category in sorted(test_path.iterdir()):
        if not category.is_dir():
            continue

        image_paths = sorted(category.glob("*.png"))
        flagged = 0

        for image_path in image_paths:
            with Image.open(image_path) as image:
                tensor = preprocess(image.convert("RGB")).unsqueeze(0)

            features = model(tensor).numpy()
            distance, _ = neighbors.kneighbors(features)
            is_anomaly = distance[0, 0] > threshold

            actual.append(0 if category.name == "good" else 1)
            predicted.append(int(is_anomaly))
            flagged += int(is_anomaly)

        print(f"  {category.name}: {flagged}/{len(image_paths)} flagged")

print("\nConfusion matrix (rows = actual, columns = predicted):")
print("                  Predicted good  Predicted anomaly")
matrix = confusion_matrix(actual, predicted, labels=[0, 1])
print(f"Actual good       {matrix[0, 0]:>14}  {matrix[0, 1]:>17}")
print(f"Actual anomaly    {matrix[1, 0]:>14}  {matrix[1, 1]:>17}")

print("\nClassification report:")
print(
    classification_report(
        actual,
        predicted,
        labels=[0, 1],
        target_names=["good", "anomaly"],
        zero_division=0,
    )
)
