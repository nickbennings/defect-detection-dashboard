# Bottle Defect Detection Dashboard

A machine learning demonstration that identifies potentially defective
bottles using images from the MVTec AD dataset. The project uses a
pretrained ResNet-18 model to extract image features, then compares those
features with a set of normal bottle images.

A Streamlit dashboard lets users upload an image and view its anomaly
score and detection result.

## How it works

1. Load normal bottle images from the MVTec AD training set.
2. Use pretrained ResNet-18 to extract features from each image.
3. Build a nearest-neighbor detector using normal reference images.
4. Set an anomaly threshold using held-out normal training images.
5. Evaluate the detector on the separate test set.
6. Upload an image to the dashboard to check for a potential anomaly.

The pretrained ResNet-18 weights are used as a feature extractor; they
are not fine-tuned in this project.

## Results

On the MVTec AD bottle test set:

- 62 of 63 defective images were flagged.
- 19 of 20 normal images were correctly identified.
- One defective image was missed, and one normal image was flagged.

These results are based on a small test set from one dataset and may not
generalize to different bottles, cameras, or lighting conditions.

## Technologies

- Python and PyTorch
- torchvision and pretrained ResNet-18
- scikit-learn
- Streamlit
- Pillow and joblib

## Dataset

This project uses the bottle category of the
[MVTec AD dataset](https://www.mvtec.com/research-teaching/datasets/mvtec-ad),
created by MVTec Software GmbH.

The dataset is available under the
[CC BY-NC-SA 4.0 license](https://creativecommons.org/licenses/by-nc-sa/4.0/).
It is for noncommercial use. The dataset is not included in this repository.

## Run locally

1. Download the bottle category from the MVTec AD dataset page.
2. Extract it so that the project contains `data/bottle/train` and
   `data/bottle/test`.
3. Create and activate a Python virtual environment.
4. Install the packages:

```bash
pip install torch torchvision streamlit scikit-learn pillow joblib numpy
```

5. Build the detector:

```bash
python train_detector.py
```

6. Evaluate it:

```bash
python evaluate_detector.py
```

7. Start the dashboard:

```bash
python -m streamlit run app.py
```

The first run downloads pretrained ResNet-18 weights. The detector file
is generated locally and is not included in the repository.

## Limitations

This is a learning demonstration, not a production inspection system.
It has been evaluated only on the bottle test images in MVTec AD and
should not be used for real-world quality-control decisions.
