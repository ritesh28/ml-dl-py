# HOG + SVM Face Detection

A classical face-detection pipeline using **HOG** features and a **linear SVM**, trained on LFW faces plus negative patches, then applied with a sliding window on a new image.

## What it does

1. **HOG demo** — Shows Histogram of Oriented Gradients features on a sample image via `skimage.feature`.
2. **Positive samples** — Loads face thumbnails from `sklearn.datasets.fetch_lfw_people`.
3. **Negative samples** — Extracts non-face patches from `skimage` images with `PatchExtractor`, sized to match the positives.
4. **Train** — Combines sets, extracts HOG features, and trains a `LinearSVC` (with a `GaussianNB` baseline and grid search).
5. **Detect** — Scans a new image with a sliding window / image pyramid and reports face locations.
6. **Caveats** — Notes limitations of this simple detector (false positives, scale, etc.).

## Setup

Uses [uv](https://docs.astral.sh/uv/) for the project environment.

```bash
cd hog-svm-face-detection
uv sync
```

Register the kernel (optional, for Jupyter / VS Code / Cursor):

```bash
uv run python -m ipykernel install --user --name=hog-svm-face-detection --display-name="Python (hog-svm-face-detection)"
```

## Run the notebook

Open `hog-svm-face-detection.ipynb`, click **Select Kernel** → **Jupyter Kernel…**, and choose **Python (hog-svm-face-detection)**. Reload the IDE if you do not see it.

## Dependencies

| Package       | Role                                      |
| ------------- | ----------------------------------------- |
| scikit-image  | Sample images, HOG, transforms            |
| scikit-learn  | LFW faces, PatchExtractor, SVM / Naive Bayes |
| matplotlib    | Plotting                                  |
| numpy         | Arrays                                    |
| ipykernel     | Jupyter kernel for the local environment  |
