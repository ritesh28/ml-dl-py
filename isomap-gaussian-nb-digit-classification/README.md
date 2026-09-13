# Isomap + GaussianNB Digit Classification

Identifies handwritten digits with **Isomap** for unsupervised dimensionality reduction, then **Gaussian Naive Bayes** for classification on the `sklearn` digits dataset.

## What it does

1. **Load & visualize** — Loads `sklearn.datasets.load_digits` and plots sample digit images.
2. **Dimensionality reduction** — Embeds digits in 2D with `sklearn.manifold.Isomap` to inspect cluster separation.
3. **Classify** — Splits train/test, fits `GaussianNB` on the (reduced) features, and reports accuracy.
4. **Evaluate** — Plots a confusion matrix with `seaborn` / `matplotlib`.

## Setup

Uses [uv](https://docs.astral.sh/uv/) for the project environment.

```bash
cd isomap-gaussian-nb-digit-classification
uv sync
```

Register the kernel (optional, for Jupyter / VS Code / Cursor):

```bash
uv run python -m ipykernel install --user --name=isomap-gaussian-nb-digit-classification --display-name="Python (isomap-gaussian-nb-digit-classification)"
```

## Run the notebook

Open `isomap-gaussian-nb-digit-classification.ipynb`, click **Select Kernel** → **Jupyter Kernel…**, and choose **Python (isomap-gaussian-nb-digit-classification)**. Reload the IDE if you do not see it.

## Dependencies

| Package       | Role                                      |
| ------------- | ----------------------------------------- |
| scikit-learn  | Digits dataset, Isomap, GaussianNB, metrics |
| seaborn       | Confusion-matrix heatmap                  |
| matplotlib    | Plotting                                  |
| ipykernel     | Jupyter kernel for the local environment  |
