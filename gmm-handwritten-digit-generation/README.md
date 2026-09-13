# GMM Handwritten Digit Generation

Uses a **Gaussian Mixture Model** (after **PCA**) as a generative model to sample new handwritten digits from the distribution of `sklearn` digits data.

## What it does

1. **Load digits** — Loads `sklearn.datasets.load_digits` and visualizes sample images.
2. **Reduce dimension** — Fits `PCA` so the mixture can be estimated in a lower-dimensional space.
3. **Fit GMM** — Trains `sklearn.mixture.GaussianMixture` on the PCA embeddings.
4. **Generate** — Draws new samples from the GMM and inverse-transforms them back to image space to show synthetic digits.

## Setup

Uses [uv](https://docs.astral.sh/uv/) for the project environment.

```bash
cd gmm-handwritten-digit-generation
uv sync
```

Register the kernel (optional, for Jupyter / VS Code / Cursor):

```bash
uv run python -m ipykernel install --user --name=gmm-handwritten-digit-generation --display-name="Python (gmm-handwritten-digit-generation)"
```

## Run the notebook

Open `gmm-handwritten-digit-generation.ipynb`, click **Select Kernel** → **Jupyter Kernel…**, and choose **Python (gmm-handwritten-digit-generation)**. Reload the IDE if you do not see it.

## Dependencies

| Package       | Role                                      |
| ------------- | ----------------------------------------- |
| scikit-learn  | Digits dataset, PCA, GaussianMixture      |
| matplotlib    | Plotting                                  |
| numpy         | Arrays                                    |
| ipykernel     | Jupyter kernel for the local environment  |
