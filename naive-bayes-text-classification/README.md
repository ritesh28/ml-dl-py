# Naive Bayes Text Classification

Text classification with **Multinomial Naive Bayes** and TF-IDF features on a subset of the [20 Newsgroups](https://scikit-learn.org/stable/datasets/real_world.html#the-20-newsgroups-text-dataset) dataset.

## What it does

1. **Fetch data** — Loads 20 Newsgroups via `sklearn.datasets.fetch_20newsgroups`, then restricts training and test sets to four categories:
   - `talk.religion.misc`
   - `soc.religion.christian`
   - `sci.space`
   - `comp.graphics`
2. **Train** — Builds a `sklearn` pipeline of `TfidfVectorizer` → `MultinomialNB`, fits on the training posts, and predicts test labels.
3. **Evaluate** — Plots a confusion matrix with `seaborn` / `matplotlib` to compare true vs predicted categories.
4. **Predict** — Exposes a small `predict_category(text)` helper for ad-hoc strings (e.g. space payloads → `sci.space`, screen resolution → `comp.graphics`).

## Setup

Uses [uv](https://docs.astral.sh/uv/) for the project environment.

```bash
cd naive-bayes-text-classification
uv sync
```

Register the kernel (optional, for Jupyter / VS Code / Cursor):

```bash
uv run python -m ipykernel install --user --name=naive-bayes-text-classification --display-name="Python (naive-bayes-text-classification)"
```

## Run the notebook

Open `naive-bayes-text-classification.ipynb`, click **Select Kernel** → **Jupyter Kernel…**, and choose **Python (naive-bayes-text-classification)**. Reload the IDE if you do not see it.

## Dependencies

| Package       | Role                                      |
| ------------- | ----------------------------------------- |
| scikit-learn  | Dataset, TF-IDF, Naive Bayes, metrics     |
| seaborn       | Confusion-matrix heatmap                  |
| matplotlib    | Plotting                                  |
| ipykernel     | Jupyter kernel for the local environment  |
