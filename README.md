<p align="center">
  <a href="https://github.com/ritesh28/ml-dl-py" target="_blank">
    <img data-source="github" loading="lazy" alt="Machine Learning" src="https://github.com/ritesh28/ml-dl-py/raw/main/banner.png" width="750"/>
  </a>
</p>

# ML-DL-py

Hands-on Machine Learning notebooks in Python — each example is a self-contained [uv](https://docs.astral.sh/uv/) project with its own environment and README.

## Description

Practical walkthroughs of classical ML workflows: load data, train a model, evaluate, and experiment. Topics include text classification, face detection, generative modeling, and digit recognition.

## Project layout

Each example lives in its own root-level folder. Folder names follow **algorithm + task** (e.g. `naive-bayes-text-classification`, `hog-svm-face-detection`).

Every project is a [uv](https://docs.astral.sh/uv/) environment: `cd` into the folder and run `uv sync`, then open the notebook with that project’s kernel.

## Hugging Face env (`.env`)

Copy [`.env.example`](.env.example) to `.env` at the repo root and set:

- `HF_HOME` — shared model cache (default `~/ml-models`)
- `HF_TOKEN` — optional Hugging Face token for higher rate limits

Notebooks / CLIs in [hf-text-to-speech](hf-text-to-speech/), [mflux-text-to-image](mflux-text-to-image/), and [rembg-background-removal](rembg-background-removal/) load this file via `python-dotenv`. `.env` is gitignored.

## Projects

| Name                                                                                | Description                                                  |
| ----------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| [naive-bayes-text-classification](naive-bayes-text-classification/)                 | Multinomial Naive Bayes text classification on 20 Newsgroups |
| [hog-svm-face-detection](hog-svm-face-detection/)                                   | HOG features + linear SVM face detector                      |
| [gmm-handwritten-digit-generation](gmm-handwritten-digit-generation/)               | PCA + GMM to synthesize handwritten digits                   |
| [isomap-gaussian-nb-digit-classification](isomap-gaussian-nb-digit-classification/) | Isomap embedding + GaussianNB digit classification           |
| [hf-text-to-speech](hf-text-to-speech/)                                             | Local TTS with hexgrad/Kokoro-82M from Hugging Face          |
| [mflux-text-to-image](mflux-text-to-image/)                                         | FLUX.1-schnell via mflux/MLX (4-bit; not HF diffusers)       |
| [rembg-background-removal](rembg-background-removal/)                               | Batch background removal with danielgatis/rembg              |
