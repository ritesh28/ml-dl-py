# Hugging Face Text-to-Speech (Kokoro-82M)

Local text-to-speech with **[hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)** — an 82M-parameter open-weight TTS model (Apache 2.0) via the [`kokoro`](https://github.com/hexgrad/kokoro) inference library.

## What it does

1. **Load env** — Reads repo-root `.env` for `HF_HOME` / `HF_TOKEN` (see `.env.example`).
2. **Load pipeline** — Builds a `KPipeline` for American English (`lang_code='a'`), which downloads Kokoro-82M weights from Hugging Face on first run into `HF_HOME`.
3. **Synthesize** — Converts sample text to speech with a chosen voice (default `af_heart`).
4. **Play & save** — Plays audio in the notebook and writes WAV files under `output/` at 24 kHz.

## Setup

Uses [uv](https://docs.astral.sh/uv/) for the project environment.

**Env (`.env`):** Copy the repo-root `.env.example` to `.env` and set `HF_HOME` (default `~/ml-models`) and optionally `HF_TOKEN`. The notebook loads these before downloading weights.

**System dependency:** Kokoro uses [espeak-ng](https://github.com/espeak-ng/espeak-ng) for English out-of-dictionary fallback. On macOS:

```bash
brew install espeak-ng
```

Then:

```bash
cd hf-text-to-speech
uv sync
```

Register the kernel (optional, for Jupyter / VS Code / Cursor):

```bash
uv run python -m ipykernel install --user --name=hf-text-to-speech --display-name="Python (hf-text-to-speech)"
```

## Run the notebook

Open `hf-text-to-speech.ipynb`, click **Select Kernel** → **Jupyter Kernel…**, and choose **Python (hf-text-to-speech)**. Reload the IDE if you do not see it.

## Voices & languages

- Voices are listed in the model’s [VOICES.md](https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md).
- Change `voice=` in the notebook (e.g. `af_heart`, `am_adam`, `bf_emma`).
- Language codes: `a` = American English, `b` = British English. Keep `lang_code` aligned with the voice.

## Dependencies

| Package       | Role                                           |
| ------------- | ---------------------------------------------- |
| kokoro        | Kokoro-82M inference (`KPipeline`)             |
| torch         | Model runtime                                  |
| transformers  | Hugging Face model loading (pinned for wheels) |
| soundfile     | Write WAV output                               |
| python-dotenv | Load repo-root `.env` (`HF_HOME` / token)      |
| ipykernel     | Jupyter kernel for the local environment       |
