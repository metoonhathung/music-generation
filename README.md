# Music Generation

[Demo](https://metoonhathung-music-generation.streamlit.app/)

Description: A Python application that generates ragtime piano music with seven deep-learning models
trained from scratch: RNN, CNN (WaveNet), VAE, Transformer, GAN, A2C and GPT-2. See
[REPORT.md](REPORT.md) for how each model works and how it trained.

Technologies: PyTorch, FastAPI, Streamlit, Hugging Face

## Project layout

| Path | What it is |
|---|---|
| `music_generation.ipynb` | Training for RNN, CNN, VAE, Transformer, GAN and A2C |
| `huggingface_transformers.ipynb` | Training for GPT-2 |
| `main.py`, `util.py` | Streamlit app (new stack) |
| `app/` | FastAPI service and model code (old stack) |
| `index.html`, `script.js` | Web page for the FastAPI service (old stack) |
| `checkpoint/` | Full training checkpoints, for resuming training (local only) |
| `app/checkpoint/` | Half-precision checkpoints the apps serve, hosted on [Hugging Face](https://huggingface.co/metoonhathung/music-generation-models) |

## Setup

```
pipenv install
pipenv shell
```

## Run the new stack (Streamlit)

All seven models run inside the Streamlit app. Missing weights are downloaded from Hugging Face on first
use; no API or API key is needed.

```
streamlit run main.py
```

Open http://localhost:8501.

## Run the old stack (FastAPI + web page)

1. Create `.env` from `.env.example` and set `API_KEY` to any value.
2. Download the model weights into `app/checkpoint/` (the API only reads local files):
   ```
   hf download metoonhathung/music-generation-models --local-dir app/checkpoint
   ```
3. Start the API:
   ```
   uvicorn app.main:app --reload --host 0.0.0.0 --port 80
   ```
4. In a second terminal, serve the web page and open http://localhost:8765/index.html, entering the same API key:
   ```
   python -m http.server 8765
   ```

Or call the API directly (`model`: `rnn`, `cnn`, `trf`, `vae`, `gan`, `a2c` or `gpt2`):

```
curl -X POST http://localhost/generate -H "X-API-Key: your-api-key" -H "Content-Type: application/json" -d '{"model": "trf", "length": 600, "prefix": [1]}' -o output.midi
```

`length` is the number of MIDI events (about 45 per second of music; `trf` and `gan` allow at most 2048).
`prefix` holds the event IDs to continue from; `1` is the start of a piece (`gpt2` reads raw IDs, 3 lower).

If port 80 is refused, use `--port 8000` and change the URL in `script.js` to match.

## Build image

The image contains only `Pipfile`, `Pipfile.lock` and `app/` (see `.dockerignore`); secrets are never copied
in. Populate `app/checkpoint/` first (step 2 above), and pass `API_KEY` at runtime.

```
docker login -u metoonhathung
docker build -t music-generation .
docker run -p 80:80 --env-file .env music-generation
docker tag music-generation metoonhathung/music-generation:latest
docker push metoonhathung/music-generation:latest
```

## Update the hosted models

After retraining, save half-precision copies to `app/checkpoint/` and upload them:

```
hf auth login
hf upload metoonhathung/music-generation-models app/checkpoint . --include rnn.pt cnn.pt transformer.pt vae.pt generator.pt a2c.pt "gpt2/*"
```

## Observability

[Docker Hub](https://hub.docker.com/r/metoonhathung/music-generation)

[Google Cloud](https://console.cloud.google.com/run/detail/us-central1/metoonhathung-music-generation-api/metrics?inv=1&invt=Ab4zGQ&project=music-generation-366602)

[Hugging Face (GPT-2 training)](https://huggingface.co/metoonhathung/music-generation)

[Hugging Face (served models)](https://huggingface.co/metoonhathung/music-generation-models)
