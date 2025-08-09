# Music Generation

[Demo](https://metoonhathung-music-generation.streamlit.app/)

Description: A Python application to generate songs using Deep Learning models.

Technologies: PyTorch, FastAPI, Streamlit, Hugging Face

## Run locally

```
pipenv install
pipenv shell
streamlit run main.py
uvicorn app.main:app --reload --host 0.0.0.0 --port 80
```

## Build image

```
docker login -u metoonhathung
docker build -t music-generation .
docker tag music-generation metoonhathung/music-generation:latest
docker push metoonhathung/music-generation:latest
```

## Observability

[Docker Hub](https://hub.docker.com/r/metoonhathung/music-generation)

[Google Cloud](https://console.cloud.google.com/run/detail/us-central1/metoonhathung-music-generation-api/metrics?inv=1&invt=Ab4zGQ&project=music-generation-366602)

[Hugging Face](https://huggingface.co/metoonhathung/music-generation)
