# Music Generation

[Demo](https://metoonhathung-music-generation.streamlit.app/)

Description: A Python application to generate songs using Deep Learning models.

Technologies: PyTorch, FastAPI, Streamlit

## Run locally

```
pipenv install
pipenv shell
streamlit run main.py
uvicorn app.main:app --reload --host 0.0.0.0 --port 80
```
