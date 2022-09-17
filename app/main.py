from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from app.util import load_rnn, load_cnn, load_transformer, load_vae, load_gan, generate_buffer, GenerateRequest

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_dict = {}

@app.on_event("startup")
async def startup():
    global model_dict
    model_dict = {
        "rnn": load_rnn(),
        "cnn": load_cnn(),
        "transformer": load_transformer(),
        "vae": load_vae(),
        "gan": load_gan()
    }

@app.get("/")
async def root():
    return {"message": "OK"}

@app.post("/generate")
async def generate(payload: GenerateRequest):
    buffer = generate_buffer(model_dict[payload.model], payload.length, payload.prefix)
    return StreamingResponse(buffer, media_type="audio/midi")
