import os
import tempfile
from functools import lru_cache
from huggingface_hub import snapshot_download
from app import util

HF_REPO = "metoonhathung/music-generation-models"
CHECKPOINTS = {"rnn": "rnn.pt", "cnn": "cnn.pt", "trf": "transformer.pt", "vae": "vae.pt", "gan": "generator.pt", "a2c": "a2c.pt", "gpt2": "gpt2"}

@lru_cache(maxsize=None)
def get_model(model):
    # Loaded on first use and kept in memory. The Hugging Face repo mirrors app/checkpoint, so a
    # missing checkpoint (file, or folder for gpt2) is downloaded to where the app's loaders look.
    if not os.path.exists(f"{util.BASE_DIR}/checkpoint/{CHECKPOINTS[model]}"):
        snapshot_download(HF_REPO, allow_patterns=[CHECKPOINTS[model], f"{CHECKPOINTS[model]}/*"], local_dir=f"{util.BASE_DIR}/checkpoint")
    return getattr(util, f"load_{model}")()

def get_midi(model, length, prefix):
    buffer = util.generate_buffer(get_model(model), length, [int(x) for x in prefix.split()])
    with tempfile.NamedTemporaryFile(suffix=".midi", delete=False) as temp_file:
        temp_file.write(buffer.getvalue())
        return temp_file.name
