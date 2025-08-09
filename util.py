import os
import tempfile
import requests
import torch
from transformers import pipeline
from app.processor import decode_midi
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
pipe = pipeline("text-generation", model="metoonhathung/music-generation", device=device)

def get_midi_from_tfm(length, prefix):
    final_txt = prefix
    tmp_txt = prefix

    while len(final_txt.split(" ")) < length:
        res_txt = pipe(tmp_txt, num_return_sequences=1, max_new_tokens=256)[0]["generated_text"]
        final_txt = final_txt.replace(tmp_txt, res_txt)
        res_lst = [int(x) for x in res_txt.split(" ")[1:]]
        tmp_txt = " " + " ".join([str(x) for x in res_lst[-32:]])
        print("len =", len(final_txt.split(" ")))

    final_lst = [int(x) for x in final_txt.split(" ")[1:]]
    with tempfile.NamedTemporaryFile(suffix=".midi", delete=False) as temp_file:
        temp_file_path = temp_file.name
        decode_midi(final_lst, temp_file_path)
        return temp_file_path

def get_midi_from_torch(model, length, prefix):
    # url = "http://localhost/generate"
    url = "https://metoonhathung-music-generation-api-24psxym5la-uc.a.run.app/generate"
    headers = {
        "Content-Type": "application/json",
        "Accept": "audio/midi",
        "X-API-Key": os.environ["API_KEY"]
    }
    data = {
        "model": model,
        "length": length,
        "prefix": [int(x) for x in prefix.split(" ")]
    }
    response = requests.post(url, json=data, headers=headers)
    with tempfile.NamedTemporaryFile(suffix=".midi", delete=False) as temp_file:
        temp_file.write(response.content)
        temp_file_path = temp_file.name
        return temp_file_path