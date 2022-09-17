import os
import torch
import itertools
from io import BytesIO
from pathlib import Path
from typing import List, Literal
from pydantic import BaseModel
from app.processor import decode_midi
from app.model import device
from app.model.rnn import RNN
from app.model.cnn import WaveNet, CNN
from app.model.transformer import TransformerDecoder, Transformer
from app.model.vae import VAE
from app.model.gan import Generator

BASE_DIR = Path(__file__).resolve(strict=True).parent

class GenerateRequest(BaseModel):
    model: Literal['rnn', 'cnn', 'transformer', 'vae', 'gan']
    length: int
    prefix: List[int]

def generate_buffer(model, length, prefix):
    buffer = BytesIO()
    primer = torch.tensor([prefix]).to(device)
    valid_len = torch.full((1,), length).to(device)
    preds = model.predict(primer, valid_len)
    for i, pred in enumerate(preds):
        raw = (pred - 3).tolist()[:valid_len[i]]
        enc = list(itertools.takewhile(lambda x: x >= 0, raw))
        decode_midi(enc, buffer)
        buffer.seek(0)
    return buffer

def load_model(model, file_dir):
    if not os.path.exists(file_dir):
        return
    checkpoint = torch.load(file_dir, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f'{file_dir} loaded')
    model.eval()

def load_rnn():
    vocab_size = 388+3
    embedding_dim = 256
    hidden_size = 512
    num_layers = 3
    rnn_net = RNN(vocab_size, embedding_dim, hidden_size, num_layers).to(device)
    load_model(rnn_net, f'{BASE_DIR}/checkpoint/rnn.pt')
    return rnn_net

def load_cnn():
    vocab_size = 388+3
    embedding_dim = 256
    res_channels = 512
    dilation_depth = 10
    num_repeat = 1
    kernel_size = 2
    wave_net = WaveNet(vocab_size, embedding_dim, res_channels, dilation_depth, num_repeat, kernel_size)
    cnn_net = CNN(wave_net).to(device)
    load_model(cnn_net, f'{BASE_DIR}/checkpoint/cnn.pt')
    return cnn_net

def load_transformer():
    vocab_size = 388+3
    d_model = 256
    ffn_l1_size = 512
    ffn_l2_size = d_model
    num_heads = 8
    num_layers = 8
    dropout = 0.1
    decoder = TransformerDecoder(vocab_size, d_model, ffn_l1_size, ffn_l2_size, num_heads, num_layers, dropout, device=device)
    transformer_net = Transformer(decoder).to(device)
    load_model(transformer_net, f'{BASE_DIR}/checkpoint/transformer.pt')
    return transformer_net
    
def load_vae():
    vocab_size = 388+3
    embedding_dim = 256
    hidden_size = 512
    num_layers = 3
    latent_dim = 64
    vae_net = VAE(vocab_size, embedding_dim, hidden_size, num_layers, latent_dim).to(device)
    load_model(vae_net, f'{BASE_DIR}/checkpoint/vae.pt')
    return vae_net

def load_gan():
    vocab_size = 388+3
    embedding_dim = 256
    hidden_size = 512
    num_layers = 3
    latent_dim = 64
    g_net = Generator(vocab_size, embedding_dim, hidden_size, num_layers, latent_dim).to(device)
    load_model(g_net, f'{BASE_DIR}/checkpoint/generator.pt')
    return g_net
