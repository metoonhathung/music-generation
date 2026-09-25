import torch
import torch.nn as nn
import torch.nn.functional as F
from app.model import pad_token, bos_token, eos_token
from app.model.transformer import Transformer

class Discriminator(nn.Module):
  def __init__(self, decoder):
    super(Discriminator, self).__init__()
    """
    inputs:
      decoder: a TransformerDecoder, initialized from the trained Transformer
    """
    # The trained Transformer's layers, with the next-note output swapped for one
    # real-vs-generated score per note. It is causal, so the score at note t only sees
    # notes up to t -- which is what a per-note reward needs.
    self.decoder = decoder
    self.head = nn.Linear(decoder.d_model, 1)

  def forward(self, X):
    """
    Inputs:
      X: (N, T) tokens, real or generated
    Outputs:
      (N, T) per-note logits
    """
    dec = self.decoder
    H = dec.pos_enc(dec.embedding(X) * (dec.d_model ** 0.5))
    for layer in dec.layers:
      H = layer(H, None)
    return self.head(H).squeeze(-1)

class Generator(Transformer):
  """The trained Transformer, fine-tuned adversarially: MLE first, GAN second (SeqGAN).

  predict (T=0.95, top-40) is inherited; forward is the Transformer's MLE loss without label
  smoothing (below). MLE
  already learned the relations a GAN from scratch could not (note_on/note_off pairing,
  durations); the adversarial phase trains what teacher forcing never does -- the model
  continuing from its own samples. That shows most in raw sampling:
  predict(..., temperature=1.0, top_k=388+3).
  """
  def forward(self, tgt_array, tgt_valid_len):
    # Plain cross-entropy for the interleaved MLE step: the Transformer's label smoothing keeps
    # ~6% of probability outside its top-40 notes -- the stray notes D keeps punishing.
    preds = self.decoder(tgt_array, tgt_valid_len)
    loss = F.cross_entropy(preds[:, :-1].reshape(-1, preds.shape[-1]), tgt_array[:, 1:].reshape(-1),
                           ignore_index=pad_token)
    return loss, preds.argmax(dim=-1)

  def step(self, tok, t, cache):
    """Logits at position t, attending to the cached keys/values of positions < t.

    A key-value cache: without it every sampled token re-runs the whole prefix, and a
    600-token rollout costs quadratic time. Reuses the decoder's own layers (eval mode).
    """
    dec = self.decoder
    X = dec.embedding(tok)[:, None] * (dec.d_model ** 0.5) + dec.pos_enc.pe[:, t:t+1]
    for layer, (K, V) in zip(dec.layers, cache):
      att = layer.attention
      split = lambda Y: Y.view(len(tok), 1, att.num_heads, -1).transpose(1, 2) # (B, heads, 1, d_k)
      K[:, :, t:t+1], V[:, :, t:t+1] = split(att.W_k(X)), split(att.W_v(X))
      a = F.scaled_dot_product_attention(split(att.W_q(X)), K[:, :, :t+1], V[:, :, :t+1])
      X = layer.addnorm_1(X, att.W_o(a.transpose(1, 2).reshape(len(tok), 1, -1)))
      X = layer.addnorm_2(X, layer.ffn(X))
    return dec.dense(X[:, 0])

  @torch.no_grad()
  def sample(self, first, T):
    # Starts from a real token, so real and fake agree at position 0 (fakes that always
    # begin with bos let D win from position 0 alone).
    heads, d = self.decoder.layers[0].attention.num_heads, self.decoder.d_model
    empty = lambda: torch.zeros(len(first), heads, T, d // heads, device=first.device)
    cache = [(empty(), empty()) for _ in self.decoder.layers]
    seq = [first]
    for t in range(T - 1):
      logits = self.step(seq[-1], t, cache)
      logits[:, [pad_token, bos_token, eos_token]] = -float('inf')
      seq.append(torch.multinomial(F.softmax(logits, dim=-1), 1).squeeze(-1))
    return torch.stack(seq, dim=1)

  def log_probs(self, seq):
    # One parallel pass, instead of backpropagating through the sampling loop.
    return F.log_softmax(self.decoder(seq, None)[:, :-1], dim=-1).gather(-1, seq[:, 1:, None]).squeeze(-1)
