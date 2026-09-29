import torch
import torch.nn as nn
import torch.nn.functional as F
from app.model import MAX_LEN, pad_token, bos_token, eos_token

mask_token = 388+3 # [MASK], one past the real vocabulary

class FullAttention(nn.Module):
  """DotProductAttention without the causal mask: every token attends to the whole piece."""
  def forward(self, query, key, value, valid_length=None):
    return F.scaled_dot_product_attention(query, key, value)

class Diffusion(nn.Module):
  """Masked discrete diffusion (MDLM): hide a random fraction of the tokens, learn to fill them in.

  The Transformer's decoder, made bidirectional by swapping in FullAttention. Generation starts
  from a fully masked piece and reveals it over `steps` parallel passes instead of one token at
  a time. [MASK] tokens inside the primer are filled too, which gives infilling for free.
  Then `refine` correction rounds (Mask-Predict, ReMDM) revisit the tokens guessed early, when
  almost nothing was revealed yet.
  """
  def __init__(self, decoder, **kwargs):
    super(Diffusion, self).__init__(**kwargs)
    self.decoder = decoder
    for layer in decoder.layers:
      layer.attention.attention = FullAttention()

  def logits(self, X):
    preds = self.decoder(X, None)
    preds[..., mask_token] = -float('inf') # never predict [MASK]
    return preds

  def forward(self, tgt_array, tgt_valid_len):
    N, T = tgt_array.shape
    # One masking rate t per piece, spread evenly over (0, 1] so every batch mixes easy and hard cases.
    t = ((torch.rand(1, device=tgt_array.device) + torch.arange(N, device=tgt_array.device) / N) % 1).clamp(min=1e-3)
    real = tgt_array != pad_token
    masked = real & (torch.rand(N, T, device=tgt_array.device) < t[:, None])
    preds = self.logits(torch.where(masked, mask_token, tgt_array))

    # Cross-entropy on the hidden tokens only; weighting by 1/t makes it the MDLM bound on NLL per token.
    ce = F.cross_entropy(preds.transpose(1, 2), tgt_array, reduction='none')
    loss = (ce * masked / t[:, None]).sum() / real.sum()

    preds = preds.argmax(dim=-1)
    self.last_acc = ((preds == tgt_array) & masked).sum() / masked.sum()
    return loss, preds

  @torch.no_grad()
  def predict(self, tgt_array, tgt_valid_len, steps=100, refine=300, temperature=0.95, top_k=40):
    N, T = tgt_array.shape
    x = torch.full((N, int(torch.max(tgt_valid_len))), mask_token, device=tgt_array.device)
    x[:, :T] = tgt_array[:, :x.shape[1]]

    # Longer than training: MAX_LEN windows, each continuing from the second half of the previous one.
    for start in range(0, max(1, x.shape[1] - MAX_LEN // 2), MAX_LEN // 2):
      w = x[:, start:start + MAX_LEN]
      free = w == mask_token # the tokens this window generates; the primer is never changed
      for i in range(steps, 0, -1):
        masked = w == mask_token
        if not masked.any():
          break
        logits = self.logits(w) / temperature
        logits[..., [pad_token, bos_token, eos_token]] = -float('inf')
        v, _ = logits.topk(top_k, dim=-1)
        logits[logits < v[..., -1:]] = -float('inf')
        sample = torch.multinomial(F.softmax(logits, dim=-1).reshape(-1, logits.shape[-1]), 1).reshape(masked.shape)
        # Linear schedule: with i steps left, each hidden token is revealed with probability 1/i.
        reveal = masked & (torch.rand(masked.shape, device=w.device) < 1 / i)
        w.copy_(torch.where(reveal, sample, w))

      # Correction rounds: re-cover 2% of the generated tokens and rewrite each with the model's best
      # guess given the other 98%, e.g. a note-off whose note was never switched on.
      for _ in range(refine):
        m = free & (torch.rand(w.shape, device=w.device) < 0.02)
        logits = self.logits(torch.where(m, mask_token, w))
        logits[..., [pad_token, bos_token, eos_token]] = -float('inf')
        w.copy_(torch.where(m, logits.argmax(dim=-1), w))

    return x[:, 1:]
