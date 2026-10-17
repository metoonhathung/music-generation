import torch
import torch.nn as nn
import torch.nn.functional as F
from app.model import MAX_LEN, pad_token
from app.model.transformer import MultiHeadAttention, AddNorm, PositionWiseFFN

mask_token = 388+3 # [MASK], one past the real vocabulary

class EncoderBlock(nn.Module):
  def __init__(self, d_model, d_k, ffn_l1_size, ffn_l2_size, num_heads,
             dropout, **kwargs):
    super(EncoderBlock, self).__init__()
    """
    Inputs:
      d_model: int, feature dimension of query/key/value
      d_k: int, feature projected dimension of query/key/value, we follow the setting in the paper, where d_v=d_k=d_q
      fnn_l1_size: int, feature dimension of the output after the first position-wise FFN.
      fnn_l2_size: int, feature dimension of the output after the second position-wise FFN.
      num_heads: int, number of head for multi-head attention layer.
      dropout: dropout probability for dropout layer.

    """
    self.attention = MultiHeadAttention(d_model, d_k, num_heads, dropout)
    self.addnorm_1 = AddNorm(dropout, d_model)
    self.ffn = PositionWiseFFN(d_model, ffn_l1_size, ffn_l2_size)
    self.addnorm_2 = AddNorm(dropout, d_model)

  def forward(self, X, valid_length):
    """
    Inputs:
      X: tensor of size (N, T, D), embedded input sequences
      valid_length: (N, ), the number of events in each sequence; the padding after them is never attended
    Outputs:
      Y: tensor of size (N, T, D_out)
    """
    # d2l's TransformerEncoderBlock: unlike DecoderBlock, every event attends to the whole sequence
    Y = self.addnorm_1(X, self.attention(X, X, X, valid_length))
    return self.addnorm_2(Y, self.ffn(Y))

class BERTEncoder(nn.Module):
  def __init__(self, vocab_size, d_model, ffn_l1_size, ffn_l2_size,
             num_heads, num_layers, dropout, max_len=MAX_LEN):
    super(BERTEncoder, self).__init__()
    """
    Inputs:
      d_model: int, feature dimension of query/key/value
      fnn_l1_size: int, feature dimension of the output after the first position-wise FFN.
      fnn_l2_size: int, feature dimension of the output after the second position-wise FFN.
      num_heads: int, number of head for multi-head attention layer.
      dropout: dropout probability for dropout layer.
      num_layers: number of encoder blocks
      max_len: the longest sequence it can read
    """
    # d2l's BERTEncoder without the segment embedding: a piece is one segment, not a sentence pair
    self.d_model = d_model
    self.token_embedding = nn.Embedding(vocab_size, d_model)
    self.layers = nn.ModuleList([EncoderBlock(d_model, d_model // num_heads, ffn_l1_size, ffn_l2_size, num_heads, dropout) for _ in range(num_layers)])
    # In BERT, positional embeddings are learnable
    self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model))

  def forward(self, X, valid_length):
    """
    Inputs:
      X: tensor of size (N, T), input tokens
      valid_length: (N, ), the number of events in each sequence
    Outputs:
      Y: tensor of size (N, T, d_model)
    """
    X = self.token_embedding(X)
    X = X + self.pos_embedding[:, :X.shape[1], :]
    for layer in self.layers:
      X = layer(X, valid_length)

    return X

class MaskLM(nn.Module):
  def __init__(self, vocab_size, d_model):
    super(MaskLM, self).__init__()
    """
    Inputs:
      vocab_size: int, the number of events in the vocabulary
      d_model: int, feature dimension of the encoder's output
    """
    self.mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.LayerNorm(d_model), nn.Linear(d_model, vocab_size))

  def forward(self, X):
    """
    Input:
      X: tensor of size (N, T, d_model), the encoder's output
    Output:
      Y: tensor of size (N, T, vocab_size)
    """
    # d2l gathers a fixed number of masked positions first; here that number varies from piece to piece,
    # so every position is scored and the loss keeps the masked ones.
    return self.mlp(X)

class Diffusion(nn.Module):
  """Masked discrete diffusion (MDLM) on a BERT: hide a random fraction of the events, learn to fill them in.

  Training is BERT's masked language modelling with a masking rate drawn per piece instead of a
  fixed 15%. Generation starts from a fully masked piece and reveals it over `steps` parallel
  passes instead of one event at a time. [MASK] events inside the primer are filled too, which
  gives infilling for free.
  """
  def __init__(self, encoder, vocab_size, **kwargs):
    super(Diffusion, self).__init__(**kwargs)
    self.encoder = encoder
    self.mlm = MaskLM(vocab_size, encoder.d_model)

  def logits(self, X, valid_len):
    preds = self.mlm(self.encoder(X, valid_len))
    preds[..., mask_token] = -float('inf') # [MASK] is only an input, never an event
    return preds

  def forward(self, tgt_array, tgt_valid_len):
    N, T = tgt_array.shape
    # One masking rate t per piece, spread evenly over (0, 1) so every batch mixes easy and hard cases
    # (MDLM's low-discrepancy sampler); the floor keeps its 1/t weight finite.
    t = ((torch.rand(1, device=tgt_array.device) + torch.arange(N, device=tgt_array.device) / N) % 1).clamp(min=1e-3)
    real = tgt_array != pad_token
    masked = real & (torch.rand(N, T, device=tgt_array.device) < t[:, None])
    preds = self.logits(torch.where(masked, mask_token, tgt_array), tgt_valid_len)

    # Cross-entropy on the hidden events only; weighting by 1/t makes it the MDLM bound on NLL per event.
    ce = F.cross_entropy(preds.transpose(1, 2), tgt_array, reduction='none')
    loss = (ce * masked / t[:, None]).sum() / real.sum()

    preds = preds.argmax(dim=-1)
    self.last_acc = ((preds == tgt_array) & masked).sum() / masked.sum()
    return loss, preds

  @torch.no_grad()
  def predict(self, tgt_array, tgt_valid_len, steps=100):
    N, T = tgt_array.shape
    x = torch.full((N, int(torch.max(tgt_valid_len))), mask_token, device=tgt_array.device)
    x[:, :T] = tgt_array[:, :x.shape[1]]

    # Longer than training: MAX_LEN windows, each continuing from the second half of the previous one.
    for start in range(0, max(1, x.shape[1] - MAX_LEN // 2), MAX_LEN // 2):
      w = x[:, start:start + MAX_LEN]
      valid_len = torch.full((N,), w.shape[1], device=w.device)
      for i in range(steps, 0, -1):
        masked = w == mask_token
        if not masked.any():
          break
        probs = F.softmax(self.logits(w, valid_len), dim=-1)
        sample = torch.multinomial(probs.reshape(-1, probs.shape[-1]), 1).reshape(masked.shape)
        # Linear schedule: with i steps left, each hidden event is revealed with probability 1/i.
        reveal = masked & (torch.rand(masked.shape, device=w.device) < 1 / i)
        w.copy_(torch.where(reveal, sample, w))

    return x[:, 1:]
