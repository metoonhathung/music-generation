import torch
import torch.nn as nn
import torch.nn.functional as F
from app.model import pad_token

def masked_softmax(X, valid_length):
  """
  inputs:
    X: 3-D tensor
    valid_length: 1-D or 2-D tensor
  """
  mask_value = -1e7

  if len(X.shape) == 2:
    X = X.unsqueeze(1)

  N, n, m = X.shape

  if len(valid_length.shape) == 1:
    valid_length = valid_length.repeat_interleave(n, dim=0)
  else:
    valid_length = valid_length.reshape((-1,))

  mask = torch.arange(m)[None, :].to(X.device) >= valid_length[:, None]
  X.view(-1, m)[mask] = mask_value

  Y = torch.softmax(X, dim=-1)


  return Y

# FAST_ATTENTION computes exactly the masked softmax below (same mask, same dropout) with PyTorch's
# fused kernel: 2-4x faster, and it never materialises the (B*heads, T, T) score tensor. Set it to
# False to step through the explicit maths.
FAST_ATTENTION = True

class DotProductAttention(nn.Module):
  def __init__(self, dropout):
      super(DotProductAttention, self).__init__()
      self.dropout = nn.Dropout(dropout)

  def forward(self, query, key, value, valid_length=None):
    """
    inputs:
      query: tensor of size (B, n, d)
      key: tensor of size (B, m, d)
      value: tensor of size (B, m, dim_v)
      valid_length: (B, ) or (B, n)

      B is the batch_size, n is the number of queries, m is the number of <key, value> pairs,
      d is the feature dimension of the query, and dim_v is the feature dimension of the value.

    Outputs:
      attention: tensor of size (B, n, dim_v), weighted sum of values
    """
    if FAST_ATTENTION:
      lens = valid_length if valid_length.dim() == 2 else valid_length[:, None]
      mask = torch.arange(key.shape[1], device=key.device) < lens[..., None]
      return F.scaled_dot_product_attention(query, key, value, attn_mask=mask,
                                            dropout_p=self.dropout.p if self.training else 0)

    d = key.shape[2]
    a = torch.bmm(query, key.permute(0,2,1))/(d ** 0.5)
    b = masked_softmax(a, valid_length)
    attention = torch.bmm(self.dropout(b), value)

    return attention

class MultiHeadAttention(nn.Module):
  def __init__(self, d_model, d_k, num_heads, dropout, **kwargs):
    super(MultiHeadAttention, self).__init__()
    """
    Inputs:
      d_model: int, the same d_model in paper, feature dimension of query/key/values
      d_k: int, feature projected dimension of query/key/value, we follow the setting in the paper, where d_v=d_k=d_q
      num_heads: int, number of heads used for this MultiHeadAttention
      dropout: dropout probability for the attention weights
    """
    self.num_heads = num_heads
    self.attention = DotProductAttention(dropout)
    self.W_q = nn.Linear(d_model, num_heads * d_k, bias=False)
    self.W_k = nn.Linear(d_model, num_heads * d_k, bias=False)
    self.W_v = nn.Linear(d_model, num_heads * d_k, bias=False)
    self.W_o = nn.Linear(num_heads * d_k, d_model, bias=False)

  def forward(self, query, key, value, valid_length):
    """
    inputs:
      query: tensor of size (B, T, d_model)
      key: tensor of size (B, T, d_model)
      value: tensor of size (B, T, d_model)
      valid_length: (B, )

      B is the batch_size, T is length of sequence, d_model is the feature dimensions of query,
      key, and value.

    Outputs:
      attention (B, T, d_model)
      """
    query = self.W_q(query)
    # query (B, T_q, num_heads * d_k)
    key = self.W_k(key)
    # key (B, T, num_heads * d_k)
    value = self.W_v(value)
    # value (B, T, num_heads * d_k)

    B, T, num_hiddens = key.shape
    _, T_q, _ = query.shape
    d_k = num_hiddens // self.num_heads

    query = query.reshape(B, T_q, self.num_heads, -1).permute(0,2,1,3).reshape(-1, T_q, d_k)
    # query (B * num_heads, T_q, d_k)
    key = key.reshape(B, T, self.num_heads, -1).permute(0,2,1,3).reshape(-1, T, d_k)
    # key (B * num_heads, T, d_k)
    value = value.reshape(B, T, self.num_heads, -1).permute(0,2,1,3).reshape(-1, T, d_k)
    # value (B * num_heads, T, d_k)

    valid_length = torch.repeat_interleave(valid_length, repeats=self.num_heads, dim=0)

    attention = self.attention(query, key, value, valid_length)
    # attention (B * num_heads, T_q, d_k)
    attention = attention.reshape(-1, self.num_heads, T_q, d_k).permute(0,2,1,3).reshape(B, T_q, -1)
    # attention (B, T_q, num_heads * d_k)
    attention = self.W_o(attention)
    # attention (B, T_q, d_model)

    return attention

class PositionWiseFFN(nn.Module):
  def __init__(self, input_size, ffn_l1_size, ffn_l2_size):
    super(PositionWiseFFN, self).__init__()
    """
    Inputs:
      input_size: int, feature dimension of the input
      fnn_l1_size: int, feature dimension of the output after the first position-wise FFN.
      fnn_l2_size: int, feature dimension of the output after the second position-wise FFN.
    """
    self.ffn_l1 = nn.Linear(input_size, ffn_l1_size)
    self.relu = nn.ReLU()
    self.ffn_l2 = nn.Linear(ffn_l1_size, ffn_l2_size)

  def forward(self, X):
    """
    Input:
      X: tensor of size (N, T, D_in)
    Output:
      o: tensor of size (N, T, D_out)
    """
    o = self.ffn_l2(self.relu(self.ffn_l1(X)))
    return o

class PositionalEncoding(nn.Module):
  def __init__(self, dim, dropout, device, max_len=2048):
    super(PositionalEncoding, self).__init__()
    """
    Inputs:
      dim: feature dimension of the positional encoding
      dropout: dropout probability, applied after adding the encoding
    """
    self.dropout = nn.Dropout(dropout)
    self.pe = torch.zeros((1, max_len, dim), device=device)
    X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(10000, torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    self.pe[:, :, 0::2] = torch.sin(X)
    self.pe[:, :, 1::2] = torch.cos(X)


  def forward(self, X, offset=0):
    """
    Inputs:
      X: tensor of size (N, T, D_in)
      offset: position of X's first step. During generation it is the number of steps already
        cached; d2l always starts at row 0, which would give every generated event position 0.
    Output:
      Y: tensor of the same size of X
    """
    N, T, D_in = X.shape
    Y = X + self.pe[:, offset:offset+T, :]

    return self.dropout(Y)

class AddNorm(nn.Module):
    def __init__(self, dropout, embedding_size):
        super(AddNorm, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embedding_size)

    def forward(self, X, Y):
        return self.norm(self.dropout(Y) + X)

class DecoderBlock(nn.Module):
  def __init__(self, d_model, d_k, ffn_l1_size, ffn_l2_size, num_heads,
             dropout, **kwargs):
    super(DecoderBlock, self).__init__()
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

  def forward(self, X, state):
    """
    Inputs:
      X: tensor of size (N, T, D), embedded input sequences
      state: this block's inputs from earlier generation steps (d2l's key_values cache), or None
    Outputs:
      Y: tensor of size (N, T, D_out), and the updated state
    """
    N, T, D = X.shape
    key_values = X if state is None else torch.cat((state, X), dim=1)
    # Query i sits at position offset + i and attends keys 0..offset+i. Without a cache this is d2l's
    # causal arange(1, T+1); for one new event it is every cached key, as d2l's None.
    offset = key_values.shape[1] - T
    dec_valid_len = torch.arange(offset + 1, offset + T + 1, device=X.device).repeat(N, 1)
    X = self.addnorm_1(X, self.attention(X, key_values, key_values, dec_valid_len))
    Y = self.addnorm_2(X, self.ffn(X))

    return Y, key_values

class TransformerDecoder(nn.Module):
  def __init__(self, vocab_size, d_model, ffn_l1_size, ffn_l2_size,
             num_heads, num_layers, dropout, device):
    super(TransformerDecoder, self).__init__()
    """
    Inputs:
      d_model: int, feature dimension of query/key/value
      fnn_l1_size: int, feature dimension of the output after the first position-wise FFN.
      fnn_l2_size: int, feature dimension of the output after the second position-wise FFN.
      num_heads: int, number of head for multi-head attention layer.
      dropout: dropout probability for dropout layer.
      num_layers: number of decoder blocks
    """
    self.d_model = d_model
    self.embedding = nn.Embedding(vocab_size, d_model)
    self.pos_enc = PositionalEncoding(d_model, dropout, device=device)
    self.layers = nn.ModuleList([DecoderBlock(d_model, d_model // num_heads, ffn_l1_size, ffn_l2_size, num_heads, dropout) for _ in range(num_layers)])
    self.dense = nn.Linear(d_model, vocab_size)

  def init_state(self):
    return [None] * len(self.layers)

  def forward(self, X, state=None):
    """
    Inputs:
      X: tensor of size (N, T), input tokens
      state: the blocks' caches from init_state() or an earlier call, or None for a new sequence
    Outputs:
      Y: tensor of size (N, T, vocab_size), and the updated state
    """
    state = self.init_state() if state is None else state
    offset = 0 if state[0] is None else state[0].shape[1]
    X = self.pos_enc(self.embedding(X) * (self.d_model ** 0.5), offset)
    for i, layer in enumerate(self.layers):
      X, state[i] = layer(X, state[i])
    Y = self.dense(X)

    return Y, state

class Transformer(nn.Module):
  """The base class for the encoder-decoder architecture."""
  def __init__(self, decoder, **kwargs):
    super(Transformer, self).__init__(**kwargs)
    self.decoder = decoder

  def forward(self, tgt_array, tgt_valid_len):
    """Forward function"""
    preds, _ = self.decoder(tgt_array)

    # d2l's masked cross-entropy: the mean over every next event that isn't padding
    loss = F.cross_entropy(preds[:, :-1].reshape(-1, preds.shape[-1]), tgt_array[:, 1:].reshape(-1),
                           ignore_index=pad_token)

    preds = preds.argmax(dim=-1)

    return loss, preds

  @torch.no_grad()
  def predict(self, tgt_array, tgt_valid_len):
    N, T = tgt_array.shape

    state = self.decoder.init_state()
    outputs = [tgt_array[:, :1]]

    for t in range(torch.max(tgt_valid_len)-1):
      # d2l's predict_step: feed only the newest event; the earlier ones are cached in state
      o, state = self.decoder(outputs[-1], state)
      if t+1 < T:
        output = tgt_array[:, t+1:t+2]
      else:
        # sample from the model's distribution; d2l's greedy argmax loops on the same few notes
        output = torch.multinomial(F.softmax(o[:, -1], dim=-1), 1)
      outputs.append(output)

    return torch.cat(outputs, dim=1)[:, 1:]
