import torch
import torch.nn as nn
import torch.nn.functional as F
from app.model import device, pad_token

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

class DotProductAttention(nn.Module): 
  def __init__(self):
      super(DotProductAttention, self).__init__()

  def forward(self, query, key, value, valid_length=None):
    """
    inputs:
      query: tensor of size (B, n, d)
      key: tensor of size (B, m, d)
      value: tensor of size (B, m, dim_v)
      valid_length: (B, )

      B is the batch_size, n is the number of queries, m is the number of <key, value> pairs,
      d is the feature dimension of the query, and dim_v is the feature dimension of the value.

    Outputs:
      attention: tensor of size (B, n, dim_v), weighted sum of values
    """
    d = key.shape[2]
    a = torch.bmm(query, key.permute(0,2,1))/(d ** 0.5)
    b = masked_softmax(a, valid_length)
    attention = torch.bmm(b, value)

    return attention

class MultiHeadAttention(nn.Module):
  def __init__(self, d_model, d_k, num_heads,  **kwargs):
    super(MultiHeadAttention, self).__init__()
    """
    Inputs:
      d_model: int, the same d_model in paper, feature dimension of query/key/values
      d_k: int, feature projected dimension of query/key/value, we follow the setting in the paper, where d_v=d_k=d_q
      num_heads: int, number of heads used for this MultiHeadAttention
    """
    self.num_heads = num_heads
    self.attention = DotProductAttention()
    self.W_q = nn.Linear(d_model, num_heads * d_k)
    self.W_k = nn.Linear(d_model, num_heads * d_k)
    self.W_v = nn.Linear(d_model, num_heads * d_k)
    self.W_o = nn.Linear(num_heads * d_k, d_model)

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
  def __init__(self, dim, device, max_len=100000):
    super(PositionalEncoding, self).__init__()
    """
    Inputs:
      dim: feature dimension of the positional encoding
    """
    self.pe = torch.zeros((1, max_len, dim), device=device)
    X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(10000, torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    self.pe[:, :, 0::2] = torch.sin(X)
    self.pe[:, :, 1::2] = torch.cos(X)


  def forward(self, X):
    """
    Inputs:
      X: tensor of size (N, T, D_in)
    Output:
      Y: tensor of the same size of X
    """
    N, T, D_in = X.shape
    Y = X + self.pe[:, :T, :]

    return Y

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
    self.attention = MultiHeadAttention(d_model, d_k, num_heads)
    self.addnorm_1 = AddNorm(dropout, d_model)
    self.ffn = PositionWiseFFN(d_model, ffn_l1_size, ffn_l2_size)
    self.addnorm_2 = AddNorm(dropout, d_model)

  def forward(self, X, valid_len):
    """
    Inputs:
      X: tensor of size (N, T, D), embedded input sequences
      **kwargs: other arguments you think is necessary for implementation
    Outputs:
      Y: tensor of size (N, T, D_out)
      
      Feel free to output variables if necessary.
    """
    N, T, D = X.shape
    if self.training:
      dec_valid_len = torch.arange(1, T+1).repeat(N, 1).to(device)
    else:
      dec_valid_len = torch.full((N,), T).to(device)
    X = self.addnorm_1(X, self.attention(X, X, X, dec_valid_len))
    Y = self.addnorm_2(X, self.ffn(X))

    return Y

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
    self.pos_enc = PositionalEncoding(d_model, device=device)
    self.layers = nn.ModuleList([DecoderBlock(d_model, d_model // num_heads, ffn_l1_size, ffn_l2_size, num_heads, dropout) for _ in range(num_layers)])
    self.dense = nn.Linear(d_model, vocab_size)


  def forward(self, X, valid_len):
    """
    Inputs:
      X: tensor of size (N, T, D), embedded input sequences
      valid_length: tensor of size (N,), valid lengths for each sequence
    """
    X = self.pos_enc(self.embedding(X) * (self.d_model ** 0.5))
    for layer in self.layers:
      X = layer(X, valid_len)
    Y = self.dense(X)
    
    return Y

class Transformer(nn.Module):
  """The base class for the encoder-decoder architecture."""
  def __init__(self, decoder, **kwargs):
    super(Transformer, self).__init__(**kwargs)
    self.decoder = decoder

  def forward(self, tgt_array, tgt_valid_len):
    """Forward function"""
    loss = 0

    preds = self.decoder(tgt_array, tgt_valid_len)

    T = tgt_array.shape[1]
    
    for t in range(T-1):
      loss += F.nll_loss(F.log_softmax(preds[:, t]), tgt_array[:, t+1], ignore_index=pad_token)

    preds = preds.argmax(dim=-1)
    
    return loss, preds
        
  def predict(self, tgt_array, tgt_valid_len):
    N, T = tgt_array.shape

    inputs = tgt_array[:, :1]
    outputs = [tgt_array[:, :1]]

    for t in range(torch.max(tgt_valid_len)-1):
      o = self.decoder(inputs, tgt_valid_len)
      if t+1 < T:
        output = tgt_array[:, t+1:t+2]
      else:
        output = o[:,-1:].argmax(dim=-1)
      outputs.append(output)
      inputs = torch.cat(outputs, dim=1)
      
    return inputs[:, 1:]
