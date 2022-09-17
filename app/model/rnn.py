import torch
import torch.nn as nn
import torch.nn.functional as F
from app.model import pad_token

class RNN(nn.Module):
  def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers):
    super(RNN, self).__init__()
    """
    inputs:
      vocab_size: int, the number of words in the vocabulary
      embedding_dim: int, dimension of the word embedding
      hidden_size: int, dimension of vallina RNN
    """
    self.hidden_size = hidden_size
    self.num_layers = num_layers

    self.embedding = nn.Embedding(vocab_size, embedding_dim)
    self.rnn = nn.GRU(embedding_dim, hidden_size, num_layers, batch_first=True)
    self.fc = nn.Linear(hidden_size, vocab_size)
    
  def forward(self, target, valid_len):
    loss = 0
    preds = []
      
    embedded = self.embedding(target)
    # embedded (N, T, embedding_dim)

    N, T = target.shape
    h = target.new_zeros(self.num_layers, N, self.hidden_size).float()

    for t in range(T-1):
      inputs = embedded[:, t].unsqueeze(1)
      # inputs (B, n, embedding_dim)
      o, h = self.rnn(inputs, h)
      # o (B, n, hidden_size)
      # h (num_layers, B, hidden_size)
      pred = self.fc(o)
      # pred (B, n, vocab_size)
      loss += F.nll_loss(F.log_softmax(pred[:, 0]), target[:, t+1], ignore_index=pad_token)
      preds.append(pred)
        
    preds = torch.cat(preds, dim=1).argmax(dim=-1)
    # preds (B, T) (32, 600)
    return loss, preds
  
  def predict(self, target, valid_len):
    N, T = target.shape
    h = target.new_zeros(self.num_layers, N, self.hidden_size).float()

    inputs = target[:, :1]
    preds = []

    for t in range(torch.max(valid_len)-1):
      inputs_embedded = self.embedding(inputs)
      o, h = self.rnn(inputs_embedded, h)
      if t+1 < T:
        inputs = target[:, t+1:t+2]
      else:
        pred = self.fc(o)
        inputs = pred.argmax(dim=-1)
      preds.append(inputs)
      
    preds = torch.cat(preds, dim=1)
    return preds
