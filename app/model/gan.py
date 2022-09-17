import torch
import torch.nn as nn
import torch.nn.functional as F
from app.model import device, bos_token, MAX_LEN

class Discriminator(nn.Module):
  def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, dense_size):
    super(Discriminator, self).__init__()
    """
    inputs:
      vocab_size: int, the number of words in the vocabulary
      embedding_dim: int, dimension of the word embedding
      hidden_size: int, dimension of vallina RNN
    """
    self.embedding = nn.Embedding(vocab_size, embedding_dim)
    self.rnn = nn.GRU(embedding_dim, hidden_size, num_layers, batch_first=True, bidirectional=True)
    self.fc1 = nn.Linear(2*hidden_size, dense_size)
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout()
    self.fc2 = nn.Linear(dense_size, 1)

  def forward(self, src):
    """
    Inputs:
      source: tensor of size (N, T), where N is the batch size, T is the length of the sequence(s)
      valid_len: tensor of size (N,), indicating the valid length of sequence(s) (the length before padding)
    """
    embedded = self.embedding(src)
    outputs, hidden = self.rnn(embedded)
    h = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
    o = self.fc2(self.dropout(self.relu(self.fc1(h))))
    return o, h

class Generator(nn.Module):
  def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, latent_dim):
    super(Generator, self).__init__()
    """
    inputs:
      vocab_size: int, the number of words in the vocabulary
      embedding_dim: int, dimension of the word embedding
      hidden_size: int, dimension of the hidden state of vanilla RNN
    """
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.latent_dim = latent_dim

    self.embedding = nn.Embedding(vocab_size, embedding_dim)
    self.rnn = nn.GRU(embedding_dim+latent_dim, hidden_size, num_layers, batch_first=True)
    self.fc = nn.Linear(hidden_size, vocab_size)
    
  def forward(self, z):
    N, _ = z.shape
    primer = z.new_full((N, 1), bos_token).int()
    h = z.new_zeros(self.num_layers, N, self.hidden_size).float()

    inputs = primer
    preds = [primer]
    
    for t in range(MAX_LEN-1):
      inputs_embedded = self.embedding(inputs)
      concat = torch.cat((inputs_embedded, z.unsqueeze(1)), dim=2)
      o, h = self.rnn(concat, h)
      pred = self.fc(o)
      inputs = pred.argmax(dim=-1)
      preds.append(inputs)
    
    preds = torch.cat(preds, dim=1)
    return preds

  def predict(self, target, valid_len):
    N, T = target.shape
    h = target.new_zeros(self.num_layers, N, self.hidden_size).float()
    z = torch.randn(N, self.latent_dim).to(device)

    inputs = target[:, :1]
    preds = []

    for t in range(torch.max(valid_len)-1):
      inputs_embedded = self.embedding(inputs)
      concat = torch.cat((inputs_embedded, z.unsqueeze(1)), dim=2)
      o, h = self.rnn(concat, h)
      pred = self.fc(o)
      if t+1 < T:
        inputs = target[:, t+1:t+2]
      else:
        inputs = pred.argmax(dim=-1)
      preds.append(inputs)
      
    preds = torch.cat(preds, dim=1)
    return preds
