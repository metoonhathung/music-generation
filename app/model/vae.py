import torch
import torch.nn as nn
import torch.nn.functional as F
from app.model import device, pad_token

class VAEEncoder(nn.Module):
  def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, latent_dim):
    super(VAEEncoder, self).__init__()
    """
    inputs:
      vocab_size: int, the number of words in the vocabulary
      embedding_dim: int, dimension of the word embedding
      hidden_size: int, dimension of vallina RNN
    """
    self.embedding = nn.Embedding(vocab_size, embedding_dim)
    self.rnn = nn.GRU(embedding_dim, hidden_size, num_layers, batch_first=True, bidirectional=True)
    self.hid2mu = nn.Linear(2*hidden_size, latent_dim)
    self.hid2logvar = nn.Linear(2*hidden_size, latent_dim)
    self.kld = 0

  def forward(self, src, src_len):
    """
    Inputs:
      source: tensor of size (N, T), where N is the batch size, T is the length of the sequence(s)
      valid_len: tensor of size (N,), indicating the valid length of sequence(s) (the length before padding)
    """
    embedded = self.embedding(src)
    packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, src_len.to('cpu'), batch_first=True, enforce_sorted=False)
    packed_outputs, hidden = self.rnn(packed_embedded)
    outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
    hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
    mu = self.hid2mu(hidden)
    logvar = self.hid2logvar(hidden)
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    z = mu + (eps * std)
    self.kld = -0.5*torch.sum(logvar-mu.pow(2)-logvar.exp()+1, dim=1).mean()
    return z

class VAEDecoder(nn.Module):
  def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, latent_dim):
    super(VAEDecoder, self).__init__()
    """
    inputs:
      vocab_size: int, the number of words in the vocabulary
      embedding_dim: int, dimension of the word embedding
      hidden_size: int, dimension of the hidden state of vanilla RNN
    """
    self.embedding = nn.Embedding(vocab_size, embedding_dim)
    self.rnn = nn.GRU(embedding_dim+latent_dim, hidden_size, num_layers, batch_first=True)
    self.fc = nn.Linear(hidden_size, vocab_size)
    
  def forward(self, z, target, h=None):
    embedded = self.embedding(target)
    N, T, _ = embedded.shape

    zs = torch.cat([z]*T, dim=1).view(N, T, -1)
    concat = torch.cat([embedded, zs], dim=2)
    
    o, h = self.rnn(concat, h)
    pred = self.fc(o)
    return pred, h

class VAE(nn.Module):
  def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, latent_dim):
    super(VAE, self).__init__()
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.latent_dim = latent_dim

    self.encoder = VAEEncoder(vocab_size, embedding_dim, hidden_size, num_layers, latent_dim)
    self.decoder = VAEDecoder(vocab_size, embedding_dim, hidden_size, num_layers, latent_dim)
        
  def forward(self, tgt_array, tgt_valid_len):
    rec_loss = 0

    z = self.encoder(tgt_array, tgt_valid_len)
    preds, _ = self.decoder(z, tgt_array)

    T = tgt_array.shape[1]
    
    for t in range(T-1):
      rec_loss += F.nll_loss(F.log_softmax(preds[:, t]), tgt_array[:, t+1], ignore_index=pad_token)
    
    elbo = rec_loss + self.encoder.kld
    preds = preds.argmax(dim=-1)
    
    return elbo, preds

  def predict(self, target, valid_len):
    N, T = target.shape
    h = target.new_zeros(self.num_layers, N, self.hidden_size).float()
    z = torch.randn(N, self.latent_dim).to(device)

    inputs = target[:, :1]
    preds = []

    for t in range(torch.max(valid_len)-1):
      pred, h = self.decoder(z, inputs, h)
      if t+1 < T:
        inputs = target[:, t+1:t+2]
      else:
        inputs = pred.argmax(dim=-1)
      preds.append(inputs)
      
    preds = torch.cat(preds, dim=1)
    return preds
