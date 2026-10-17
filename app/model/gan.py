import torch
import torch.nn as nn
from app.model import bos_token

class Discriminator(nn.Module):
  def __init__(self, encoder):
    super(Discriminator, self).__init__()
    """
    inputs:
      encoder: a BERTEncoder, initialized from the trained Diffusion model
    """
    # d2l's BERT classifier: the encoder, then its pooler (hidden) on the <cls> position and one
    # real-vs-generated logit in place of NextSentencePred's two classes.
    self.encoder = encoder
    self.hidden = nn.Sequential(nn.Linear(encoder.d_model, encoder.d_model), nn.Tanh())
    self.output = nn.Linear(encoder.d_model, 1)

  def forward(self, X):
    """
    Inputs:
      X: (N, T) windows of whole segments, real or generated
    Outputs:
      (N, ) one logit per window
    """
    # Real and generated windows share their first event, so it carries no signal; bos takes its
    # place as BERT's <cls>.
    X = torch.cat((torch.full_like(X[:, :1], bos_token), X[:, 1:]), dim=1)
    valid_len = torch.full((X.shape[0],), X.shape[1], device=X.device)
    H = self.encoder(X, valid_len)
    return self.output(self.hidden(H[:, 0])).squeeze(-1)
