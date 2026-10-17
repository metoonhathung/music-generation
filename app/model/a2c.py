import torch
import torch.nn as nn
import torch.nn.functional as F
from app.model import pad_token

class A2C(nn.Module):
  def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers):
    super(A2C, self).__init__()
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
    self.actor_fc = nn.Linear(hidden_size, vocab_size)
    self.critic_fc = nn.Linear(hidden_size, 1)

  def forward(self, target, valid_len):
    """Pure policy gradient from random init -- there is no MLE anywhere in this
    model. The policy samples an event and is rewarded only for matching the
    dataset's next event; it is never told what that event was.

    The GRU state is teacher-forced, so the next state does not depend on the
    action. Each position is therefore a contextual bandit rather than an MDP:
    gamma is 0 (a bootstrap term would be uncontrollable by the action and add
    only variance), and the critic acts as a pure per-state baseline.
    """
    o, _ = self.rnn(self.embedding(target))
    # one fused GRU call over the whole sequence; no per-timestep python loop
    logits = self.actor_fc(o[:, :-1])
    # logits (B, T-1, vocab_size) -- position t predicts target[:, t+1]
    values = self.critic_fc(o[:, :-1]).squeeze(-1)
    # values (B, T-1)
    tgt = target[:, 1:]
    mask = (tgt != pad_token).float()
    log_prob = F.log_softmax(logits, dim=-1)
    denom = mask.sum().clamp(min=1)

    with torch.no_grad():
      action = torch.multinomial(log_prob.exp().flatten(0, 1), 1).view(tgt.shape)
      reward = (action == tgt).float()
      advantage = reward - values

    lp_a = log_prob.gather(-1, action.unsqueeze(-1)).squeeze(-1)
    entropy = -(log_prob.exp() * log_prob).sum(-1)
    actor_loss = -(lp_a * advantage * mask).sum() / denom
    critic_loss = (((values - reward) ** 2) * mask).sum() / denom
    entropy_loss = (entropy * mask).sum() / denom
    # entropy coefficient 0.05, not 0.01: this objective's optimum is a single
    # deterministic action per state, so the bonus is the only thing keeping the
    # policy from collapsing. Measured -- 0.01 -> 0.115 nats (collapsed, loops),
    # 0.05 -> 0.349 nats, 0.2 -> 5.09 nats (near-uniform noise). Cliff is sharp
    # above 0.1: raise slightly if generation repeats, lower if it sounds random.
    loss = actor_loss + 0.5 * critic_loss - 0.05 * entropy_loss

    # The loss above is a policy-gradient surrogate: its VALUE is meaningless and
    # goes negative (the entropy bonus is subtracted). Accuracy is the real signal,
    # so expose it for train_net to print.
    preds = logits.argmax(dim=-1)
    self.last_acc = (((preds == tgt).float() * mask).sum() / denom).detach()

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
        pred = self.actor_fc(o)
        inputs = pred.argmax(dim=-1)
      preds.append(inputs)

    preds = torch.cat(preds, dim=1)
    return preds
