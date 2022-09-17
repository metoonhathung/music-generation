import torch
import torch.nn as nn
import torch.nn.functional as F
from app.model import pad_token

class CausalConv1d(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
    super(CausalConv1d, self).__init__()
    self.padding = (kernel_size - 1) * dilation
    self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=self.padding, dilation=dilation, groups=groups, bias=bias)

  def forward(self, inputs):
    out = self.conv1d(inputs)
    if self.padding != 0:
      return out[:, :, : -self.padding]
    return out

class ResBlock(nn.Module):
  def __init__(self, res_channels, skip_channels, kernel_size, dilation):
    super(ResBlock, self).__init__()
    self.res_channels = res_channels
    self.conv_dilated = CausalConv1d(res_channels, res_channels * 2, kernel_size, dilation=dilation)
    self.conv_res = CausalConv1d(res_channels, res_channels, 1)
    self.conv_skip = CausalConv1d(res_channels, skip_channels, 1)

  def forward(self, inputs):
    dilated = self.conv_dilated(inputs)
    dilated_split = torch.split(dilated, self.res_channels, dim=1)
    gated = torch.tanh(dilated_split[0]) * torch.sigmoid(dilated_split[1])
    out = self.conv_res(gated)
    out += inputs
    skip = self.conv_skip(gated)
    return out, skip

class ResStack(nn.Module):
  def __init__(self, res_channels, skip_channels, kernel_size, dilation_depth, num_repeat):
    super(ResStack, self).__init__()
    dilations = [2 ** d for d in range(dilation_depth)] * num_repeat
    self.res_blocks = nn.ModuleList([ResBlock(res_channels, skip_channels, kernel_size, d) for d in dilations])

  def forward(self, inputs):
    out = inputs
    skips = 0
    for res_block in self.res_blocks:
      out, skip = res_block(out)
      skips += skip
    return skips

class WaveNet(nn.Module):
  def __init__(self, vocab_size, embedding_dim, res_channels, dilation_depth, num_repeat, kernel_size):
    super(WaveNet, self).__init__()
    self.embedding = nn.Embedding(vocab_size, embedding_dim)
    self.causal = CausalConv1d(embedding_dim, res_channels, kernel_size)
    self.res_stack = ResStack(res_channels, embedding_dim, kernel_size, dilation_depth, num_repeat)
    self.relu1 = nn.ReLU()
    self.linear1 = nn.Conv1d(embedding_dim, vocab_size, 1)
    self.relu2 = nn.ReLU()
    self.linear2 = nn.Conv1d(vocab_size, vocab_size, 1)

  def forward(self, target, valid_len):
    embedded = self.embedding(target)
    embedded = embedded.permute(0,2,1)
    causal = self.causal(embedded)
    skips = self.res_stack(causal)
    linear = self.linear2(self.relu2(self.linear1(self.relu1(skips))))
    preds = linear.permute(0,2,1)
    return preds

class CNN(nn.Module):
  def __init__(self, cnn, **kwargs):
    super(CNN, self).__init__(**kwargs)
    self.cnn = cnn

  def forward(self, tgt_array, tgt_valid_len):
    loss = 0

    preds = self.cnn(tgt_array, tgt_valid_len)

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
      o = self.cnn(inputs, tgt_valid_len)
      if t+1 < T:
        output = tgt_array[:, t+1:t+2]
      else:
        output = o[:,-1:].argmax(dim=-1)
      outputs.append(output)
      inputs = torch.cat(outputs, dim=1)
      
    return inputs[:, 1:]
