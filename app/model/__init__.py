import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 600
pad_token = 0
bos_token = 1
eos_token = 2
batch_size = 32
