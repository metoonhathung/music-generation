import torch
from transformers import pipeline

class GPT2:
  """GPT-2 from huggingface_transformers.ipynb, with the same predict interface as the other models.

  GPT-2 was trained on raw event IDs (3 lower than the other models, no start token), so its
  output is shifted up by 3 to match what generate_buffer expects.
  """
  def __init__(self, model_dir, device):
    # float32: the saved weights are half precision, and half-precision math is slow on most CPUs
    self.pipe = pipeline("text-generation", model=model_dir, device=device, torch_dtype=torch.float32)

  def predict(self, target, valid_len):
    final_txt = " " + " ".join(str(x) for x in target[0].tolist())

    # Sliding window: continue from the last 128 tokens, so the prompt and the 128 new tokens fit in the
    # 256-token window GPT-2 was trained on.
    while len(final_txt.split()) < valid_len[0]:
      prompt = self.pipe.tokenizer.decode(self.pipe.tokenizer(final_txt)["input_ids"][-128:])
      final_txt += self.pipe(prompt, num_return_sequences=1, max_new_tokens=128, return_full_text=False)[0]["generated_text"]

    final_lst = [int(x) for x in final_txt.split()]
    return torch.tensor([final_lst]) + 3
