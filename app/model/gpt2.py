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
    tmp_txt = final_txt

    while len(final_txt.split(" ")) < valid_len[0]:
      res_txt = self.pipe(tmp_txt, num_return_sequences=1, max_new_tokens=256)[0]["generated_text"]
      final_txt = final_txt.replace(tmp_txt, res_txt)
      res_lst = [int(x) for x in res_txt.split(" ")[1:]]
      tmp_txt = " " + " ".join([str(x) for x in res_lst[-32:]])

    final_lst = [int(x) for x in final_txt.split(" ")[1:]]
    return torch.tensor([final_lst]) + 3
