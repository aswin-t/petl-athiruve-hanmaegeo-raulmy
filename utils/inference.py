import os
from transformers import T5Tokenizer
from utils.model import TFPromptT5ForConditionalGeneration

os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/lib/cuda/'

tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = TFPromptT5ForConditionalGeneration.from_pretrained("t5-small")

input_ids = tokenizer("translate English to German: The house is wonderful.", return_tensors="pt").input_ids
outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
