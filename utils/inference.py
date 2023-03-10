import os
from transformers import T5Tokenizer
from utils.model import TFT5ForConditionalGeneration

os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/lib/cuda/'

tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = TFT5ForConditionalGeneration.from_pretrained("t5-small")

input_s = ["translate English to German: The house is wonderful.", "I am happy because <extra_id_0>"]
input_ids = tokenizer(input_s, padding=True).input_ids
outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
